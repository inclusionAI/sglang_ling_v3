#!/usr/bin/env python3
"""Image-text serving performance benchmark for Bailing-V3-VL (BailingMoeV3VL).

Sends synthetic (image + text) requests to a running sglang server and reports
TTFT / ITL / TPOT / end-to-end latency and throughput at one or more concurrency
levels. Decode length is pinned with ``ignore_eos`` so every request produces the
same number of output tokens -> deterministic, comparable load.

Why a dedicated client instead of ``sglang.bench_serving --dataset-name image``?
The Bailing-V3-VL checkpoints ship a chat template that omits role markers and the
image placeholder, so the OpenAI-chat path drops images. This client builds the
authoritative prompt directly with the canonical Bailing VL image placeholder:

    <role>SYSTEM</role>{sys}<|role_end|><role>HUMAN</role><|vision_start|><|image_pad|><|vision_end|>...{q}<|role_end|><role>ASSISTANT</role>

and posts to /generate with image_data, which is the robust path on the current
server build.

Usage (server must already be running):
    python bench_bailing_vl_image.py --host 127.0.0.1 --port 31013 \
        --num-prompts 64 --concurrency 1 8 32 \
        --image-resolution 720p --image-count 1 --output-len 64
"""

import argparse
import asyncio
import base64
import io
import json
import time
from dataclasses import dataclass, field
from typing import List

import aiohttp
import numpy as np
from PIL import Image

EOS = "<|role_end|>"
USER = "<role>HUMAN</role>"
ASST = "<role>ASSISTANT</role>"
SYS_NOTHINK = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
SYS_THINK = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking on"

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

RES_PRESETS = {
    "4k": (3840, 2160),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "360p": (640, 360),
}


def parse_resolution(s: str):
    if s in RES_PRESETS:
        return RES_PRESETS[s]
    if "x" in s:
        w, h = s.lower().split("x", 1)
        return int(w), int(h)
    raise ValueError(f"bad resolution {s}")


def make_image_b64(w, h, fmt="jpeg", seed=0):
    rng = np.random.RandomState(seed)
    arr = (rng.rand(h, w, 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt.upper())
    mime = "jpeg" if fmt.lower() in ("jpg", "jpeg") else fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def build_prompt(question, n_images, think):
    sysp = SYS_THINK if think else SYS_NOTHINK
    return sysp + EOS + USER + (IMAGE_PLACEHOLDER * n_images) + question + EOS + ASST


@dataclass
class ReqResult:
    ok: bool = False
    ttft: float = 0.0
    latency: float = 0.0
    out_tokens: int = 0
    prompt_tokens: int = 0
    itls: List[float] = field(default_factory=list)
    error: str = ""


async def one_request(session, url, prompt, images, output_len, sem) -> ReqResult:
    payload = {
        "text": prompt,
        "image_data": images,
        "sampling_params": {
            "max_new_tokens": output_len,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": True,
    }
    res = ReqResult()
    async with sem:
        st = time.perf_counter()
        last = st
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    res.error = f"HTTP {resp.status}: {(await resp.text())[:120]}"
                    return res
                async for raw in resp.content:
                    chunk = raw.strip()
                    if not chunk or not chunk.startswith(b"data:"):
                        continue
                    data = chunk[len(b"data:") :].strip()
                    if data == b"[DONE]":
                        continue
                    now = time.perf_counter()
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    meta = obj.get("meta_info", {}) or {}
                    ctok = meta.get("completion_tokens", 0) or 0
                    if res.ttft == 0.0:
                        res.ttft = now - st
                    else:
                        res.itls.append(now - last)
                    last = now
                    res.out_tokens = ctok
                    res.prompt_tokens = meta.get("prompt_tokens", 0) or 0
                res.latency = time.perf_counter() - st
                res.ok = res.out_tokens > 0
        except Exception as e:  # noqa
            res.error = repr(e)[:160]
    return res


async def run_level(base_url, prompts, images_per, output_len, concurrency):
    url = base_url + "/generate"
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        t0 = time.perf_counter()
        tasks = [
            one_request(session, url, p, imgs, output_len, sem)
            for p, imgs in zip(prompts, images_per)
        ]
        results = await asyncio.gather(*tasks)
        dur = time.perf_counter() - t0
    return results, dur


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def report(tag, results, dur, concurrency):
    ok = [r for r in results if r.ok]
    n = len(ok)
    fail = len(results) - n
    ttfts = [r.ttft * 1000 for r in ok]
    lats = [r.latency * 1000 for r in ok]
    all_itls = [x * 1000 for r in ok for x in r.itls]
    out_tok = sum(r.out_tokens for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)
    print(f"\n========= concurrency={concurrency}  ({tag}) =========")
    print(f"  successful / failed     : {n} / {fail}")
    if fail and n == 0:
        print("  (all failed) first error:", results[0].error)
        return
    print(f"  duration (s)            : {dur:.2f}")
    print(f"  request throughput      : {n / dur:.2f} req/s")
    print(f"  output token throughput : {out_tok / dur:.1f} tok/s")
    print(f"  total token throughput  : {(in_tok + out_tok) / dur:.1f} tok/s")
    print(f"  mean prompt tokens/req  : {in_tok / max(n,1):.0f}  (incl. vision)")
    print(f"  mean output tokens/req  : {out_tok / max(n,1):.0f}")
    print(
        f"  TTFT   ms  mean/p50/p99 : {np.mean(ttfts):.1f} / {pct(ttfts,50):.1f} / {pct(ttfts,99):.1f}"
    )
    print(
        f"  E2E    ms  mean/p50/p99 : {np.mean(lats):.1f} / {pct(lats,50):.1f} / {pct(lats,99):.1f}"
    )
    if all_itls:
        print(
            f"  ITL    ms  mean/p50/p99 : {np.mean(all_itls):.2f} / {pct(all_itls,50):.2f} / {pct(all_itls,99):.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=31013)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--image-resolution", default="720p")
    ap.add_argument("--image-count", type=int, default=1)
    ap.add_argument("--image-format", default="jpeg")
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--question", default="Describe this image in detail.")
    ap.add_argument("--think", action="store_true")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    w, h = parse_resolution(args.image_resolution)
    tag = f"{args.image_count}x{args.image_resolution} out={args.output_len} think={args.think}"

    # Pre-build prompts and images (distinct seeds -> distinct images, no cache hits)
    prompt = build_prompt(args.question, args.image_count, args.think)
    prompts = [prompt] * args.num_prompts
    images_per = []
    for i in range(args.num_prompts):
        imgs = [
            make_image_b64(w, h, args.image_format, seed=i * args.image_count + j)
            for j in range(args.image_count)
        ]
        images_per.append(imgs)

    print(f"[bench] server={base_url} model=Bailing-V3-VL")
    print(
        f"[bench] {args.num_prompts} prompts, {args.image_count}x {args.image_resolution} "
        f"({w}x{h}) {args.image_format}, output_len={args.output_len} (ignore_eos), "
        f"think={args.think}"
    )

    # warmup
    asyncio.run(run_level(base_url, prompts[:1], images_per[:1], args.output_len, 1))

    for c in args.concurrency:
        results, dur = asyncio.run(
            run_level(base_url, prompts, images_per, args.output_len, c)
        )
        report(tag, results, dur, c)


if __name__ == "__main__":
    main()
