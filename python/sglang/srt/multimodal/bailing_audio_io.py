# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Native-rate decoding for Bailing's device-side audio preprocessing."""

import io
import logging
import os
from urllib.parse import unquote, urlparse

import numpy as np
import pybase64

from sglang.srt.multimodal.audio_from_video import is_audio_container
from sglang.srt.utils.common import download_remote_media
from sglang.srt.utils.video_decoder import _BACKEND

logger = logging.getLogger(__name__)
_INVALID_CONTAINER = (
    "Invalid input_audio: no decodable audio stream was found in the media container."
)


class _BailingAudioDecodeError(ValueError):
    pass


def _decode_native_container(source: bytes | str) -> tuple[np.ndarray, int]:
    """Decode the first audio stream without resampling or downmixing."""
    try:
        import av

        input_source = io.BytesIO(source) if isinstance(source, bytes) else source
        with av.open(input_source) as container:
            if not container.streams.audio:
                raise _BailingAudioDecodeError(_INVALID_CONTAINER)
            native_sr = None
            resampler = None
            chunks = []
            skipped_packets = 0
            first_decode_error = None
            for packet in container.demux(container.streams.audio[0]):
                try:
                    for frame in packet.decode():
                        if native_sr is None:
                            native_sr = frame.sample_rate
                            # Convert sample format only; preserve rate and channels.
                            resampler = av.audio.resampler.AudioResampler(
                                format="fltp", layout=frame.layout, rate=native_sr
                            )
                        elif frame.sample_rate != native_sr:
                            raise _BailingAudioDecodeError(
                                "Audio sample rate changed within the stream"
                            )
                        chunks.extend(
                            item.to_ndarray().T for item in resampler.resample(frame)
                        )
                except av.error.FFmpegError as error:
                    skipped_packets += 1
                    if first_decode_error is None:
                        first_decode_error = error
            if resampler is not None:
                chunks.extend(item.to_ndarray().T for item in resampler.resample(None))
            if skipped_packets:
                logger.warning(
                    "Skipped %d undecodable audio packet(s); kept %d decoded "
                    "chunk(s). First decode error: %s",
                    skipped_packets,
                    len(chunks),
                    first_decode_error,
                )
    except _BailingAudioDecodeError:
        raise
    except Exception as error:
        raise ValueError(_INVALID_CONTAINER) from error

    if not chunks:
        raise ValueError(_INVALID_CONTAINER) from first_decode_error
    waveform = np.concatenate(chunks, axis=0)
    if waveform.ndim != 2 or waveform.size == 0:
        raise ValueError(_INVALID_CONTAINER)
    return np.ascontiguousarray(waveform, dtype=np.float32), int(native_sr)


def load_bailing_audio_native(audio_file: str | bytes) -> tuple[np.ndarray, int]:
    """Return original-rate audio and its sample rate, preserving all channels.

    Multi-channel arrays use (samples, channels), matching the shared loader.
    Bailing's frontend owns the subsequent downmix and device-side resampling.
    """
    if isinstance(audio_file, bytes):
        source = audio_file
    elif isinstance(audio_file, str) and audio_file.startswith("data:"):
        source = pybase64.b64decode(audio_file.split(",")[1], validate=True)
    elif isinstance(audio_file, str) and audio_file.startswith(("http://", "https://")):
        source = download_remote_media(
            audio_file, timeout=int(os.getenv("REQUEST_TIMEOUT", "5"))
        )
    elif isinstance(audio_file, str) and audio_file.startswith("file://"):
        source = unquote(urlparse(audio_file).path)
    elif isinstance(audio_file, str):
        source = audio_file
    else:
        raise ValueError(f"Invalid audio format: {audio_file}")

    if isinstance(source, bytes):
        header = source[:16]
    else:
        with open(source, "rb") as audio_stream:
            header = audio_stream.read(16)
    if is_audio_container(header):
        return _decode_native_container(source)

    if _BACKEND == "torchcodec":
        from torchcodec.decoders import AudioDecoder

        try:
            decoder = AudioDecoder(source, sample_rate=None, num_channels=None)
            samples = decoder.get_all_samples()
            return samples.data.T.numpy(), int(samples.sample_rate)
        except Exception as error:
            logger.warning(
                "torchcodec AudioDecoder failed (%s); falling back to soundfile.",
                error,
            )

    import soundfile as sf

    try:
        audio, sample_rate = sf.read(
            io.BytesIO(source) if isinstance(source, bytes) else source
        )
    except sf.LibsndfileError as error:
        raise ValueError(f"Could not decode audio: {error}") from error
    return audio, int(sample_rate)
