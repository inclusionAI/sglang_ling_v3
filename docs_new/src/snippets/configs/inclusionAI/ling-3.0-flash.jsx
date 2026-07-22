export const config = {
  modelName: "Ling-3.0-flash",

  supportedHardware: ["h20-3e", "h200", "h800", "h100", "b200"],

  // Model-specific GPUs the shared HARDWARE_CATALOG doesn't carry — the engine
  // merges these in (h100 / h200 / b200 come from the catalog).
  hardware: [
    { id: "h20-3e", label: "H20-3e", vram: "141 GB", vendor: "nvidia" },
    { id: "h800", label: "H800", vram: "80 GB", vendor: "nvidia" },
  ],

  variants: [
    { id: "default", label: "Ling-3.0-flash" },
  ],
  quantizations: [
    { id: "bf16", label: "BF16" },
  ],
  strategies: [
    { id: "balanced", label: "Balanced" },
  ],
  nodesOptions: [
    { id: "single", label: "Single Node" },
  ],

  modelNames: {
    "default|bf16": "inclusionAI/Ling-3.0-flash",
  },

  placeholders: {
    HOST_IP:   { target: "command", label: "Bind host",   default: "0.0.0.0"   },
    PORT:      { target: "command", label: "Bind port",   default: "30000"     },
    HF_TOKEN:  { target: "command", label: "HF token (Docker)", default: "<your-hf-token>" },
    CURL_HOST: { target: "curl",    label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl",    label: "Server port", default: "30000"     },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"What is the capital of France?"}] }'`,

  dockerImages: {
    "h20-3e": "lmsysorg/sglang:dev",
    "h200": "lmsysorg/sglang:dev",
    "h800": "lmsysorg/sglang:dev",
    "h100": "lmsysorg/sglang:dev",
    "b200": "lmsysorg/sglang:dev",
  },

  github: {
    cookbookModel: "inclusionAI/Ling-3.0-flash",
  },

  playgroundFeatures: {
    attention: {
      knobs: [
        { id: "tp", label: "TP", values: [null, 4, 8] },
      ],
    },
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser ling3" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser ling3" },
      ],
    },
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off", label: "Off (greedy)" },
        { id: "nextn", label: "NEXTN", flags: ["--speculative-algorithm NEXTN"] },
      ],
    },
  },

  cells: [
    {
      match: { hw: "h20-3e", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--enable-cache-report",
        "--log-level info",
        "--enable-metrics",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--chunked-prefill-size 8192",
        "--allow-auto-truncate",
        "--enable-trace",
        "--tool-call-parser ling3",
        "--reasoning-parser ling3",
        "--context-length 262144",
        "--max-mamba-cache-size 320",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--enable-cache-report",
        "--log-level info",
        "--enable-metrics",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--chunked-prefill-size 8192",
        "--allow-auto-truncate",
        "--enable-trace",
        "--tool-call-parser ling3",
        "--reasoning-parser ling3",
        "--context-length 262144",
        "--max-mamba-cache-size 320",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h800", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--enable-cache-report",
        "--log-level info",
        "--enable-metrics",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--chunked-prefill-size 8192",
        "--allow-auto-truncate",
        "--enable-trace",
        "--tool-call-parser ling3",
        "--reasoning-parser ling3",
        "--context-length 262144",
        "--max-mamba-cache-size 320",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "h100", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 8",
        "--enable-cache-report",
        "--log-level info",
        "--enable-metrics",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--chunked-prefill-size 8192",
        "--allow-auto-truncate",
        "--enable-trace",
        "--tool-call-parser ling3",
        "--reasoning-parser ling3",
        "--context-length 262144",
        "--max-mamba-cache-size 320",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
    {
      match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
      verified: false,
      env: [],
      flags: [
        "--trust-remote-code",
        "--model-path {{MODEL_NAME}}",
        "--tp 4",
        "--enable-cache-report",
        "--log-level info",
        "--enable-metrics",
        "--mem-fraction-static 0.85",
        "--max-running-requests 64",
        "--chunked-prefill-size 8192",
        "--allow-auto-truncate",
        "--enable-trace",
        "--tool-call-parser ling3",
        "--reasoning-parser ling3",
        "--context-length 262144",
        "--max-mamba-cache-size 320",
        "--speculative-algorithm NEXTN",
        "--json-model-override-args '{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"rope_theta\":6000000,\"partial_rotary_factor\":0.5,\"original_max_position_embeddings\":131072}}'",
        "--host {{HOST_IP}}",
        "--port {{PORT}}",
      ],
    },
  ],
};
