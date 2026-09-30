export const config = {
  modelName: "Ling-3.0-plus",
  supportedHardware: ["h20-3e"],
  groupHardware: false,
  runModes: ["python"],

  modelNames: { default: "ling-3.0-plus" },
  placeholders: {
    FP8_MODEL_PATH: {
      target: "command",
      label: "FP8 checkpoint",
      default: "/path/to/Ling-3.0-plus-fp8",
    },
    MXFP4_MODEL_PATH: {
      target: "command",
      label: "MXFP4 checkpoint",
      default: "/path/to/Ling-3.0-plus-mxfp4",
    },
    DSPARK_MODEL_PATH: {
      target: "command",
      label: "DSpark draft checkpoint",
      default: "/path/to/Ling-3.0-plus-dspark",
    },
    HOST_IP: { target: "command", label: "Bind host", default: "0.0.0.0" },
    PORT: { target: "command", label: "Bind port", default: "30000" },
    CURL_HOST: { target: "curl", label: "Server host", default: "localhost" },
    CURL_PORT: { target: "curl", label: "Server port", default: "30000" },
  },

  curl: `curl http://{{CURL_HOST}}:{{CURL_PORT}}/v1/chat/completions \\
-H 'Content-Type: application/json' \\
-d '{ "model": "{{MODEL_NAME}}", "messages": [{"role":"user","content":"What is 15% of 240?"}] }'`,

  playgroundFeatures: {
    attention: { knobs: [{ id: "tp", label: "TP", values: [null, 4, 8] }] },
    parsers: {
      items: [
        { id: "reasoning", label: "Reasoning Parser", flag: "--reasoning-parser ling3" },
        { id: "toolCall", label: "Tool Call Parser", flag: "--tool-call-parser ling3" },
      ],
    },
    speculative: {
      options: [
        { id: "current", label: "Inherited from base" },
        { id: "off", label: "Off" },
        {
          id: "dspark",
          label: "DSPARK (draft model)",
          flags: [
            "--speculative-algorithm DSPARK",
            "--speculative-draft-model-path {{DSPARK_MODEL_PATH}}",
            "--enable-linear-replayssm-spec",
            "--linear-replayssm-cache-len 32",
          ],
        },
      ],
    },
  },

  matchDims: [
    {
      id: "quant",
      title: "Quantization",
      options: [
        { id: "fp8", label: "FP8" },
        { id: "mxfp4", label: "MXFP4" },
      ],
    },
    {
      id: "spec",
      title: "Spec Decode",
      options: [
        { id: "off", label: "Off" },
        { id: "dspark", label: "DSPARK" },
      ],
    },
    { id: "nodes", title: "Nodes", options: [{ id: "single", label: "Single Node" }] },
  ],

  cells: (() => {
    const common = [
      "--served-model-name ling-3.0-plus",
      "--reasoning-parser ling3",
      "--tool-call-parser ling3",
      "--host {{HOST_IP}}",
      "--port {{PORT}}",
    ];
    const dspark = [
      "--speculative-algorithm DSPARK",
      "--speculative-draft-model-path {{DSPARK_MODEL_PATH}}",
      "--enable-linear-replayssm-spec",
      "--linear-replayssm-cache-len 32",
    ];
    const recipes = [
      {
        quant: "fp8",
        flags: [
          "--model-path {{FP8_MODEL_PATH}}",
          "--tp 8",
          "--ep-size 8",
          "--mem-fraction-static 0.8",
        ],
      },
      {
        quant: "mxfp4",
        flags: [
          "--model-path {{MXFP4_MODEL_PATH}}",
          "--tp 4",
          "--moe-runner-backend humming",
          "--mem-fraction-static 0.85",
        ],
      },
    ];
    return recipes.flatMap(({ quant, flags }) => [
      {
        match: { hw: "h20-3e", quant, spec: "off", nodes: "single" },
        verified: true,
        flags: [...flags, ...common],
      },
      {
        match: { hw: "h20-3e", quant, spec: "dspark", nodes: "single" },
        verified: false,
        flags: [...flags, ...dspark, ...common],
      },
    ]);
  })(),
};
