from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.qwen3_vl import Qwen3VLMoeVisionConfig


class BailingMoeV2Config(PretrainedConfig):
    model_type = "bailing_moe_v2"

    ignore_keys_at_rope_validation = {"mrope_section", "use_video_rope"}

    def __init__(
        self,
        vocab_size=30592,
        hidden_size=1024,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=0,
        hidden_act="silu",
        use_qkv_bias=False,  # bailing only
        use_qk_norm=False,
        use_bias=True,  # bailing only
        rms_norm_eps=1e-05,
        norm_head=False,  # bailing only
        tie_word_embeddings=False,  # PretrainedConfig key, here change default value.
        embedding_dropout=0.1,
        attention_dropout=0.1,
        output_dropout=0.1,
        initializer_range=0.02,
        max_position_embeddings=16384,
        rope_theta=10000.0,
        use_cache=True,
        use_sliding_window=False,
        sliding_window=81920,
        max_window_layers=28,
        rope_scaling=None,
        pad_token_id=126081,
        num_experts=16,
        num_shared_experts=0,
        num_experts_per_tok=2,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        moe_intermediate_size=None,
        first_k_dense_replace=0,
        head_dim=None,
        output_router_logits=False,
        partial_rotary_factor=0.5,
        router_type="topN",
        _attn_implementation="flash_attention_2",
        use_interleaved_frame_timestamp=True,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.use_bias = use_bias
        self.norm_head = norm_head
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling
        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.partial_rotary_factor = partial_rotary_factor
        self.router_type = router_type
        self.use_interleaved_frame_timestamp = use_interleaved_frame_timestamp
        super().__init__(
            pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )
        self._attn_implementation = _attn_implementation


class WhisperEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        whisper_encoder_config: dict = None,
        ds_kernel_size=3,
        ds_stride=2,
        **kwargs,
    ):
        self.whisper_encoder_config = whisper_encoder_config
        self.ds_kernel_size = ds_kernel_size
        self.ds_stride = ds_stride

        super().__init__(**kwargs)


class BailingMM2Config(PretrainedConfig):
    model_type = "bailingmm_moe_v2_lite"

    def __init__(
        self,
        mlp_depth=1,
        llm_config=None,
        vision_config=None,
        audio_config=None,
        mrope_section=None,
        **kwargs,
    ):
        if isinstance(audio_config, WhisperEncoderConfig):
            self.audio_config = audio_config
        elif isinstance(audio_config, dict):
            self.audio_config = WhisperEncoderConfig(**audio_config)
        else:
            self.audio_config = WhisperEncoderConfig()

        if isinstance(vision_config, Qwen3VLMoeVisionConfig):
            self.vision_config = vision_config
        elif isinstance(vision_config, dict):
            self.vision_config = Qwen3VLMoeVisionConfig(**vision_config)
        else:
            self.vision_config = Qwen3VLMoeVisionConfig()

        if isinstance(llm_config, BailingMoeV2Config):
            self.llm_config = llm_config
        elif isinstance(llm_config, dict):
            self.llm_config = BailingMoeV2Config(**llm_config)
        else:
            self.llm_config = BailingMoeV2Config()
        self.mlp_depth = mlp_depth
        # M-RoPE section: split qk_rope_head_dim // 2 frequencies into [T, H, W]
        # Default: [8, 12, 12] sums to 32 = qk_rope_head_dim(64) // 2
        if mrope_section is None:
            mrope_section = self.llm_config.rope_parameters.get(
                "mrope_section", [8, 12, 12]
            )
        self.mrope_section = mrope_section
        self.llm_config.rope_parameters.update(
            rope_type="default",
            mrope_section=mrope_section,
            video_rope=True,
        )
        super().__init__(**kwargs)
