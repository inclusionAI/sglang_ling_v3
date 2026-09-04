from typing import List, Optional, Tuple, Union

import torch
from transformers.configuration_utils import PretrainedConfig

from sglang.kernels.ops.attention.rotary_triton import (
    triton_ernie45_rope_fused_inplace,
)
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding


class BailingMRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
        video_rope: Optional[bool] = False,
    ) -> None:
        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        position_start = -4 * max_position_embeddings if video_rope else 0
        self.cache_max_position_num = max_position_embeddings * 4 - position_start
        super().__init__(
            head_size,
            rotary_dim,
            self.cache_max_position_num,
            base,
            is_neox_style,
            dtype,
            position_start,
        )

        if mrope_section:
            assert sum(mrope_section) == rotary_dim // 2
            # Convert mrope_section from [t, h, w] to [h, w, t] for the
            # Bailing/Ernie4.5 interleaved layout: [h,w,h,w,...,t,t,t,...]
            mrope_section = [
                mrope_section[1],
                mrope_section[2],
                mrope_section[0],
            ]
        self.mrope_section = mrope_section

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        fused_set_kv_buffer_arg=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert positions.ndim == 1 or positions.ndim == 2
        positions = positions - self.position_start
        if positions.ndim == 2:
            assert (
                fused_set_kv_buffer_arg is None
            ), "fused_set_kv_buffer_arg is not supported for bailing rotary embedding."
            assert self.mrope_section
            # MLA passes 3D q/k (num_tokens, num_heads, head_dim) which is
            # non-contiguous from split(). The triton kernel expects 2D
            # (num_tokens, num_heads * head_dim). Make contiguous 2D copies,
            # run the inplace kernel, then return reshaped results.
            q_shape = query.shape
            k_shape = key.shape
            if query.ndim == 3:
                query = query.reshape(q_shape[0], -1)
                key = key.reshape(k_shape[0], -1)
            triton_ernie45_rope_fused_inplace(
                query,
                key,
                self.cos_sin_cache,
                positions,
                self.mrope_section,
                self.head_size,
                self.rotary_dim,
                self.is_neox_style,
            )
            if q_shape != query.shape:
                query = query.view(q_shape)
                key = key.view(k_shape)
            return query, key
        return RotaryEmbedding.forward(self, positions, query, key)

    @classmethod
    def bailing_3drope_get_input_positions_tensor(
        cls,
        input_ids: torch.Tensor,
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Bailing currently uses a fixed temporal stride and does not consume
        # absolute frame timestamps.
        scale_factor = 2.0
        second_per_grid_ts = None
        attention_mask = None
        use_abs_time_pos = second_per_grid_ts is not None
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        # tokens_per_second = hf_config.vision_config.tokens_per_second
        llm_config = hf_config.llm_config
        # audio_patch_token_id = llm_config.audio_patch_token
        image_patch_id = llm_config.image_patch_token
        video_patch_id = llm_config.video_patch_token
        # audio_start_token_id = llm_config.audio_start_token
        # audio_end_token_id = llm_config.audio_end_token
        image_start_token_id = llm_config.image_start_token
        # image_end_token_id = llm_config.image_end_token
        video_start_token_id = llm_config.video_start_token
        use_interleaved_frame_timestamp = getattr(
            llm_config, "use_interleaved_frame_timestamp", False
        )
        # video_end_token_id = llm_config.video_end_token
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                # input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            image_index, video_index = 0, 0
            image_nums, video_nums = 0, 0
            if image_grid_thw is not None:
                vision_start_indices = torch.argwhere(
                    input_ids == image_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_patch_id).sum()
            if video_grid_thw is not None:
                # for qwen3-vit
                if use_interleaved_frame_timestamp:
                    if isinstance(video_grid_thw, torch.Tensor):
                        video_grid_thw = torch.repeat_interleave(
                            video_grid_thw, video_grid_thw[:, 0], dim=0
                        )
                        video_grid_thw[:, 0] = 1

                    vision_start_indices = torch.argwhere(
                        input_ids == image_start_token_id
                    ).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_patch_id).sum()
                else:
                    vision_start_indices = torch.argwhere(
                        input_ids == video_start_token_id
                    ).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_patch_id).sum()

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_patch_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_patch_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_patch_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_patch_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )

                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0

                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1][0].max() + 1
                    if len(llm_pos_ids_list) > 0
                    else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                # body-diagonal symmetry
                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                    - (llm_grid_h - 1) // 2
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                    - (llm_grid_w - 1) // 2
                )

                # time dim adjust step size
                if use_abs_time_pos:
                    t_index = t_index * second_per_grid_t * scale_factor
                else:
                    t_index = t_index * scale_factor
                t_index = t_index + text_len + st_idx

                h_index = h_index + t_index
                w_index = w_index + t_index
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]))
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1][0].max() + 1
                    if len(llm_pos_ids_list) > 0
                    else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            # Keep the same [3, batch, seq] contract as the text-only path.
            # The processor removes this request axis; without it, squeezing a
            # valid single-position [3, 1] result would collapse it to [3].
            position_ids = llm_positions.to(
                dtype=position_ids.dtype, device=position_ids.device
            ).unsqueeze(1)

            mrope_position_deltas.append(
                llm_positions[0].max() + 1 - len(total_input_ids)
            )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            )
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                input_ids = input_ids.unsqueeze(0)
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )

                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
        return position_ids, mrope_position_deltas
