"""Stage input processor for Qwen3-TTS prepare -> talker."""

from __future__ import annotations

import copy

import torch

from vllm_omni.inputs.data import OmniTokensPrompt


def prepare2talker(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | None = None,
    requires_multimodal_data: bool = False,
):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    prepare_outputs = stage_list[source_stage_id].engine_outputs
    if not isinstance(prompt, list):
        prompt = [prompt]

    talker_inputs = []
    codec_pad_id = 4196
    target_stage_id = source_stage_id + 1
    if target_stage_id < len(stage_list):
        try:
            cfg = stage_list[target_stage_id].vllm_config.model_config.hf_config
            codec_pad_id = int(cfg.talker_config.codec_pad_id)
        except Exception:
            pass
    for prepare_output, raw_prompt in zip(prepare_outputs, prompt):
        output = prepare_output.outputs[0]
        mm = getattr(output, "multimodal_output", None) or {}
        prompt_embeds = mm["prepared_prompt_embeds"]
        if not isinstance(prompt_embeds, torch.Tensor):
            raise TypeError("prepare2talker expects `prepared_prompt_embeds` tensor")

        prompt_len = max(1, int(prompt_embeds.shape[0]))
        add_info = copy.deepcopy((raw_prompt or {}).get("additional_information", {}) or {})
        add_info["text"] = add_info.get("text", [""])
        add_info["talker_prompt_embeds"] = prompt_embeds.to(torch.float32)
        add_info["tailing_text_hidden"] = mm["tailing_text_hidden"].to(torch.float32)
        add_info["tts_pad_embed"] = mm["tts_pad_embed"].to(torch.float32)

        ref_len_t = mm.get("ref_code_len")
        if isinstance(ref_len_t, torch.Tensor) and ref_len_t.numel() > 0:
            ref_len = int(ref_len_t.reshape(-1)[0].item())
            if ref_len >= 0:
                add_info["ref_code_len"] = [ref_len]

        ref_code_t = mm.get("ref_code")
        if isinstance(ref_code_t, list):
            ref_code_t = ref_code_t[0] if ref_code_t else None
        if isinstance(ref_code_t, torch.Tensor) and ref_code_t.numel() > 0:
            add_info["ref_code"] = ref_code_t.to(torch.long).cpu().contiguous()

        streaming_t = mm.get("codec_streaming")
        if isinstance(streaming_t, torch.Tensor) and streaming_t.numel() > 0:
            add_info["codec_streaming"] = [bool(int(streaming_t.reshape(-1)[0].item()))]

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[codec_pad_id] * prompt_len,
                additional_information=add_info,
                multi_modal_data=(
                    (raw_prompt or {}).get("multi_modal_data", None) if requires_multimodal_data else None
                ),
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs
