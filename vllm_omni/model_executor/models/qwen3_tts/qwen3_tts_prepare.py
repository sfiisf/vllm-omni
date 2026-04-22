from __future__ import annotations

import base64
import io
import os
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_qwen3_tts import Qwen3TTSConfig, Qwen3TTSTalkerConfig
from .qwen3_tts_tokenizer import Qwen3TTSTokenizer
from .qwen3_tts_talker import (
    Qwen3TTSSpeakerEncoder,
    Qwen3TTSTalkerForConditionalGeneration,
    Qwen3TTSTalkerResizeMLP,
    mel_spectrogram,
)

logger = init_logger(__name__)


class Qwen3TTSPrepare(nn.Module):
    """Lightweight request-preparation stage for Qwen3-TTS.

    This stage materializes the expensive request-local state required by the
    AR talker's first prefill:
    - talker_prompt_embeds
    - tailing_text_hidden
    - tts_pad_embed
    - ref_code_len / codec_streaming

    It intentionally does not instantiate the talker transformer blocks.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "talker.model.codec_embedding.": "codec_embedding.",
            "talker.model.text_embedding.": "text_embedding.",
            "talker.text_projection.": "text_projection.",
            "talker.code_predictor.model.codec_embedding.": "code_predictor_input_embeddings.",
            "speaker_encoder.": "speaker_encoder.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config: Qwen3TTSConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.talker_config: Qwen3TTSTalkerConfig = self.config.talker_config

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = False
        self.requires_raw_input_tokens = True

        self.codec_embedding = nn.Embedding(self.talker_config.vocab_size, self.talker_config.hidden_size)
        self.text_embedding = nn.Embedding(self.talker_config.text_vocab_size, self.talker_config.text_hidden_size)
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            self.talker_config.text_hidden_size,
            self.talker_config.text_hidden_size,
            self.talker_config.hidden_size,
            self.talker_config.hidden_act,
            bias=True,
        )
        self.code_predictor_input_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.talker_config.code_predictor_config.vocab_size, self.talker_config.hidden_size)
                for _ in range(int(self.talker_config.num_code_groups) - 1)
            ]
        )
        self.speaker_encoder: Qwen3TTSSpeakerEncoder | None = None
        self._tokenizer = None
        self._speech_tokenizer: Qwen3TTSTokenizer | None = None

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.codec_embedding(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            import transformers

            kwargs = dict(trust_remote_code=True, use_fast=True)
            if transformers.__version__ < "5":
                kwargs["fix_mistral_regex"] = True
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, **kwargs)
            self._tokenizer.padding_side = "left"
        return self._tokenizer

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            if u.scheme in ("http", "https"):
                return bool(u.netloc)
            return u.scheme == "file"
        except Exception:
            return False

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def _load_audio_to_np(self, x: str) -> tuple[np.ndarray, int]:
        import librosa

        if self._is_url(x):
            from vllm.multimodal.media import MediaConnector

            connector = MediaConnector(allowed_local_media_path="/")
            audio, sr = connector.fetch_audio(x)
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)

        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return np.asarray(audio, dtype=np.float32), int(sr)

    def _normalize_ref_audio(self, ref_audio: object) -> tuple[np.ndarray, int]:
        if isinstance(ref_audio, str):
            return self._load_audio_to_np(ref_audio)

        def _is_sr(x: object) -> bool:
            try:
                v = int(x)  # type: ignore[arg-type]
            except Exception:
                return False
            return 1_000 <= v <= 200_000

        def _is_number_sequence(xs: list[object]) -> bool:
            if not xs:
                return False
            for v in xs[:8]:
                if not isinstance(v, (int, float, np.number)):
                    return False
            return True

        wav_candidates: list[object] = []
        sr_candidates: list[int] = []

        def _scan(obj: object, depth: int = 0) -> None:
            if depth > 4 or obj is None:
                return
            if _is_sr(obj):
                sr_candidates.append(int(obj))  # type: ignore[arg-type]
                return
            if isinstance(obj, np.ndarray) and obj.size > 0:
                wav_candidates.append(obj)
                return
            if isinstance(obj, torch.Tensor) and obj.numel() > 0:
                wav_candidates.append(obj)
                return
            if isinstance(obj, dict):
                wav_obj = obj.get("array") or obj.get("wav") or obj.get("audio")
                sr_obj = obj.get("sampling_rate") or obj.get("sr") or obj.get("sample_rate")
                if wav_obj is not None:
                    _scan(wav_obj, depth + 1)
                if sr_obj is not None:
                    _scan(sr_obj, depth + 1)
                return
            if isinstance(obj, (tuple, list)):
                obj_list = list(obj)
                while isinstance(obj_list, list) and len(obj_list) == 1:
                    inner = obj_list[0]
                    if isinstance(inner, (np.ndarray, torch.Tensor, dict)):
                        _scan(inner, depth + 1)
                        return
                    if isinstance(inner, (tuple, list)):
                        obj_list = list(inner)
                        continue
                    break
                if len(obj_list) >= 512 and _is_number_sequence(obj_list):
                    wav_candidates.append(obj_list)
                    return
                for item in obj_list:
                    if isinstance(item, list) and len(item) >= 512 and _is_number_sequence(item):
                        wav_candidates.append(item)
                        continue
                    _scan(item, depth + 1)

        _scan(ref_audio)
        if not sr_candidates:
            raise TypeError("ref_audio missing sample_rate")
        sr = int(sr_candidates[0])
        if not wav_candidates:
            raise TypeError("ref_audio missing waveform")
        wav_obj = max(
            wav_candidates,
            key=lambda x: int(x.size if isinstance(x, np.ndarray) else x.numel() if isinstance(x, torch.Tensor) else len(x)),  # type: ignore[arg-type]
        )
        if isinstance(wav_obj, np.ndarray):
            wav_np = wav_obj.astype(np.float32).reshape(-1)
        elif isinstance(wav_obj, torch.Tensor):
            wav_np = wav_obj.detach().to("cpu").float().contiguous().numpy().reshape(-1)
        else:
            wav_np = np.asarray(wav_obj, dtype=np.float32).reshape(-1)
        if wav_np.size < 1024:
            raise ValueError(f"ref_audio waveform too short: {wav_np.size} samples")
        return wav_np, sr

    def _extract_speaker_embedding(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        if self.speaker_encoder is None:
            raise ValueError("This checkpoint does not provide `speaker_encoder` weights.")
        dev = next(self.parameters()).device
        try:
            spk_param = next(self.speaker_encoder.parameters())
            if spk_param.device != dev or spk_param.dtype != torch.bfloat16:
                self.speaker_encoder.to(device=dev, dtype=torch.bfloat16)
        except StopIteration:
            pass
        target_sr = int(getattr(self.config.speaker_encoder_config, "sample_rate", 24000))
        if sr != target_sr:
            import librosa

            wav = librosa.resample(y=wav.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)

        mels = mel_spectrogram(
            torch.from_numpy(wav).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        spk = self.speaker_encoder(mels.to(dev, dtype=torch.bfloat16))[0]
        return spk.to(dtype=torch.bfloat16)

    def _ensure_speech_tokenizer_loaded(self) -> Qwen3TTSTokenizer:
        if self._speech_tokenizer is not None:
            return self._speech_tokenizer
        speech_tokenizer_path = cached_file(self.model_path, "speech_tokenizer/config.json")
        if speech_tokenizer_path is None:
            raise ValueError(f"{self.model_path}/speech_tokenizer/config.json not found")
        preprocessor_config_path = cached_file(self.model_path, "speech_tokenizer/preprocessor_config.json")
        if preprocessor_config_path is None:
            raise ValueError(f"{self.model_path}/speech_tokenizer/preprocessor_config.json not found")
        speech_tokenizer_dir = os.path.dirname(speech_tokenizer_path)
        tok = Qwen3TTSTokenizer.from_pretrained(speech_tokenizer_dir, torch_dtype=torch.bfloat16)
        dev = next(self.parameters()).device
        if getattr(dev, "type", None) == "cuda":
            tok.model.to(dev)
            tok.device = dev
        else:
            tok.device = dev
        self._speech_tokenizer = tok
        return tok

    def _encode_ref_audio_to_code(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        tok = self._ensure_speech_tokenizer_loaded()
        enc = tok.encode(wav, sr=int(sr), return_dict=True)
        ref_code = getattr(enc, "audio_codes", None)
        if isinstance(ref_code, list):
            ref_code = ref_code[0] if ref_code else None
        if isinstance(ref_code, torch.Tensor):
            if ref_code.ndim == 3:
                ref_code = ref_code[0]
            return ref_code.to(device=next(self.parameters()).device, dtype=torch.long)
        raise ValueError("SpeechTokenizer.encode did not return audio_codes tensor")

    def _generate_icl_prompt(
        self,
        *,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_embed = self.text_projection(self.text_embedding(torch.cat([ref_id, text_id], dim=-1)))
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

        codec_embed: list[torch.Tensor] = []
        for i in range(int(self.talker_config.num_code_groups)):
            if i == 0:
                codec_embed.append(self.embed_input_ids(ref_code[:, :1]))
            else:
                codec_embed.append(self.code_predictor_input_embeddings[i - 1](ref_code[:, i : i + 1]))
        codec_embed_sum = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed_sum = torch.cat(
            [
                self.embed_input_ids(
                    torch.tensor([[self.talker_config.codec_bos_id]], device=codec_embed_sum.device, dtype=torch.long)
                ),
                codec_embed_sum,
            ],
            dim=1,
        )

        text_lens = int(text_embed.shape[1])
        codec_lens = int(codec_embed_sum.shape[1])
        if non_streaming_mode:
            icl_input_embed = text_embed + self.embed_input_ids(
                torch.tensor(
                    [[self.talker_config.codec_pad_id] * text_lens],
                    device=codec_embed_sum.device,
                    dtype=torch.long,
                )
            )
            icl_input_embed = torch.cat([icl_input_embed, codec_embed_sum + tts_pad_embed], dim=1)
            return icl_input_embed, tts_pad_embed
        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed_sum, text_embed[:, codec_lens:]
        text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
        return text_embed + codec_embed_sum, tts_pad_embed

    def _build_prompt_embeds(
        self,
        *,
        task_type: str,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None, bool, torch.Tensor | None]:
        text = (info_dict.get("text") or [""])[0]
        language = (info_dict.get("language") or ["Auto"])[0]
        non_streaming_mode_val = info_dict.get("non_streaming_mode")
        if isinstance(non_streaming_mode_val, list):
            non_streaming_mode_raw = non_streaming_mode_val[0] if non_streaming_mode_val else None
        else:
            non_streaming_mode_raw = non_streaming_mode_val
        if isinstance(non_streaming_mode_raw, bool):
            non_streaming_mode = non_streaming_mode_raw
        else:
            non_streaming_mode = task_type in ("CustomVoice", "VoiceDesign")

        tok = self._get_tokenizer()
        input_ids = tok(
            Qwen3TTSTalkerForConditionalGeneration._build_assistant_text(text),
            return_tensors="pt",
            padding=False,
        )["input_ids"].to(device=next(self.parameters()).device)

        instruct = (info_dict.get("instruct") or [""])[0]
        instruct_embed = None
        if isinstance(instruct, str) and instruct.strip():
            instruct_ids = tok(
                Qwen3TTSTalkerForConditionalGeneration._build_instruct_text(instruct),
                return_tensors="pt",
                padding=False,
            )["input_ids"].to(device=input_ids.device)
            instruct_embed = self.text_projection(self.text_embedding(instruct_ids))

        tts_tokens = torch.tensor(
            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.text_projection(self.text_embedding(tts_tokens)).chunk(
            3, dim=1
        )

        language_id = None
        if isinstance(language, str) and language.lower() != "auto":
            language_id = self.talker_config.codec_language_id.get(language.lower())
        if language_id is None and isinstance(language, str) and language.lower() in ("chinese", "auto"):
            speaker_for_dialect = None
            if task_type == "CustomVoice":
                speaker_for_dialect = (info_dict.get("speaker") or [""])[0]
            if isinstance(speaker_for_dialect, str) and speaker_for_dialect.strip():
                spk_is_dialect = getattr(self.talker_config, "spk_is_dialect", None) or {}
                dialect = spk_is_dialect.get(speaker_for_dialect.lower())
                if isinstance(dialect, str) and dialect:
                    language_id = self.talker_config.codec_language_id.get(dialect)
        if language_id is None:
            codec_prefill_list = [[
                self.talker_config.codec_nothink_id,
                self.talker_config.codec_think_bos_id,
                self.talker_config.codec_think_eos_id,
            ]]
        else:
            codec_prefill_list = [[
                self.talker_config.codec_think_id,
                self.talker_config.codec_think_bos_id,
                int(language_id),
                self.talker_config.codec_think_eos_id,
            ]]

        codec_input_0 = self.embed_input_ids(torch.tensor(codec_prefill_list, device=input_ids.device, dtype=torch.long))
        codec_input_1 = self.embed_input_ids(
            torch.tensor([[self.talker_config.codec_pad_id, self.talker_config.codec_bos_id]], device=input_ids.device)
        )

        speaker_embed = None
        ref_code_len: int | None = None
        ref_code_prompt: torch.Tensor | None = None

        def _as_singleton(x: object) -> object:
            if isinstance(x, list):
                return x[0] if x else None
            return x

        def _to_long_tensor(x: object, *, device: torch.device) -> torch.Tensor | None:
            x = _as_singleton(x)
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                t = x
            elif isinstance(x, np.ndarray):
                t = torch.from_numpy(x)
            elif isinstance(x, list) and x and all(isinstance(v, (int, np.integer)) for v in x):
                t = torch.tensor(x, dtype=torch.long)
            else:
                return None
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return t.to(device=device, dtype=torch.long)

        def _normalize_voice_clone_prompt(raw: object) -> dict[str, object] | None:
            raw = _as_singleton(raw)
            if raw is None:
                return None
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                return raw[0]
            return None

        codec_streaming = task_type == "Base"
        if task_type == "Base":
            xvec_only = bool((info_dict.get("x_vector_only_mode") or [False])[0])
            in_context_mode = not xvec_only
            voice_clone_prompt = _normalize_voice_clone_prompt(info_dict.get("voice_clone_prompt"))
            if voice_clone_prompt is not None and "icl_mode" in voice_clone_prompt:
                icl_flag = _as_singleton(voice_clone_prompt.get("icl_mode"))
                if isinstance(icl_flag, bool):
                    in_context_mode = icl_flag
                    xvec_only = not in_context_mode
            ref_code = _as_singleton(voice_clone_prompt.get("ref_code")) if voice_clone_prompt is not None else None
            ref_code_t = None
            if isinstance(ref_code, torch.Tensor):
                ref_code_t = ref_code
            elif isinstance(ref_code, np.ndarray):
                ref_code_t = torch.from_numpy(ref_code)
            if isinstance(ref_code_t, torch.Tensor):
                if ref_code_t.ndim == 3:
                    ref_code_t = ref_code_t[0]
                ref_code_t = ref_code_t.to(device=input_ids.device, dtype=torch.long)
                ref_code_len = int(ref_code_t.shape[0])
            elif in_context_mode:
                ref_audio_list = info_dict.get("ref_audio")
                if not isinstance(ref_audio_list, list) or not ref_audio_list:
                    raise ValueError("Base requires `ref_audio`.")
                wav_np, sr = self._normalize_ref_audio(ref_audio_list[0])
                ref_code_t = self._encode_ref_audio_to_code(wav_np, sr).to(device=input_ids.device)
                ref_code_len = int(ref_code_t.shape[0])
            if isinstance(ref_code_t, torch.Tensor):
                ref_code_prompt = ref_code_t
            spk = _as_singleton(voice_clone_prompt.get("ref_spk_embedding")) if voice_clone_prompt is not None else None
            if isinstance(spk, torch.Tensor):
                speaker_embed = spk.to(device=input_ids.device, dtype=torch.bfloat16).view(1, 1, -1)
            else:
                ref_audio_list = info_dict.get("ref_audio")
                if not isinstance(ref_audio_list, list) or not ref_audio_list:
                    raise ValueError("Base requires `ref_audio`.")
                wav_np, sr = self._normalize_ref_audio(ref_audio_list[0])
                speaker_embed = self._extract_speaker_embedding(wav_np, sr).view(1, 1, -1)

            codec_input = torch.cat([codec_input_0, speaker_embed, codec_input_1], dim=1)
            role_embed = self.text_projection(self.text_embedding(input_ids[:, :3]))
            codec_prefix = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1)
            codec_prefix = codec_prefix + codec_input[:, :-1]
            talker_prompt = torch.cat((role_embed, codec_prefix), dim=1)

            if in_context_mode:
                ref_ids = _to_long_tensor(info_dict.get("ref_ids"), device=input_ids.device)
                if ref_ids is None and voice_clone_prompt is not None:
                    ref_ids = _to_long_tensor(
                        voice_clone_prompt.get("ref_ids") or voice_clone_prompt.get("ref_id"),
                        device=input_ids.device,
                    )
                if ref_ids is None:
                    ref_text = _as_singleton(info_dict.get("ref_text"))
                    if not isinstance(ref_text, str) or not ref_text.strip():
                        raise ValueError("Base in-context voice cloning requires `ref_text` or tokenized `ref_ids`.")
                    ref_ids = tok(
                        Qwen3TTSTalkerForConditionalGeneration._build_ref_text(ref_text),
                        return_tensors="pt",
                        padding=False,
                    )["input_ids"].to(device=input_ids.device)
                icl_input_embed, trailing_text_hidden = self._generate_icl_prompt(
                    text_id=input_ids[:, 3:-5],
                    ref_id=ref_ids[:, 3:-2],
                    ref_code=ref_code_t,  # type: ignore[arg-type]
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_prompt = torch.cat([talker_prompt, icl_input_embed], dim=1)
            elif non_streaming_mode:
                text_all = self.text_projection(self.text_embedding(input_ids[:, 3:-5]))
                text_all = torch.cat([text_all, tts_eos_embed], dim=1)
                pad_ids = torch.full(
                    (1, int(text_all.shape[1])),
                    int(self.talker_config.codec_pad_id),
                    device=input_ids.device,
                    dtype=torch.long,
                )
                talker_prompt = torch.cat(
                    [
                        talker_prompt,
                        text_all + self.embed_input_ids(pad_ids),
                        tts_pad_embed
                        + self.embed_input_ids(torch.tensor([[self.talker_config.codec_bos_id]], device=input_ids.device)),
                    ],
                    dim=1,
                )
                trailing_text_hidden = tts_pad_embed
            else:
                first_text = self.text_projection(self.text_embedding(input_ids[:, 3:4])) + codec_input[:, -1:]
                talker_prompt = torch.cat([talker_prompt, first_text], dim=1)
                trailing_text_hidden = torch.cat(
                    (self.text_projection(self.text_embedding(input_ids[:, 4:-5])), tts_eos_embed),
                    dim=1,
                )

        elif task_type == "CustomVoice":
            speaker = (info_dict.get("speaker") or [""])[0]
            if not isinstance(speaker, str) or not speaker.strip():
                raise ValueError("CustomVoice requires additional_information.speaker.")
            spk_id_map = getattr(self.talker_config, "spk_id", None) or {}
            if speaker.lower() not in spk_id_map:
                raise ValueError(f"Unsupported speaker: {speaker}")
            spk_tensor = torch.tensor([spk_id_map[speaker.lower()]], device=input_ids.device, dtype=torch.long)
            spk_embed = self.embed_input_ids(spk_tensor).view(1, 1, -1)
            codec_input = torch.cat([codec_input_0, spk_embed, codec_input_1], dim=1)
            role_embed = self.text_projection(self.text_embedding(input_ids[:, :3]))
            codec_prefix = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1)
            codec_prefix = codec_prefix + codec_input[:, :-1]
            talker_prompt = torch.cat((role_embed, codec_prefix), dim=1)
            if non_streaming_mode:
                text_all = self.text_projection(self.text_embedding(input_ids[:, 3:-5]))
                text_all = torch.cat([text_all, tts_eos_embed], dim=1)
                pad_ids = torch.full(
                    (1, int(text_all.shape[1])),
                    int(self.talker_config.codec_pad_id),
                    device=input_ids.device,
                    dtype=torch.long,
                )
                talker_prompt = torch.cat(
                    [
                        talker_prompt,
                        text_all + self.embed_input_ids(pad_ids),
                        tts_pad_embed
                        + self.embed_input_ids(torch.tensor([[self.talker_config.codec_bos_id]], device=input_ids.device)),
                    ],
                    dim=1,
                )
                trailing_text_hidden = tts_pad_embed
            else:
                first_text = self.text_projection(self.text_embedding(input_ids[:, 3:4])) + codec_input[:, -1:]
                talker_prompt = torch.cat([talker_prompt, first_text], dim=1)
                trailing_text_hidden = torch.cat(
                    (self.text_projection(self.text_embedding(input_ids[:, 4:-5])), tts_eos_embed),
                    dim=1,
                )
        elif task_type == "VoiceDesign":
            codec_input = torch.cat([codec_input_0, codec_input_1], dim=1)
            role_embed = self.text_projection(self.text_embedding(input_ids[:, :3]))
            codec_prefix = torch.cat((tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1), tts_bos_embed), dim=1)
            codec_prefix = codec_prefix + codec_input[:, :-1]
            talker_prompt = torch.cat((role_embed, codec_prefix), dim=1)
            if non_streaming_mode:
                text_all = self.text_projection(self.text_embedding(input_ids[:, 3:-5]))
                text_all = torch.cat([text_all, tts_eos_embed], dim=1)
                pad_ids = torch.full(
                    (1, int(text_all.shape[1])),
                    int(self.talker_config.codec_pad_id),
                    device=input_ids.device,
                    dtype=torch.long,
                )
                talker_prompt = torch.cat(
                    [
                        talker_prompt,
                        text_all + self.embed_input_ids(pad_ids),
                        tts_pad_embed
                        + self.embed_input_ids(torch.tensor([[self.talker_config.codec_bos_id]], device=input_ids.device)),
                    ],
                    dim=1,
                )
                trailing_text_hidden = tts_pad_embed
            else:
                first_text = self.text_projection(self.text_embedding(input_ids[:, 3:4])) + codec_input[:, -1:]
                talker_prompt = torch.cat([talker_prompt, first_text], dim=1)
                trailing_text_hidden = torch.cat(
                    (self.text_projection(self.text_embedding(input_ids[:, 4:-5])), tts_eos_embed),
                    dim=1,
                )
        else:
            raise ValueError(f"Unsupported task_type={task_type}")

        if instruct_embed is not None:
            talker_prompt = torch.cat([instruct_embed, talker_prompt], dim=1)

        return (
            talker_prompt.squeeze(0),
            trailing_text_hidden.squeeze(0),
            tts_pad_embed.squeeze(0),
            ref_code_len,
            codec_streaming,
            ref_code_prompt.contiguous() if isinstance(ref_code_prompt, torch.Tensor) else None,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = runtime_additional_information or []
        prepared_prompt_embeds: list[torch.Tensor] = []
        tailing_text_hidden: list[torch.Tensor] = []
        tts_pad_embed: list[torch.Tensor] = []
        ref_code_len: list[torch.Tensor] = []
        codec_streaming: list[torch.Tensor] = []
        ref_code: list[torch.Tensor | None] = []

        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            task_type = (info.get("task_type") or ["CustomVoice"])[0]
            prompt, tail, pad, ref_len, streaming, ref_code_prompt = self._build_prompt_embeds(
                task_type=task_type,
                info_dict=info,
            )
            prepared_prompt_embeds.append(prompt.to(dtype=torch.float32))
            tailing_text_hidden.append(tail.to(dtype=torch.float32))
            tts_pad_embed.append(pad.to(dtype=torch.float32))
            ref_code_len.append(torch.tensor([ref_len if ref_len is not None else -1], dtype=torch.int32, device=pad.device))
            codec_streaming.append(torch.tensor([1 if streaming else 0], dtype=torch.int8, device=pad.device))
            if ref_code_prompt is not None:
                ref_code.append(
                    ref_code_prompt.detach().to("cpu").contiguous()
                    if isinstance(ref_code_prompt, torch.Tensor) and ref_code_prompt.numel() > 0
                    else None
                )

        if len(ref_code) == 0:
            return OmniOutput(
                text_hidden_states=torch.empty((0, 1), device=self._module_device(self), dtype=torch.float32),
                multimodal_outputs={
                    "prepared_prompt_embeds": prepared_prompt_embeds,
                    "tailing_text_hidden": tailing_text_hidden,
                    "tts_pad_embed": tts_pad_embed,
                    "ref_code_len": ref_code_len,
                    "codec_streaming": codec_streaming,
                },
            )
        
        return OmniOutput(
            text_hidden_states=torch.empty((0, 1), device=self._module_device(self), dtype=torch.float32),
            multimodal_outputs={
                "prepared_prompt_embeds": prepared_prompt_embeds,
                "tailing_text_hidden": tailing_text_hidden,
                "tts_pad_embed": tts_pad_embed,
                "ref_code_len": ref_code_len,
                "codec_streaming": codec_streaming,
                "ref_code": ref_code,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        speaker_weights: list[tuple[str, torch.Tensor]] = []

        def _prepare_weights(ws: Iterable[tuple[str, torch.Tensor]]):
            for k, v in ws:
                if k.startswith("speaker_encoder."):
                    speaker_weights.append((k, v))
                    continue
                if k.startswith(
                    (
                        "talker.model.codec_embedding.",
                        "talker.model.text_embedding.",
                        "talker.text_projection.",
                        "talker.code_predictor.model.codec_embedding.",
                    )
                ):
                    yield k, v

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(_prepare_weights(weights), mapper=self.hf_to_vllm_mapper)

        if speaker_weights:
            if self.speaker_encoder is None:
                self.speaker_encoder = Qwen3TTSSpeakerEncoder(self.config.speaker_encoder_config)
            loaded |= loader.load_weights(speaker_weights, mapper=self.hf_to_vllm_mapper)

        logger.info("Loaded %d weights for Qwen3TTSPrepare", len(loaded))
        return loaded
