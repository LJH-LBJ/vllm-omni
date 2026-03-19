# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Inference-only Qwen3-Omni-Moe Code2Wav model."""

from __future__ import annotations

from collections.abc import Iterable
import time

import json
import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavTransformerModel,
    Qwen3OmniMoeConvNeXtBlock,
    SnakeBeta,
)
from vllm.config import VllmConfig  # type: ignore
from vllm.logger import init_logger  # type: ignore
from vllm.model_executor.models.utils import (  # type: ignore
    AutoWeightsLoader,
    WeightsMapper,
)

logger = init_logger(__name__)
_DEBUG_SESSION_ID = "e9bef0"


def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        logger.info("[AGENT DEBUG NDJSON] %s", json.dumps(payload, ensure_ascii=True))
    except Exception:
        pass


class Qwen3OmniMoeCode2Wav(nn.Module):
    """
    Qwen3 Omni MoE Code2Wav - Converts num_quantizers-layer RVQ codec codes to audio waveform.

    Architecture:
    1. Code Embedding: Embed and average num_quantizers RVQ layers
    2. Pre-Transformer: Add temporal context via sliding-window attention
    3. Upsampling: Progressive upsampling with ConvNeXt blocks
    4. Decoder: Multi-stage upsampling + residual units → waveform

    Input: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
    Output: [batch, 1, waveform_len] - Audio waveform [-1, 1]

    Total upsampling factor: ~1280x
    Example: 100 codec frames → 128,000 audio samples (8 seconds at 16kHz)
    """

    input_modalities = "audio"

    # Weight mapper
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "code2wav.pre_transformer.": "pre_transformer.",
            "code2wav.code_embedding.": "code_embedding.",
            "code2wav.upsample.": "upsample.",
            "code2wav.decoder.": "decoder.",
            "code2wav.": "",
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config: Qwen3OmniMoeCode2WavConfig = vllm_config.model_config.hf_config

        # Calculate total upsampling factor
        self.total_upsample = np.prod(self.config.upsample_rates + self.config.upsampling_ratios)

        # Pre-transformer
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel._from_config(self.config)

        # Code embedding: Single embedding table for all RVQ layers
        self.code_embedding = nn.Embedding(
            self.config.codebook_size * self.config.num_quantizers, self.config.hidden_size
        )

        # Offset for each RVQ layer (layer 0: 0-1023, layer 1: 1024-2047, etc.)
        self.register_buffer(
            "code_offset",
            torch.arange(self.config.num_quantizers).view(1, -1, 1) * self.config.codebook_size,
            persistent=False,
        )

        # Upsampling blocks (e.g., 2x, 2x)
        upsample = []
        for factor in self.config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3OmniMoeCausalTransConvNet(
                            self.config.hidden_size, self.config.hidden_size, factor, factor
                        ),
                        Qwen3OmniMoeConvNeXtBlock(self.config.hidden_size),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        # Decoder: Initial projection + progressive upsampling blocks
        decoder = [Qwen3OmniMoeCausalConvNet(self.config.hidden_size, self.config.decoder_dim, kernel_size=7)]

        # Add decoder blocks (each upsamples and reduces channels)
        for i in range(len(self.config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(self.config, i))

        # Final projection to waveform
        output_dim = self.config.decoder_dim // 2 ** len(self.config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, kernel_size=7),
        ]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert num_quantizers-layer RVQ codes to audio waveform.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codec codes

        Returns:
            waveform: [batch, 1, waveform_len] - Audio waveform clipped to [-1, 1]
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layers of codes, got {codes.shape[1]}")

        # Stage 1: Code Embedding
        # Add offset to separate layer vocabularies, then embed and average
        hidden = self.code_embedding(codes + self.code_offset).mean(1)
        # Shape: [batch, seq_len, hidden_size]

        # Stage 2: Pre-Transformer (add temporal context)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        # Shape: [batch, seq_len, hidden_size]

        # Stage 3: Upsampling
        hidden = hidden.permute(0, 2, 1)  # [batch, hidden_size, seq_len]
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        # Shape: [batch, hidden_size, seq_len * upsample_factor]

        # Stage 4: Decoder (progressive upsampling to waveform)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        # Shape: [batch, 1, waveform_len]

        # Clamp to valid audio range
        return wav.clamp(min=-1.0, max=1.0)

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Decode long sequences in chunks to avoid OOM.

        Uses overlapping chunks with left context to avoid boundary artifacts.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            chunk_size: Number of codec frames per chunk
            left_context_size: Number of overlapping frames for context
            seq_token_counts: Token count for each request in batch

        Returns:
            list[torch.Tensor]: Complete waveform decoded from the input
                codes. For ``batch_size == 1``, this is a list containing a
                single tensor with shape ``[1, waveform_len]``.
        """
        wavs = []
        start_index = 0

        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index >= left_context_size else start_index

            # Extract chunk with left context
            codes_chunk = codes[..., start_index - context_size : end_index]

            # Decode chunk
            wav_chunk = self(codes_chunk)

            # Remove context from output (context_size * total_upsample samples)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])

            start_index = end_index

        if seq_token_counts is not None:
            code_seq_lens = [seq_len // self.config.num_quantizers for seq_len in seq_token_counts]
        else:
            # Fallback: assume all batch elements share the same sequence length.
            code_seq_lens = [codes.shape[-1]] * codes.shape[0]
        batch_wav = torch.cat(wavs, dim=-1)
        wavs = []
        for idx, code_seq_len in enumerate(code_seq_lens):
            wav_chunk = batch_wav[idx, :, : code_seq_len * self.total_upsample]
            wavs.append(wav_chunk)
        return wavs

    def chunked_decode_streaming(
        self,
        codes: torch.Tensor,
        left_context_size: list[int] | None = None,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Decode long sequences in chunks to avoid OOM.

        Uses overlapping chunks with left context to avoid boundary artifacts.

        No longer need chunk size here, which is different from chunked_decode

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            left_context_size: Number of overlapping frames for context
            seq_token_counts: Token count for each request in batch

        Returns:
            list[torch.Tensor]: Complete waveform decoded from the input
                codes. For ``batch_size == 1``, this is a list containing a
                single tensor with shape ``[1, waveform_len]``.
        """
        # region agent log
        _agent_debug_log(
            run_id="noise-debug",
            hypothesis_id="H1_H3_H5",
            location="qwen3_omni_code2wav.py:chunked_decode_streaming:entry",
            message="code2wav streaming metadata",
            data={
                "codes_shape": list(codes.shape),
                "left_context_len": len(left_context_size) if isinstance(left_context_size, list) else None,
                "left_context_tail": left_context_size[-4:] if isinstance(left_context_size, list) else None,
                "seq_token_counts_len": len(seq_token_counts) if isinstance(seq_token_counts, list) else None,
                "seq_token_counts_tail": seq_token_counts[-4:] if isinstance(seq_token_counts, list) else None,
            },
        )
        # endregion
        if not (left_context_size and seq_token_counts and len(left_context_size) == len(seq_token_counts)):
            # region agent log
            _agent_debug_log(
                run_id="noise-debug",
                hypothesis_id="H1",
                location="qwen3_omni_code2wav.py:chunked_decode_streaming:fallback",
                message="left_context_size fallback to zeros",
                data={
                    "codes_batch": int(codes.shape[0]),
                    "left_context_size": left_context_size,
                    "seq_token_counts": seq_token_counts,
                },
            )
            # endregion
            logger.warning(
                "chunked_decode_streaming: missing/invalid left_context_size or seq_token_counts; "
                "defaulting to left_context_size=zeros(len=codes.shape[0])."
            )
            left_context_size = [0] * codes.shape[0]
        # Decode chunk
        wavs = []
        batch_wav = self(codes)
        if seq_token_counts is not None:
            code_seq_lens = [n // self.config.num_quantizers for n in seq_token_counts]
        else:
            # Fallback: assume all batch elements share the same sequence length.
            code_seq_lens = [codes.shape[-1]] * codes.shape[0]
        model_latency = max(0, int(codes.shape[-1] * self.total_upsample - batch_wav.shape[-1]))
        # region agent log
        _agent_debug_log(
            run_id="noise-debug",
            hypothesis_id="H2_H4",
            location="qwen3_omni_code2wav.py:chunked_decode_streaming:derived_lens",
            message="derived code/audio slicing lengths",
            data={
                "code_seq_lens": code_seq_lens,
                "left_context_size": left_context_size,
                "total_upsample": int(self.total_upsample),
                "model_latency_samples": model_latency,
                "batch_wav_len": int(batch_wav.shape[-1]),
            },
        )
        # endregion
        # region agent log
        codebook_size = int(self.config.codebook_size)
        code_rows = []
        for idx, code_seq_len in enumerate(code_seq_lens):
            valid_t = min(int(code_seq_len), int(codes.shape[-1]))
            row = codes[idx, :, :valid_t]
            invalid_hi = int((row >= codebook_size).sum().item()) if valid_t > 0 else 0
            invalid_lo = int((row < 0).sum().item()) if valid_t > 0 else 0
            repeat_ratio = (
                float((row[:, 1:] == row[:, :-1]).float().mean().item()) if valid_t > 1 else None
            )
            new_start = int(left_context_size[idx]) if idx < len(left_context_size) else 0
            boundary_jump_mean = None
            if 0 < new_start < valid_t:
                prev_col = row[:, new_start - 1].float()
                curr_col = row[:, new_start].float()
                boundary_jump_mean = float((curr_col - prev_col).abs().mean().item())
            code_rows.append(
                {
                    "idx": idx,
                    "valid_t": valid_t,
                    "invalid_hi": invalid_hi,
                    "invalid_lo": invalid_lo,
                    "repeat_ratio": repeat_ratio,
                    "boundary_jump_mean": boundary_jump_mean,
                    "q0_head": row[0, : min(4, valid_t)].tolist() if valid_t > 0 else [],
                    "q0_tail": row[0, max(0, valid_t - 4) : valid_t].tolist() if valid_t > 0 else [],
                }
            )
        _agent_debug_log(
            run_id="noise-debug",
            hypothesis_id="H7_H8_H9",
            location="qwen3_omni_code2wav.py:chunked_decode_streaming:code_stats",
            message="codec token integrity stats",
            data={
                "codebook_size": codebook_size,
                "codes_shape": list(codes.shape),
                "code_rows": code_rows,
            },
        )
        # endregion
        # Compare batched vs per-request decode for the first few multi-request calls.
        compare_budget = int(getattr(self, "_agent_batched_compare_budget", 2))
        if codes.shape[0] > 1 and compare_budget > 0:
            setattr(self, "_agent_batched_compare_budget", compare_budget - 1)
            compare_rows: list[dict] = []
            try:
                for idx, code_seq_len in enumerate(code_seq_lens):
                    start_sample = left_context_size[idx] * self.total_upsample
                    batch_end_sample = min(code_seq_len * self.total_upsample, batch_wav.shape[-1])
                    batch_slice = batch_wav[idx, :, start_sample:batch_end_sample]
                    single_wav = self(codes[idx : idx + 1])
                    single_end_sample = min(code_seq_len * self.total_upsample, single_wav.shape[-1])
                    single_slice = single_wav[0, :, start_sample:single_end_sample]
                    compare_len = min(batch_slice.shape[-1], single_slice.shape[-1])
                    if compare_len <= 0:
                        compare_rows.append(
                            {
                                "idx": idx,
                                "compare_len": int(compare_len),
                                "max_abs_diff": None,
                                "mean_abs_diff": None,
                            }
                        )
                        continue
                    diff = (batch_slice[..., :compare_len] - single_slice[..., :compare_len]).abs().float()
                    compare_rows.append(
                        {
                            "idx": idx,
                            "compare_len": int(compare_len),
                            "max_abs_diff": float(diff.max().item()),
                            "mean_abs_diff": float(diff.mean().item()),
                        }
                    )
            except Exception as e:
                compare_rows = [{"error": str(e)}]
            # region agent log
            _agent_debug_log(
                run_id="noise-debug",
                hypothesis_id="H6",
                location="qwen3_omni_code2wav.py:chunked_decode_streaming:batch_vs_single",
                message="batched vs per-request decode diff",
                data={
                    "codes_shape": list(codes.shape),
                    "code_seq_lens": code_seq_lens,
                    "left_context_size": left_context_size,
                    "compare_rows": compare_rows,
                },
            )
            # endregion
        for idx, code_seq_len in enumerate(code_seq_lens):
            # Remove context from output (left_context_size * total_upsample samples).
            start_sample = left_context_size[idx] * self.total_upsample
            end_sample = max(start_sample, code_seq_len * self.total_upsample)
            end_sample = min(end_sample, batch_wav.shape[-1])
            wav_chunk = batch_wav[idx, :, start_sample:end_sample]
            wavs.append(wav_chunk)
        # region agent log
        _agent_debug_log(
            run_id="noise-debug",
            hypothesis_id="H4",
            location="qwen3_omni_code2wav.py:chunked_decode_streaming:exit",
            message="streaming decode output stats",
            data={
                "output_audio_lens": [int(w.shape[-1]) for w in wavs],
                "output_abs_max": [float(w.abs().amax().item()) if w.numel() > 0 else 0.0 for w in wavs],
                "output_rms": [
                    float(torch.sqrt((w.float().pow(2).mean())).item()) if w.numel() > 0 else 0.0 for w in wavs
                ],
            },
        )
        # endregion
        return wavs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from HuggingFace checkpoint."""
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "talker."],  # Already loaded above
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Log load summary
        try:
            total_bytes = 0
            for name, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            logger.error("Error logging model load summary")

        return loaded
