"""Qwen3-Omni Code Predictor with MTP (Multi-Token Prediction) support.

This module implements the code predictor component for Qwen3-Omni talker models.

The code predictor generates residual RVQ (Residual Vector Quantization) codes
autoregressively, predicting layers 1 to N based on layer-0 codes from the talker.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = init_logger(__name__)

# ============================================================================
# Code Predictor Attention Layer
# ============================================================================


class Qwen3OmniCodePredictorAttention(nn.Module):
    """Standalone multi-head self-attention for code predictor.

    Uses F.scaled_dot_product_attention (SDPA) directly instead of
    the HF backend-fallback loop.  No KV cache — the code predictor
    always re-prefills the full (short) sequence each step.

    Input : [B, seq_len, hidden_size]
    Output: [B, seq_len, hidden_size]
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        cp_cfg = config.code_predictor_config
        self.num_heads = cp_cfg.num_attention_heads
        self.num_kv_heads = cp_cfg.num_key_value_heads
        self.head_dim = getattr(
            cp_cfg,
            "head_dim",
            cp_cfg.hidden_size // cp_cfg.num_attention_heads,
        )
        self.hidden_size = cp_cfg.hidden_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self._use_gqa = self.num_kv_heads != self.num_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=cp_cfg.max_position_embeddings,
            rope_parameters=None,
            dual_chunk_attention_config=None,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=cp_cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cp_cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # Flatten to 2D before qkv_proj so q/k are [num_tokens, size],
        # which is the shape vLLM's rotary_emb expects.
        qkv, _ = self.qkv_proj(hidden_states.reshape(bsz * seq_len, -1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # RMSNorm over head_dim, then RoPE (both operate on 2D [num_tokens, size])
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)

        q, k = self.rotary_emb(position_ids, q, k)

        # Reshape to [B, heads, seq, head_dim] for SDPA
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scaling,
            is_causal=True,
            enable_gqa=self._use_gqa,
        )

        attn_out = attn_out.transpose(1, 2).reshape(bsz * seq_len, -1)
        output, _ = self.o_proj(attn_out)
        return output.view(bsz, seq_len, -1)


# ============================================================================
# Code Predictor MLP Layer
# ============================================================================


class Qwen3OmniCodePredictorMLP(nn.Module):
    """Feed-forward network for code predictor with fused gate/up projection."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.code_predictor_config.hidden_size
        intermediate_size = config.code_predictor_config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )

        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        down, _ = self.down_proj(F.silu(gate) * up)
        return down


# ============================================================================
# MTP Layer (Multi-Token Prediction Layer)
# ============================================================================


class Qwen3OmniCodePredictorMTPLayer(nn.Module):
    """Transformer decoder layer for code predictor (SDPA, no KV cache)."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3OmniCodePredictorAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3OmniCodePredictorMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        cp_cfg = config.code_predictor_config
        self.input_layernorm = RMSNorm(cp_cfg.hidden_size, eps=cp_cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cp_cfg.hidden_size, eps=cp_cfg.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3OmniCodePredictorBaseModel(nn.Module):
    """
    Base model for code predictor - matches HF Qwen3OmniMoeTalkerCodePredictorModel structure.

    This is a simple transformer that processes inputs_embeds and outputs hidden states.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.code_predictor_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups

        # Codec embeddings (for layers 1-num_code_groups-1)
        self.codec_embedding = nn.ModuleList(
            [
                VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                )
                for _ in range(config.num_code_groups - 1)
            ]
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3OmniCodePredictorMTPLayer(
                    vllm_config.model_config.hf_config,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs_embeds: [batch, seq_len, hidden_size]
            position_ids: Flat position IDs for RoPE

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_input_embeddings(self):
        """Return codec embeddings for HF compatibility."""
        return self.codec_embedding


class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    """
    Code predictor wrapper matching HF structure.

    Structure:
    - self.model: Qwen3OmniCodePredictorBaseModel (transformer)
    - self.lm_head: ModuleList of output heads
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        talker_code_predictor_config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.config = talker_code_predictor_config
        self.vocab_size = self.config.code_predictor_config.vocab_size
        self.num_code_groups = self.config.code_predictor_config.num_code_groups

        # Base transformer model (matches HF structure)
        # prefix must include ".model" to match PyTorch's parameter hierarchy
        # (e.g. "code_predictor.model.layers.0.self_attn.qkv_proj") for
        # correct quantization config lookup and TP sharding.
        self.model = Qwen3OmniCodePredictorBaseModel(vllm_config=vllm_config, prefix=f"{prefix}.model")

        # Output heads for each residual layer (1-num_layers-1)
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(
                    self.config.code_predictor_config.hidden_size,
                    self.config.code_predictor_config.vocab_size,
                    bias=False,
                )
                for _ in range(self.num_code_groups - 1)
            ]
        )
        self.logits_processors = LogitsProcessorList(
            [
                TopKLogitsWarper(top_k=50),
                TopPLogitsWarper(top_p=0.8),
            ]
        )

        # Store hidden_size for use in forward() buffer allocation
        self._hidden_size = self.config.code_predictor_config.hidden_size

        # Lazy torch.compile on the inner transformer only (not the outer
        # autoregressive loop + sampling).  mode="default" avoids conflicts
        # with vLLM's CUDAGraphWrapper on the main stream.
        self._compiled_model_fwd: object | None = None

    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for code predictor (zero-allocation inner loop).

        Uses pre-allocated buffers (_proj_buf, _all_codes, _pos_ids) to
        eliminate torch.cat / torch.stack in the autoregressive loop.

        Args:
            layer0_code:        [batch_size, 1]  int64
            layer0_embed:       [batch_size, 1, hidden_size]
                Layer-0 codec embedding, produced by the **talker's**
                language_model.model.codec_embedding (NOT code_predictor's).
            last_talker_hidden: [batch_size, 1, hidden_size]
                Last hidden state from the talker transformer.

        Returns:
            all_codes:     [batch_size, num_code_groups, 1]
                Predicted codes for all RVQ layers (layer-0 is input, rest are predicted).
            proj_buf_view: [batch_size, num_code_groups + 1, hidden_size]
                The projection buffer used as transformer input.
                Buffer layout (num_code_groups + 1 positions total):

                  pos 0   : last_talker_hidden  ← NOT a codec embedding
                  pos 1   : layer0_embed        ← talker's codec_embedding(layer0_code)
                  pos 2   : codec_embedding[0](layer1_code)  ← code_predictor's
                  pos 3   : codec_embedding[1](layer2_code)  ← code_predictor's
                  ...     : ...
                  pos G   : codec_embedding[G-2](layerG-1_code)

                where G = num_code_groups.

                Caller can get the summed codec embedding by:
                    proj_buf_view[:, 1:, :].sum(dim=1)
                which sums positions 1..G, i.e. all G codec embeddings,
                skipping position 0 (last_talker_hidden, not a codec embed).
        """
        batch_size = layer0_code.shape[0]
        device = layer0_code.device
        dtype = last_talker_hidden.dtype

        # Allocate scratch tensors on the correct device each call.
        # register_buffer is intentionally avoided: registered buffers captured
        # as graph inputs by torch.compile make the compiled artifact
        # non-serializable in vLLM's compilation cache.
        # CUDAGraph replay captures these fixed-shape allocations, so the
        # per-call overhead is negligible during inference.
        proj_buf = torch.zeros(
            batch_size,
            self.num_code_groups + 1,
            self._hidden_size,
            device=device,
            dtype=dtype,
        )
        all_codes = torch.empty(
            batch_size,
            self.num_code_groups,
            1,
            dtype=torch.int64,
            device=device,
        )
        pos_ids = torch.arange(self.num_code_groups + 1, dtype=torch.int64, device=device)

        # ---- Fill buffer (replaces torch.cat) ----
        # pos 0 ← talker hidden state (drives code predictor, NOT a codec embed)
        proj_buf[:, 0:1, :] = last_talker_hidden
        # pos 1 ← layer-0 codec embed (from talker's codec_embedding)
        proj_buf[:, 1:2, :] = layer0_embed

        # Write layer-0 code into all_codes (replaces list + torch.stack)
        all_codes[:, 0] = layer0_code

        # Lazily set up torch.compile on the inner transformer forward.
        # Only the 5-layer transformer is compiled; the outer loop + sampling
        # stay in eager mode (varying seq_len + multinomial are not compile-friendly).
        if self._compiled_model_fwd is None:
            self._compiled_model_fwd = torch.compile(
                self.model.forward,
                mode="default",
                dynamic=True,
            )
            logger.info("code_predictor: torch.compile enabled (mode=default)")
        model_fwd = self._compiled_model_fwd

        # Cache module references to bypass nn.Module.__call__ overhead
        lm_heads = list(self.lm_head)
        codec_embeds = list(self.model.codec_embedding)

        # ---- Autoregressive loop: predict layers 1..num_code_groups-1 ----
        for layer_idx in range(self.num_code_groups - 1):
            seq_len = layer_idx + 2  # positions 0..layer_idx+1 are filled

            # View into buffer – no copy, no allocation
            projected = proj_buf[:, :seq_len, :]
            # Slice pre-computed position IDs (1D flat)
            position_ids = pos_ids[:seq_len] if batch_size == 1 else pos_ids[:seq_len].repeat(batch_size)

            hidden_out = model_fwd(projected, position_ids)

            # Sample next code from the last position's logits (inline, no custom op)
            logits = lm_heads[layer_idx](hidden_out[:, -1, :])
            logits = self.logits_processors(None, logits)
            probs = F.softmax(logits, dim=-1)
            code = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # Write predicted code directly into pre-allocated tensor
            all_codes[:, layer_idx + 1] = code

            # Write embedding into buffer.  This is needed both for the next
            # transformer iteration AND because proj_buf is returned to the
            # caller who sums proj_buf[:, 1:, :] over all codec positions.
            new_embed = codec_embeds[layer_idx](code)
            proj_buf[:, layer_idx + 2 : layer_idx + 3, :] = new_embed

        return all_codes, proj_buf

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with mapping for fused QKV and gate_up projections.

        Maps original HF weights (q_proj, k_proj, v_proj, gate_proj, up_proj)
        to fused vLLM weights (qkv_proj, gate_up_proj).
        """
        # Mapping for fused projections
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip rotary embeddings
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle stacked/fused parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip if parameter doesn't exist (e.g., bias)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Non-stacked parameters - use default loading
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight)

            loaded_params.add(name)

        return loaded_params
