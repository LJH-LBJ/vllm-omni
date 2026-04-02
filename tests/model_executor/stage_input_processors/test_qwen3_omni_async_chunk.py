# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tm():
    return SimpleNamespace(
        put_req_chunk=defaultdict(int),
        request_payload={},
    )


def _req(rid: str, prompt_len: int):
    # Prompt and all-token ids are only used for length checks / metadata packing.
    return SimpleNamespace(
        external_req_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        all_token_ids=list(range(prompt_len)),
        output_token_ids=[],
    )


def _pool(prefill_rows: int):
    h = 4
    return {
        "0": torch.arange(prefill_rows * h, dtype=torch.float32).reshape(prefill_rows, h),
        "24": torch.arange(prefill_rows * h, dtype=torch.float32).reshape(prefill_rows, h),
        "tts_bos_embed": torch.zeros((1, 1, h), dtype=torch.float32),
        "tts_eos_embed": torch.zeros((1, 1, h), dtype=torch.float32),
        "tts_pad_embed": torch.zeros((1, 1, h), dtype=torch.float32),
    }


def test_thinker2talker_async_chunk_keeps_accumulating_until_full_prompt():
    tm = _tm()
    rid = "req-1"
    req = _req(rid, prompt_len=6)

    # chunk-1: store and wait
    out1 = thinker2talker_async_chunk(tm, _pool(prefill_rows=2), req, is_finished=False)
    assert out1 is None
    assert rid in tm.request_payload
    assert tm.request_payload[rid]["thinker_prefill_embeddings"].shape[0] == 2

    # chunk-2: still not enough (4 < prompt_len=6), should keep waiting
    out2 = thinker2talker_async_chunk(tm, _pool(prefill_rows=2), req, is_finished=False)
    assert out2 is None
    assert rid in tm.request_payload
    assert tm.request_payload[rid]["thinker_prefill_embeddings"].shape[0] == 4

    # chunk-3: now enough (6 == prompt_len), should emit payload
    out3 = thinker2talker_async_chunk(tm, _pool(prefill_rows=2), req, is_finished=False)
    assert out3 is not None
    assert out3["thinker_prefill_embeddings"].shape[0] == 6
    assert out3["thinker_hidden_states"].shape[0] == 6
    assert rid not in tm.request_payload
