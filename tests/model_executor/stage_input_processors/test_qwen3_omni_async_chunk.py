# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
    _compute_partial_talker_prompt_ids_length,
    thinker2talker_async_chunk,
)

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


def test_thinker2talker_async_chunk_sends_first_prefill_chunk_immediately():
    tm = _tm()
    rid = "req-1"
    req = _req(rid, prompt_len=6)

    # chunk-1 should be emitted immediately for pipelined thinker->talker prefill.
    out1 = thinker2talker_async_chunk(tm, _pool(prefill_rows=2), req, is_finished=False)
    assert out1 is not None
    assert out1["thinker_prefill_embeddings"].shape[0] == 2
    assert rid not in tm.request_payload

    # Simulate sender increment after a successful connector.put.
    tm.put_req_chunk[rid] = 1

    # chunk-2 prefill update should still be delivered via chunk_id>0 path.
    out2 = thinker2talker_async_chunk(tm, _pool(prefill_rows=2), req, is_finished=False)
    assert out2 is not None
    assert out2["thinker_prefill_embeddings"].shape[0] == 2
    assert "thinker_input_ids" not in out2
    assert "thinker_sequences" not in out2

def test_compute_partial_talker_prompt_ids_length_progresses_with_received_prefill():
    # Prompt structure: [im_start, user, a, b, im_start, assistant, c, d, e, f]
    # For this shape, full talker prefill len is user span (4) + assistant bootstrap (9) = 13.
    im_start = 151644
    user = 872
    assistant = 77091
    prompt_ids = [im_start, user, 1, 2, im_start, assistant, 3, 4, 5, 6]

    assert _compute_partial_talker_prompt_ids_length(prompt_ids, 0) == 0
    assert _compute_partial_talker_prompt_ids_length(prompt_ids, 1) == 0
    assert _compute_partial_talker_prompt_ids_length(prompt_ids, 4) == 4
    # Assistant bootstrap requires 4 assistant thinker tokens from segment start.
    assert _compute_partial_talker_prompt_ids_length(prompt_ids, 8) == 4
    assert _compute_partial_talker_prompt_ids_length(prompt_ids, 10) == 13
