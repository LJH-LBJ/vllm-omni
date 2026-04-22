# Fix: thinker→talker assistant bootstrap guarantee

## 目标 / Goal

确保 `_get_talker_assistant_parts` 被调用时，总能收到完整的 bootstrap 包：
- **3 prefill tokens**: `<|im_start|>`(151644) + `assistant`(77091) + `\n`(任意token)，在 `thinker_prefill_embeddings` 中
- **1 first_text embed**: 第一个生成 token 的 embedding，在 `thinker_decode_embeddings[0]` 中

从而删除 `_get_talker_assistant_parts` 里的 zero-padding fallback，保证质量。

---

## 背景日志

```
chunk 0 to 40 of full sequence (total_thinker_tokens=102)
chunk 40 to 98 of full sequence (total_thinker_tokens=102)
chunk 98 to 101 of full sequence (total_thinker_tokens=102)
```

三行日志来自**同一次** `_thinker_to_talker_prefill` 调用内部对 im_start 段的循环迭代（segment loop），  
不是三次独立的网络传输。talker 在一次 prefill 中处理了完整的 102-token 序列，分成三个角色段：
- 0–40: system
- 40–98: user  
- 98–101: assistant header (`<|im_start|>assistant\n`，3 tokens only)

问题：第 3 个 segment (98–101) 调用 `_get_talker_assistant_parts(local_start=0, local_end=3, decode_assistant_fill=None)` ——  
`decode_assistant_fill` 为 None，因为 first_text embedding 尚未到达，触发 zero-padding fallback。

---

## 根本原因

### 关键数据关系

每次 `thinker2talker_async_chunk(chunk_id==0)` 被调用时：
- `request.prompt_token_ids` = **累积全量** prompt（长度随 chunked-prefill 递增）
- `pooling_output["0"]` = **当前批次**的 embedding（仅本批 delta，非累积）
- `batch_size = len(embeds)`
- `current_batch_start = len(prompt_token_ids) - batch_size`（本批在全序列中的起始偏移）

对应日志中的三次调用：

| 调用 | 本批 tokens | `prompt_token_ids` 长度 | assistant boundary | 正确行为 |
|------|------------|------------------------|--------------------|---------|
| Call 1 | 0-39 (40 tokens) | 40 | 无 | **直接发送** |
| Call 2 | 40-97 (58 tokens) | 98 | 无（im_start 在 98，还没到） | **直接发送** |
| Call 3 | 98-100 (3 tokens) | 101 | `last_im=98`，完整（101-98=3≥3） | **buffer，等 decode embed** |

### Bug 1: 现有代码"全部积压"策略

当前代码对所有 `chunk_id==0` 批次统一用 `request_payload` 积压，直到有 `output_token_ids` 才发送。  
这导致：
1. Call 1 和 Call 2（完全不含 assistant 的批次）被不必要地延迟。
2. 3+ 批次时 `request_payload` / `_pending_assistant` 出现分裂（Call 3 绕过 `_pending_assistant` 重新存入 `request_payload`），两份数据永远不合并。

### Bug 2: `_assistant_parts_complete` 过早返回 True

```python
# 当前实现
return len(output_token_ids) > 0  # 只检查 token ID
```

在 **transition step** (chunk_id>=1，is_transition_or_pure_prefill=True)：
- `output_token_ids` 非空 → `_assistant_parts_complete` 返回 True → 立即 flush pending
- 但 `thinker_decode_embeddings`（first_text 的 embedding）**尚未到达**（embedding 在下一个 pure-decode step 才出现）
- 结果：talker 收到的 `thinker_decode_embeddings=None`，`_get_talker_assistant_parts` 进入 fallback

---

## 设计方案

### 核心算法（chunk_id==0）

```
每次 chunk_id==0 调用：

计算 last_im = _find_assistant_boundary(prompt_token_ids)
                  # 在累积 prompt 中找最后一个 <|im_start|>assistant

如果 last_im == -1：
    → 无 assistant segment，直接发送本批 embedding

如果 last_im >= 0：
    assistant_complete = (len(prompt_token_ids) - last_im) >= 3
    current_batch_start = len(prompt_token_ids) - batch_size
    local_im = last_im - current_batch_start   # boundary 在本批内的偏移

    if local_im > 0（boundary 在本批内部）：
        → 本批前半段 embeds[:local_im] 存入 _ready_pre_payload（下次调用优先发送）
        → 本批后半段 embeds[local_im:] 存入 _pending_assistant
        → return None

    if local_im <= 0（boundary 在前批，本批全在 assistant 范围内）：
        → 合并进 _pending_assistant

    if assistant_complete（im_start+assistant+\n 都已在 prompt 中）：
        → 留在 _pending_assistant，等 pure-decode step 提供 decode embed

    if not assistant_complete（\n 还未到达）：
        → 留在 _pending_assistant，继续等下一个 chunk_id==0 批次
```

对应日志中的三次调用（0-40, 40-98, 98-101）：

```
Call 1: last_im=-1 → 直接发 embeds[0:40]                ✓ 无等待
Call 2: last_im=-1 → 直接发 embeds[0:58]                ✓ 无等待
Call 3: last_im=98, local_im=0, assistant_complete=True
        → 存入 _pending_assistant                         ✓ 等 first_text embed
        
pure-decode step (chunk_id>=1):
        → pending + decode_embeds[0:1] → flush → 发送    ✓
```

**假设 boundary 落在批次内部**（如 Call 2 包含 tokens 40-101）：

```
Call 2: last_im=98, local_im=58 (=98-40), assistant_complete=True
        → _ready_pre_payload[rid] = embeds[:58]  (tokens 40-97, user segment)
        → _pending_assistant[rid] = embeds[58:]  (tokens 98-100, assistant header)
        → return None

"Call 3"（实际是下一次 thinker decode 回调，chunk_id==1）：
        → 函数入口先检查 _ready_pre_payload → 发送 user 部分    ✓
        → put_req_chunk 递增到 1

chunk_id==1, 再下一步:
        → pending 存在，pure-decode → flush assistant 部分       ✓
```

---

## 具体修改

### File 1: `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`

#### 1.1 新增辅助函数 `_find_assistant_boundary`

```python
def _find_assistant_boundary(prompt_token_ids: list[int]) -> int:
    """
    Return the index of the last <|im_start|> token that starts an assistant role segment.
    Returns -1 if no assistant segment found.
    
    Example: [..., 151644, 77091, 198, ...] → returns index of 151644
    """
    IM_START = 151644
    ASSISTANT = 77091
    for i in range(len(prompt_token_ids) - 1, -1, -1):
        if prompt_token_ids[i] == IM_START:
            # Check if next token is assistant role
            if i + 1 < len(prompt_token_ids) and prompt_token_ids[i + 1] == ASSISTANT:
                return i
            break  # The last <|im_start|> is not assistant role, stop searching
    return -1
```

#### 1.2 修改 `_assistant_parts_complete`

增加 `has_decode_embed: bool` 参数，要求 embedding 也已到达：

```python
def _assistant_parts_complete(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    has_decode_embed: bool = False,
) -> bool:
    """
    Returns True only when ALL of the following are true:
      1. prompt ends with <|im_start|> assistant \n (3 tokens)
      2. at least one output token ID is available
      3. has_decode_embed == True (the embedding for output_token_ids[0] is available)
    """
    IM_START = 151644
    ASSISTANT = 77091

    last_im = _find_assistant_boundary(prompt_token_ids)
    if last_im == -1:
        return True  # No assistant segment

    assistant_tokens_in_prompt = len(prompt_token_ids) - last_im
    if assistant_tokens_in_prompt < 3:
        return False  # \n not yet in prefill

    # Need the embedding, not just the token ID
    return len(output_token_ids) > 0 and has_decode_embed
```

#### 1.3 重构 `thinker2talker_async_chunk` 中 `chunk_id==0` 分支

**关键改动**：废弃 `request_payload` 积压机制，改为按 boundary 决策。

```python
if chunk_id == 0:
    # ... 构建 talker_additional_info（不变）...
    embeds_cpu = pooling_output.get("0").detach().cpu()
    hidden_cpu = pooling_output.get("24").detach().cpu()
    batch_size = embeds_cpu.shape[0]
    full_prompt_len = len(prompt_token_ids)
    current_batch_start = full_prompt_len - batch_size  # 本批在全序列中的起始偏移

    # === 检查 assistant boundary ===
    last_im = _find_assistant_boundary(prompt_token_ids)

    if last_im == -1:
        # 无 assistant segment，直接发送
        talker_additional_info["thinker_prefill_embeddings"] = embeds_cpu
        talker_additional_info["thinker_hidden_states"] = hidden_cpu
        return talker_additional_info

    assistant_complete = (full_prompt_len - last_im) >= 3  # im_start + assistant + \n
    local_im = last_im - current_batch_start  # boundary 在本批内的 local 偏移

    # 合并之前已积累的 pending（若有）
    pending = transfer_manager._pending_assistant.pop(request_id, None)

    if local_im > 0:
        # Boundary 在本批内部：前半段可立即发，后半段要 buffer
        pre_payload = dict(talker_additional_info)
        pre_payload["thinker_prefill_embeddings"] = embeds_cpu[:local_im]
        pre_payload["thinker_hidden_states"] = hidden_cpu[:local_im]
        pre_payload["finished"] = torch.tensor(False, dtype=torch.bool)
        if not hasattr(transfer_manager, "_ready_pre_payload"):
            transfer_manager._ready_pre_payload = {}
        transfer_manager._ready_pre_payload[request_id] = pre_payload

        asst_payload = dict(talker_additional_info)
        asst_payload["thinker_prefill_embeddings"] = embeds_cpu[local_im:]
        asst_payload["thinker_hidden_states"] = hidden_cpu[local_im:]
        if pending is not None:
            asst_payload = _merge_prefill_payloads(pending, asst_payload)
        transfer_manager._pending_assistant[request_id] = asst_payload
        return None

    # local_im <= 0: 本批全在 assistant 范围内（boundary 在前批或恰在起始）
    cur_payload = dict(talker_additional_info)
    cur_payload["thinker_prefill_embeddings"] = embeds_cpu
    cur_payload["thinker_hidden_states"] = hidden_cpu
    if pending is not None:
        cur_payload = _merge_prefill_payloads(pending, cur_payload)
    # assistant header 完整 → 等 decode embed；不完整 → 等下一个 chunk_id==0
    transfer_manager._pending_assistant[request_id] = cur_payload
    return None
```

> **注意**：`request_payload` 在此路径不再使用。原来的 `elif not finished` 保存逻辑完全由上述逻辑替代。

#### 1.4 chunk_id>=1 pending 分支修改

修改 `_assistant_parts_complete` 的调用，传入 `has_decode_embed`：

```python
# 原来：
if not _assistant_parts_complete(prompt_token_ids_now, output_token_ids):
    return None

# 改为：
is_pure_decode = not is_transition_or_pure_prefill
has_decode_embed = (
    is_pure_decode
    and isinstance(embeds, torch.Tensor)
    and embeds.numel() > 0
)
if not _assistant_parts_complete(prompt_token_ids_now, output_token_ids, has_decode_embed):
    pending["finished"] = torch.tensor(finished, dtype=torch.bool)
    return None
```

在 flush 时，只有 pure-decode step 才附加 `thinker_decode_embeddings`（不变，已正确）。  
**删除 transition step 的 flush**（因为 has_decode_embed=False，transition step 不再满足条件）。

---

### File 2: `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py`

#### 2.1 修改 `_get_talker_assistant_parts`

确认 connector 已保证 `decode_assistant_fill` 总是可用，删除 zero-padding fallback，改为 assertion：

```python
def _get_talker_assistant_parts(
    self, im_start_index, segment_end_index, speaker_id, thinker_embed,
    tts_pad_embed, tts_bos_embed, tts_eos_embed,
    request_id=None, decode_assistant_fill=None,
):
    assistant_hidden = self.talker.text_projection(
        thinker_embed[im_start_index:segment_end_index]
    ).to(tts_pad_embed.device)  # [t, d], t >= 3
    self._assistant_decode_fill_consumed = 0

    has_decode = (
        isinstance(decode_assistant_fill, torch.Tensor)
        and decode_assistant_fill.ndim >= 2
        and int(decode_assistant_fill.shape[0]) >= 1
    )
    if has_decode:
        first_text_embed = self.talker.text_projection(
            decode_assistant_fill[0:1].to(tts_pad_embed.device)
        ).to(assistant_hidden.dtype)
        assistant_hidden = torch.cat([assistant_hidden, first_text_embed], dim=0)
        self._assistant_decode_fill_consumed = 1
    else:
        # This should NOT happen if the connector guarantees complete bootstrap.
        # Log a warning instead of silently zero-padding.
        logger.warning(
            "[_get_talker_assistant_parts] req=%s: decode_assistant_fill missing! "
            "This indicates a connector bug. Falling back to zero-pad.",
            request_id,
        )
        hidden_dim = assistant_hidden.shape[-1]
        zero = torch.zeros((1, hidden_dim), device=tts_pad_embed.device, dtype=assistant_hidden.dtype)
        assistant_hidden = torch.cat([assistant_hidden, zero], dim=0)
        self._assistant_decode_fill_consumed = 0

    # ... 其余逻辑不变 ...
```

---

## `_ready_pre_payload` 入口检查

`thinker2talker_async_chunk` 函数**最开始**（任何 chunk_id 之前）加入：

```python
# 优先发送上一次拆分出来的 pre-assistant 部分
if not hasattr(transfer_manager, "_ready_pre_payload"):
    transfer_manager._ready_pre_payload = {}
if request_id in transfer_manager._ready_pre_payload:
    return transfer_manager._ready_pre_payload.pop(request_id)
```

发送 pre-part 后，`put_req_chunk` 递增为 1。下一次 thinker 回调（transition/decode step）进入 `chunk_id>=1` 分支，走 `pending is not None` 路径，提供 `decode_embeds[0:1]` 后 flush `_pending_assistant`。

---

## 关键辅助函数 `_merge_prefill_payloads`

```python
def _merge_prefill_payloads(old: dict, new: dict) -> dict:
    """Merge two chunk_id==0 payloads by concatenating prefill embeddings."""
    merged = {**new}
    merged["thinker_prefill_embeddings"] = torch.cat(
        (old["thinker_prefill_embeddings"], new["thinker_prefill_embeddings"]), dim=0
    )
    merged["thinker_hidden_states"] = torch.cat(
        (old["thinker_hidden_states"], new["thinker_hidden_states"]), dim=0
    )
    # tts_*_embed: keep from old (same for entire request)
    for k in ("tts_bos_embed", "tts_eos_embed", "tts_pad_embed", "speaker", "language"):
        if k in old and k not in new:
            merged[k] = old[k]
    return merged
```

---

## cleanup_sender 修改

在 `OmniChunkTransferAdapter.cleanup_sender` 中添加对新字典的清理：

```python
self._pending_assistant.pop(external_req_id, None)  # 已存在
self._ready_pre_payload.pop(external_req_id, None)   # 新增（方案 A）
```

---

## 不变量（Invariants）

实现完成后，以下不变量应成立：

1. **进入 `_get_talker_assistant_parts` 时**：
   - `thinker_embed[im_start_index:segment_end_index].shape[0] >= 3`（im_start+assistant+\n 已在 prefill）
   - `decode_assistant_fill` 是形状 `[≥1, hidden_dim]` 的 tensor（first_text embed 可用）
   - → `has_decode == True`，`_assistant_decode_fill_consumed == 1`
   - → zero-padding fallback 永远不触发

2. **`_assistant_parts_complete(..., has_decode_embed=True)`** 只在 pure-decode step 下返回 True。  
   Transition step 中：`has_decode_embed=False` → 函数返回 False → 继续缓存。

3. **3+ prefill chunks 场景（如 logs 所示）**：  
   数据始终在 `_pending_assistant` 中正确累积，`request_payload` 不与之冲突。

---

## 文件清单

| 文件 | 修改内容 |
|------|---------|
| `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py` | `_find_assistant_boundary`（新增）；`_assistant_parts_complete`（加 `has_decode_embed` 参数）；`_merge_prefill_payloads`（新增）；`thinker2talker_async_chunk`（修复 Bug 1+2，可选方案 A） |
| `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py` | `_get_talker_assistant_parts`（fallback → warning-only） |
| `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py` | `__init__` 加 `_ready_pre_payload`（方案 A）；`cleanup_sender` 加清理 |
