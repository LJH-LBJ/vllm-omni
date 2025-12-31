from __future__ import annotations

import enum
import fcntl
import json
import logging
import multiprocessing as mp
import os
import traceback
from multiprocessing import shared_memory as _shm
from typing import Any

from omegaconf import OmegaConf

from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector
from vllm_omni.distributed.omni_connectors.utils.initialization import build_stage_connectors
from vllm_omni.entrypoints.log_utils import count_tokens_from_outputs
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion
from vllm_omni.entrypoints.omni_llm import OmniLLM
from vllm_omni.entrypoints.omni_stage import prepare_sampling_params

logger = logging.getLogger(__name__)


class OmniStageTaskType(enum.Enum):
    GENERATE = "generate"
    ABORT = "abort"
    SHUTDOWN = "shutdown"


SHUTDOWN_TASK = {"type": OmniStageTaskType.SHUTDOWN}


def set_stage_devices(
    stage_id: int,
    devices: str | int | None,
    device_type: str | None = None,
) -> None:
    """Configure per-stage device visibility and current device (CUDA or NPU).

    This function sets environment variables that control which devices are visible
    to the process, and sets the current device. It must be called BEFORE worker
    initialization so that workers see the correct devices.

    Args:
        stage_id: Stage identifier for logging
        devices: Device specification:
            - Comma-separated string (e.g. "2,5,7"): interpreted as logical
              indices against the current device visibility env var (e.g.
              CUDA_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES) when present;
              falls back to physical IDs if no mapping exists. Logical index 0
              is used as current device.
            - Integer or digit-string: treat as logical index (0-based) into the
              current device visibility mapping; map to physical device, then set
              env var to this single device.
            - None/"cpu": keep default visibility.
            - Otherwise: set env var to the provided single device string.
        device_type: Device type ("cuda" or "npu"). If None, auto-detects.

    Behavior:
        - CUDA: Sets CUDA_VISIBLE_DEVICES and calls torch.cuda.set_device()
        - NPU: Sets ASCEND_RT_VISIBLE_DEVICES and calls torch.npu.set_device()
    """
    from vllm_omni.utils import detect_device_type, get_device_control_env_var

    if device_type is None:
        device_type = detect_device_type()

    env_var = get_device_control_env_var()

    # Select device-specific torch functions
    if device_type == "npu":
        try:
            import torch.npu  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("[Stage-%s] torch.npu not available, skipping NPU device setup", stage_id)
            return

        is_available_fn = torch.npu.is_available
        set_device_fn = torch.npu.set_device
        device_count_fn = torch.npu.device_count
        get_device_properties_fn = torch.npu.get_device_properties
        mem_get_info_fn = torch.npu.mem_get_info
        get_device_name_fn = torch.npu.get_device_name
        device_type_label = "NPU"
    elif device_type == "cuda":
        import torch

        is_available_fn = torch.cuda.is_available
        set_device_fn = torch.cuda.set_device
        device_count_fn = torch.cuda.device_count
        get_device_properties_fn = torch.cuda.get_device_properties
        mem_get_info_fn = torch.cuda.mem_get_info
        get_device_name_fn = torch.cuda.get_device_name
        device_type_label = "CUDA"
    else:
        logger.debug("[Stage-%s] Unsupported device type: %s", stage_id, device_type)
        return

    try:
        selected_physical: int | None = None
        logical_idx: int | None = None

        if isinstance(devices, str) and "," in devices:
            toks = [t.strip() for t in devices.split(",") if t.strip() != ""]
            vis = os.environ.get(env_var)
            mapped_devices: list[str] = []
            mapping: list[int] = []
            if vis:
                try:
                    mapping = [int(x) for x in vis.split(",") if x.strip() != ""]
                except Exception as e:
                    logger.debug("[Stage-%s] Failed to parse existing %s: %s", stage_id, env_var, e)
            for tok in toks:
                try:
                    idx = int(tok)
                except Exception:
                    mapped_devices.append(tok)
                    continue
                if mapping and 0 <= idx < len(mapping):
                    mapped_devices.append(str(mapping[idx]))
                else:
                    mapped_devices.append(str(idx))
            mapped_devices_str = ",".join(mapped_devices)
            os.environ[env_var] = mapped_devices_str
            if toks:
                try:
                    selected_physical = int(mapped_devices[0])
                    logger.debug(
                        "[Stage-%s] Set %s to %s; logical 0 -> physical %s",
                        stage_id,
                        env_var,
                        mapped_devices_str,
                        selected_physical,
                    )
                except Exception as e:
                    logger.debug("[Stage-%s] Failed to parse first %s device: %s", stage_id, device_type_label, e)
                    selected_physical = None
        elif isinstance(devices, (int, str)) and (isinstance(devices, int) or str(devices).isdigit()):
            logical_idx = max(0, int(devices))
            vis = os.environ.get(env_var)
            if vis:
                try:
                    mapping = [int(x) for x in vis.split(",") if x.strip() != ""]
                    if 0 <= logical_idx < len(mapping):
                        selected_physical = mapping[logical_idx]
                except Exception as e:
                    logger.debug("[Stage-%s] Failed to map logical index via %s: %s", stage_id, env_var, e)
                    selected_physical = None
            if selected_physical is None:
                selected_physical = int(logical_idx)
            os.environ[env_var] = str(selected_physical)
            logger.debug(
                "[Stage-%s] Logical index %d -> physical %s; set %s to single device",
                stage_id,
                logical_idx + 1,
                selected_physical,
                env_var,
            )
        elif devices in (None, "cpu"):
            logger.debug("[Stage-%s] Using default device visibility (devices=%s)", stage_id, devices)
        else:
            selected_physical = int(str(devices))
            os.environ[env_var] = str(selected_physical)
            logger.debug("[Stage-%s] Set %s to single device %s (fallback)", stage_id, env_var, selected_physical)

        try:
            import torch

            if is_available_fn():
                try:
                    set_device_fn(0)
                except Exception as e:
                    logger.debug(
                        "[Stage-%s] %s set_device(0) failed: %s", stage_id, device_type_label, e, exc_info=True
                    )
                num = device_count_fn()
                info = []
                for i in range(num):
                    total = get_device_properties_fn(i).total_memory
                    free, _ = mem_get_info_fn(i)
                    info.append(
                        {
                            "idx": i,
                            "name": get_device_name_fn(i),
                            "total": int(total),
                            "free": int(free),
                        }
                    )
                logger.debug("[Stage-%s] %s devices visible=%s info=%s", stage_id, device_type_label, num, info)
        except Exception as e:
            logger.debug("[Stage-%s] Failed to query %s devices: %s", stage_id, device_type_label, e, exc_info=True)
    except Exception as e:
        logger.warning("Failed to interpret devices for stage %s: %s", stage_id, e)


def serialize_obj(obj: Any) -> bytes:
    """Serialize a Python object to bytes using centralized serializer (defaults to cloudpickle)."""
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    return OmniSerializer.serialize(obj)


def shm_write_bytes(payload: bytes) -> dict[str, Any]:
    """Write bytes into SharedMemory and return meta dict {name,size}.

    Caller should close the segment; the receiver should unlink.
    """
    shm = _shm.SharedMemory(create=True, size=len(payload))
    mv = memoryview(shm.buf)
    mv[: len(payload)] = payload
    del mv
    meta = {"name": shm.name, "size": len(payload)}
    try:
        shm.close()
    except Exception as e:
        logger.debug("Failed to close shared memory: %s", e)
    return meta


def shm_read_bytes(meta: dict[str, Any]) -> bytes:
    """Read bytes from SharedMemory by meta {name,size} and cleanup."""
    shm = _shm.SharedMemory(name=meta["name"])  # type: ignore[index]
    mv = memoryview(shm.buf)
    data = bytes(mv[: meta["size"]])
    del mv
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass
    return data


def _ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory for a file path exists (best-effort)."""
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append a JSON record as one line to a JSONL file (best-effort).

    This is safe to call from multiple processes when each process writes
    to a distinct file. For concurrent writes to the same file, OS append
    semantics typically suffice, but no additional locking is provided.
    """
    try:
        _ensure_parent_dir(path)
        line = json.dumps(record, ensure_ascii=False)
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        logger.exception("Failed to append JSONL to %s", path)


def maybe_dump_to_shm(obj: Any, threshold: int) -> tuple[bool, Any]:
    """Dump object to SHM if serialized size exceeds threshold.

    Returns (True, meta) when dumped; otherwise (False, original_obj).
    """
    payload = serialize_obj(obj)
    if len(payload) > threshold:
        return True, shm_write_bytes(payload)
    return False, obj


def maybe_load_from_ipc(container: dict[str, Any], obj_key: str, shm_key: str) -> Any:
    """Load object from container that may carry SHM or inline object.

    Deprecated: prefer `maybe_load_from_ipc_with_metrics` to also obtain
    decode-time and size metrics.
    """
    if shm_key in container:
        from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

        return OmniSerializer.deserialize(shm_read_bytes(container[shm_key]))
    return container[obj_key]


def maybe_load_from_ipc_with_metrics(
    container: dict[str, Any], obj_key: str, shm_key: str
) -> tuple[Any, dict[str, float]]:
    """Load object and return (object, metrics) with RX bytes and decode time.

    Metrics keys:
      - rx_transfer_bytes: int
      - rx_decode_time_ms: float
    """
    import time as _time  # local import to avoid overhead at module import

    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    t0 = _time.time()
    if shm_key in container:
        meta = container[shm_key]  # type: ignore[index]
        payload = shm_read_bytes(meta)
        obj = OmniSerializer.deserialize(payload)
        try:
            rx_bytes = int(meta.get("size", len(payload)))  # type: ignore[call-arg]
        except Exception:
            rx_bytes = len(payload)
    else:
        obj = container[obj_key]
        try:
            rx_bytes = len(serialize_obj(obj))
        except Exception:
            rx_bytes = 0
    t1 = _time.time()
    rx_decode_ms = (t1 - t0) * 1000.0
    return obj, {
        "rx_transfer_bytes": int(rx_bytes),
        "rx_decode_time_ms": float(rx_decode_ms),
    }


def encode_for_ipc(obj: Any, threshold: int, obj_key: str, shm_key: str) -> dict[str, Any]:
    """Return a dict payload for IPC: inline (obj_key) or SHM (shm_key).

    When serialized size exceeds threshold, returns {shm_key: {name,size}};
    otherwise returns {obj_key: obj}.
    """
    payload: dict[str, Any] = {}
    use_shm, data = maybe_dump_to_shm(obj, threshold)
    if use_shm:
        payload[shm_key] = data
    else:
        payload[obj_key] = data
    return payload


# Convert OmegaConf/objects to plain dicts
def _to_dict(x: Any) -> dict[str, Any]:
    try:
        if isinstance(x, dict):
            return dict(x)
        return OmegaConf.to_container(x, resolve=True)  # type: ignore[arg-type]
    except Exception:
        try:
            return dict(x)
        except Exception:
            return {}


def acquire_device_locks(
    device_type: str | None,
    engine_args: dict[str, Any],
    stage_id: Any,
    stage_init_timeout: int,
    _os,
    _time,
) -> list[int]:
    lock_files: list[int] = []
    if device_type == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                # Get all parallel sizes from engine_args or parallel_config (defaults to 1)
                if "parallel_config" in engine_args:
                    parallel_config = engine_args["parallel_config"]
                    tensor_parallel_size = parallel_config.get("tensor_parallel_size", 1)
                    pipeline_parallel_size = parallel_config.get("pipeline_parallel_size", 1)
                    data_parallel_size = parallel_config.get("data_parallel_size", 1)
                    prefill_context_parallel_size = 1  # not used for diffusion
                    sequence_parallel_size = parallel_config.get("sequence_parallel_size", 1)
                else:
                    tensor_parallel_size = engine_args.get("tensor_parallel_size", 1)
                    pipeline_parallel_size = engine_args.get("pipeline_parallel_size", 1)
                    data_parallel_size = engine_args.get("data_parallel_size", 1)
                    prefill_context_parallel_size = engine_args.get("prefill_context_parallel_size", 1)
                    sequence_parallel_size = 1  # not use in omni model

                # Calculate total number of devices needed for this stage
                # For a single stage worker:
                # - TP: splits model across GPUs (always needed)
                # - PP: splits layers across pipelinestages, but each stage uses TP devices
                # - DP: replicates model, but each replica uses TP devices
                # - PCP: context parallelism, typically uses TP devices
                # - SP: sequence parallelism, typically uses TP devices
                # The number of devices per stage is determined by TP * PP * DP * PCP * SP size
                # (PP/DP/PCP are higher-level parallelism that don't add devices per stage)
                num_devices_per_stage = (
                    tensor_parallel_size
                    * pipeline_parallel_size
                    * data_parallel_size
                    * prefill_context_parallel_size
                    * sequence_parallel_size
                )

                # Get physical device IDs from CUDA_VISIBLE_DEVICES
                # After set_stage_devices, CUDA_VISIBLE_DEVICES is set to physical device(s)
                cuda_visible_devices = _os.environ.get("CUDA_VISIBLE_DEVICES")
                physical_devices = []

                if cuda_visible_devices:
                    try:
                        physical_devices = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
                    except (ValueError, IndexError):
                        pass

                if not physical_devices:
                    # Fallback: use logical device count if CUDA_VISIBLE_DEVICES not set
                    num_devices = torch.cuda.device_count()
                    physical_devices = list(range(num_devices))

                # Determine which devices will be used (min of devices per stage and available devices)
                num_devices_to_lock = min(num_devices_per_stage, len(physical_devices))
                devices_to_lock = physical_devices[:num_devices_to_lock]

                # Sort devices_to_lock to prevent deadlock (all processes acquire locks in same order)
                devices_to_lock = sorted(devices_to_lock)

                logger.debug(
                    "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d, SP=%d; will lock %d devices: %s",
                    tensor_parallel_size,
                    pipeline_parallel_size,
                    data_parallel_size,
                    prefill_context_parallel_size,
                    sequence_parallel_size,
                    num_devices_to_lock,
                    devices_to_lock,
                )

                # Acquire exclusive locks for all devices using fcntl.flock
                # Locks are automatically released when process dies
                wait_start = _time.time()
                acquired_lock_fds = []  # Store file descriptors to keep locks alive

                for device_id in devices_to_lock:
                    lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
                    lock_acquired = False

                    while not lock_acquired:
                        try:
                            # Open or create the lock file
                            lock_fd = _os.open(lock_file, _os.O_CREAT | _os.O_RDWR, 0o644)

                            # Try to acquire exclusive lock (non-blocking first)
                            try:
                                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                # Successfully acquired lock - write PID
                                _os.ftruncate(lock_fd, 0)  # Clear file
                                _os.write(lock_fd, f"{_os.getpid()}\n".encode())
                                _os.fsync(lock_fd)  # Ensure written to disk
                                lock_acquired = True
                                acquired_lock_fds.append(lock_fd)
                                logger.debug("Acquired exclusive lock for device %s", device_id)
                            except BlockingIOError:
                                # Lock is held by another process
                                _os.close(lock_fd)

                                # Check if we've been waiting too long
                                if _time.time() - wait_start > stage_init_timeout:
                                    logger.warning(
                                        "Timeout waiting for device %s initialization lock, "
                                        "proceeding anyway with timeout %s",
                                        device_id,
                                        stage_init_timeout,
                                    )
                                    break

                                # Wait a bit before retrying
                                _time.sleep(0.1)
                        except OSError as e:
                            # Other error - log and continue without lock
                            logger.debug(
                                "Failed to acquire lock for device %s: %s, continuing anyway",
                                device_id,
                                e,
                            )
                            try:
                                _os.close(lock_fd)
                            except (OSError, NameError):
                                pass
                            break

                lock_files = acquired_lock_fds
        except Exception as e:
            logger.debug("[Stage-%s] Failed to set up sequential initialization lock: %s", stage_id, e)
    return lock_files


def initialize_stage_engine(
    stage_type: str,
    model: str,
    engine_args: dict[str, Any],
    lock_files: list[int],
    _os,
):
    try:
        if stage_type == "diffusion":
            engine_args.pop("model_stage")
            stage_engine = OmniDiffusion(**engine_args)
        else:
            # Default to LLM engine
            stage_engine = OmniLLM(model=model, **engine_args)
    finally:
        # Release all locks by closing file descriptors
        # Locks are automatically released when file descriptors are closed
        # or when process dies
        for lock_fd in lock_files:
            try:
                _os.close(lock_fd)
                logger.debug("Released initialization lock (fd=%s)", lock_fd)
            except (OSError, ValueError):
                pass
    logger.debug("Engine initialized")
    return stage_engine


def initialize_connectors(stage_id: Any, connectors_config: dict[str, Any]) -> dict[str, Any] | None:
    # Initialize OmniConnectors if configured
    connectors: dict[str, Any] = {}
    if connectors_config:
        built_connectors = build_stage_connectors(
            stage_id=stage_id,
            connectors_config=connectors_config,
        )
        if built_connectors is None:
            return None
        connectors = built_connectors
    return connectors


def collect_batch_tasks(
    in_q: mp.Queue,
    first_task: dict[str, Any],
    max_batch_size: int,
    batch_timeout: int,
    _time,
) -> list[dict[str, Any]]:
    batch_tasks: list[dict[str, Any]] = [first_task]
    start_time = _time.time()
    if max_batch_size > 1:
        while len(batch_tasks) < max_batch_size:
            if not in_q.empty():
                extra = in_q.get_nowait()
                if extra == SHUTDOWN_TASK:
                    in_q.put(SHUTDOWN_TASK)
                    break
                batch_tasks.append(extra)
                end_time = _time.time()
                duration = end_time - start_time
                if duration > batch_timeout:
                    break
                else:
                    continue
            else:
                end_time = _time.time()
                duration = end_time - start_time
                _time.sleep(0.05)
                if duration > batch_timeout:
                    break
                else:
                    continue
    return batch_tasks


def prepare_batch_payloads(
    batch_tasks: list[dict[str, Any]],
    connectors: dict[str, Any],
    stage_id: Any,
    _recv_dequeue_ts: float,
):
    batch_request_ids: list[Any] = []
    batch_engine_inputs: list[Any] = []
    _rx_bytes_by_rid: dict[Any, int] = {}
    _rx_decode_ms_by_rid: dict[Any, float] = {}
    _in_flight_ms_by_rid: dict[Any, float] = {}
    for t in batch_tasks:
        rid = t["request_id"]
        try:
            sent_ts = float(t.get("sent_ts", None)) if isinstance(t, dict) else None
            if sent_ts is not None:
                _in_flight_ms_by_rid[rid] = (_recv_dequeue_ts - sent_ts) * 1000.0
            else:
                _in_flight_ms_by_rid[rid] = 0.0
        except Exception:
            _in_flight_ms_by_rid[rid] = 0.0

        # Resolve input data strictly via connectors if payload
        # is larger than shm_threshold_bytes or using other connectors
        ein, _rx_metrics = try_recv_via_connector(
            task=t,
            connectors=connectors,
            stage_id=stage_id,
        )

        if ein is None or _rx_metrics is None:
            raise RuntimeError(
                f"[Stage-{stage_id}] Missing connector payload for request {rid}. "
                "Ensure connectors are configured for all incoming edges."
            )

        _rx_decode_ms_by_rid[rid] = float(_rx_metrics.get("rx_decode_time_ms", 0.0))
        _rx_bytes_by_rid[rid] = int(_rx_metrics.get("rx_transfer_bytes", 0))

        batch_request_ids.append(rid)
        if isinstance(ein, list):
            batch_engine_inputs.extend(ein)
        elif isinstance(ein, dict):
            batch_engine_inputs.append(ein)
        elif isinstance(ein, str):
            # For diffusion stage-0, ein might be a string prompt directly
            batch_engine_inputs.append(ein)
        else:
            # For other types (e.g., OmniTokensPrompt, TextPrompt), append as-is
            batch_engine_inputs.append(ein)
    return (
        batch_request_ids,
        batch_engine_inputs,
        _rx_bytes_by_rid,
        _rx_decode_ms_by_rid,
        _in_flight_ms_by_rid,
    )


def generate_batch_outputs(
    stage_type: str,
    stage_engine: Any,
    batch_engine_inputs: list[Any],
    sampling_params: Any,
    batch_request_ids: list[Any],
    batch_task_count: int,
    _time,
):
    gen_outputs: list[Any] = []
    _gen_t0 = _time.time()
    if stage_type == "diffusion":
        # For diffusion, batch_engine_inputs should be prompts (strings)
        # Convert to list of strings if needed
        prompts = []
        for ein in batch_engine_inputs:
            if isinstance(ein, str):
                prompts.append(ein)
            elif isinstance(ein, dict) and "prompt" in ein:
                prompts.append(ein["prompt"])
            elif hasattr(ein, "prompt"):
                prompts.append(ein.prompt)
            else:
                prompts.append(str(ein))
        # Prepare diffusion kwargs from sampling parameters
        diffusion_kwargs = prepare_sampling_params(sampling_params, "diffusion")
        # Diffusion generate returns results directly, not an iterator
        diffusion_results = stage_engine.generate(prompts, **diffusion_kwargs)
        # Convert to list format compatible with LLM outputs
        # Ensure each result has a request_id for proper mapping
        if isinstance(diffusion_results, list):
            gen_outputs = diffusion_results
            # Assign request_ids if not present
            for idx, result in enumerate(gen_outputs):
                if not hasattr(result, "request_id") or result.request_id is None:
                    if idx < len(batch_request_ids):
                        if hasattr(result, "request_id"):
                            result.request_id = batch_request_ids[idx]
                        else:
                            # Create a wrapper object if result doesn't support request_id
                            from types import SimpleNamespace

                            wrapped = SimpleNamespace()
                            wrapped.request_id = batch_request_ids[idx]
                            wrapped.output = result
                            gen_outputs[idx] = wrapped
        else:
            gen_outputs = [diffusion_results]
            # Assign request_id to single result
            if len(batch_request_ids) > 0:
                if hasattr(gen_outputs[0], "request_id"):
                    gen_outputs[0].request_id = batch_request_ids[0]
                else:
                    from types import SimpleNamespace

                    wrapped = SimpleNamespace()
                    wrapped.request_id = batch_request_ids[0]
                    wrapped.output = gen_outputs[0]
                    gen_outputs[0] = wrapped
    else:
        # LLM engine: use vLLM native SamplingParams
        llm_sampling_params = prepare_sampling_params(sampling_params, "llm")
        for ro in stage_engine.generate(batch_engine_inputs, llm_sampling_params, use_tqdm=False):
            gen_outputs.append(ro)
    _gen_t1 = _time.time()
    _gen_ms = (_gen_t1 - _gen_t0) * 1000.0
    logger.debug(f"Generate done: batch={batch_task_count}, req_ids={batch_request_ids}, gen_ms={_gen_ms:.1f}")
    return gen_outputs, _gen_ms


def group_outputs_by_request(
    batch_request_ids: list[Any],
    gen_outputs: list[Any],
) -> dict[Any, list[Any]]:
    # Group outputs per request id with fallback
    req_to_outputs: dict[Any, list[Any]] = {rid: [] for rid in batch_request_ids}
    unmapped: list[Any] = []
    for ro in gen_outputs:
        rid = getattr(ro, "request_id", None)
        if rid in req_to_outputs:
            req_to_outputs[rid].append(ro)
        else:
            unmapped.append(ro)
    if unmapped:
        idx = 0
        for ro in unmapped:
            target_rid = batch_request_ids[idx % len(batch_request_ids)]
            ro.request_id = target_rid
            req_to_outputs[target_rid].append(ro)
            idx += 1
    return req_to_outputs


def emit_batch_results(
    batch_request_ids: list[Any],
    req_to_outputs: dict[Any, list[Any]],
    _rx_decode_ms_by_rid: dict[Any, float],
    _rx_bytes_by_rid: dict[Any, int],
    _in_flight_ms_by_rid: dict[Any, float],
    _batch_seq: int,
    _gen_ms: float,
    shm_threshold_bytes: int,
    out_q: mp.Queue,
    stage_id: Any,
    _agg_total_tokens: int,
    _agg_total_gen_time_ms: float,
) -> int:
    for i, rid in enumerate(batch_request_ids):
        r_outputs = req_to_outputs.get(rid, [])
        _metrics = make_request_stats(
            r_outputs,
            _gen_ms,
            int(_batch_seq),
            int(len(batch_request_ids)),
            float(_rx_decode_ms_by_rid.get(rid, 0.0)),
            int(_rx_bytes_by_rid.get(rid, 0)),
            float(_in_flight_ms_by_rid.get(rid, 0.0)),
        )
        _agg_total_tokens += _metrics.num_tokens_out
        if i == len(batch_request_ids) - 1:
            _metrics.stage_stats = make_stage_stats(_agg_total_tokens, _agg_total_gen_time_ms)
        else:
            _metrics.stage_stats = None
        enqueue_one_result(
            out_q=out_q,
            stage_id=stage_id,
            request_id=rid,
            engine_outputs=r_outputs,
            metrics=_metrics,
            shm_threshold_bytes=shm_threshold_bytes,
            log_exception=False,
        )
        logger.debug(
            "Enqueued result for request %s to downstream",
            rid,
        )
    return _agg_total_tokens


def enqueue_one_result(
    *,
    out_q: mp.Queue,
    stage_id: Any,
    request_id: Any,
    engine_outputs: list[Any],
    metrics: Any,
    shm_threshold_bytes: int,
    log_exception: bool,
) -> None:
    """Enqueue a single request result to downstream."""
    try:
        use_shm, payload = maybe_dump_to_shm(engine_outputs, shm_threshold_bytes)
        if use_shm:
            out_q.put(
                {
                    "request_id": request_id,
                    "stage_id": stage_id,
                    "engine_outputs_shm": payload,
                    "metrics": metrics,
                }
            )
        else:
            out_q.put(
                {
                    "request_id": request_id,
                    "stage_id": stage_id,
                    "engine_outputs": payload,
                    "metrics": metrics,
                }
            )
        logger.debug(
            f"Enqueued req={request_id}, use_shm={use_shm}, tokens_out={getattr(metrics, 'num_tokens_out', None)}"
        )
        return
    except Exception as e:
        if log_exception:
            logger.exception(
                "Failed to enqueue result for request %s: %s",
                request_id,
                e,
            )
        out_q.put(
            {
                "request_id": request_id,
                "stage_id": stage_id,
                "engine_outputs": engine_outputs,
                "metrics": metrics,
            }
        )
        logger.debug(
            f"Enqueued req={request_id}, use_shm={False}, tokens_out={getattr(metrics, 'num_tokens_out', None)}"
        )
        return


def handle_batch_exception(out_q: mp.Queue, batch_request_ids: list[Any], stage_id: Any, exc: Exception) -> None:
    logger.exception("Failed on batch %s: %s", batch_request_ids, exc)
    _tb = traceback.format_exc()
    for rid in batch_request_ids:
        out_q.put(
            {
                "request_id": rid,
                "stage_id": stage_id,
                "error": str(exc),
                "error_tb": _tb,
            }
        )


def make_request_stats(
    req_output: list[Any],
    stage_gen_time_ms: float,
    batch_id: int,
    batch_size: int,
    rx_decode_time_ms: float,
    rx_transfer_bytes: int,
    rx_in_flight_time_ms: float,
):
    from vllm_omni.entrypoints.log_utils import (
        StageRequestMetrics,
    )

    num_tokens_out = count_tokens_from_outputs(req_output)
    return StageRequestMetrics(
        num_tokens_out=num_tokens_out,
        stage_gen_time_ms=stage_gen_time_ms,
        batch_id=batch_id,
        batch_size=batch_size,
        rx_decode_time_ms=rx_decode_time_ms,
        rx_transfer_bytes=rx_transfer_bytes,
        rx_in_flight_time_ms=rx_in_flight_time_ms,
        stage_stats=None,
    )


def make_stage_stats(_agg_total_tokens: int, _agg_total_gen_time_ms: float):
    from vllm_omni.entrypoints.log_utils import StageStats

    return StageStats(total_token=_agg_total_tokens, total_gen_time=_agg_total_gen_time_ms)
