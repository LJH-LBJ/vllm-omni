import warnings
from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
from vllm.benchmarks.serve import MILLISECONDS_TO_SECONDS_CONVERSION, TERM_PLOTLIB_AVAILABLE, BenchmarkMetrics, TaskType


@dataclass
class MultiModalsBenchmarkMetrics(BenchmarkMetrics):
    mean_audio_ttfp_ms: float = 0.0
    median_audio_ttfp_ms: float = 0.0
    std_audio_ttfp_ms: float = 0.0
    percentiles_audio_ttfp_ms: list[tuple[float, float]] = None
    total_audio_duration_s: float = 0.0
    total_audio_frames: int = 0
    audio_throughput: float = 0.0
    mean_audio_rtf: float = 0.0
    median_audio_rtf: float = 0.0
    std_audio_rtf: float = 0.0
    percentiles_audio_rtf: list[tuple[float, float]] = None
    mean_audio_duration_s: float = 0.0
    median_audio_duration_s: float = 0.0
    std_audio_duration_s: float = 0.0
    percentiles_audio_duration_s: list[tuple[float, float]] = None
    total_talker_input_tokens: int = 0
    total_talker_output_tokens: int = 0
    talker_output_throughput: float = 0.0
    mean_talker_tpot_ms: float = 0.0
    median_talker_tpot_ms: float = 0.0
    std_talker_tpot_ms: float = 0.0
    percentiles_talker_tpot_ms: list[tuple[float, float]] = None
    mean_talker_itl_ms: float = 0.0
    median_talker_itl_ms: float = 0.0
    std_talker_itl_ms: float = 0.0
    percentiles_talker_itl_ms: list[tuple[float, float]] = None
    total_code2wav_input_tokens: int = 0
    mean_code2wav_tpot_ms: float = 0.0
    median_code2wav_tpot_ms: float = 0.0
    std_code2wav_tpot_ms: float = 0.0
    percentiles_code2wav_tpot_ms: list[tuple[float, float]] = None
    mean_code2wav_itl_ms: float = 0.0
    median_code2wav_itl_ms: float = 0.0
    std_code2wav_itl_ms: float = 0.0
    percentiles_code2wav_itl_ms: list[tuple[float, float]] = None


def print_metrics(
    task_type,
    selected_percentile_metrics,
    max_concurrency,
    request_rate,
    benchmark_duration,
    goodput_config_dict,
    metrics: MultiModalsBenchmarkMetrics,
):
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failed))
    if max_concurrency is not None:
        print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
    if request_rate != float("inf"):
        print("{:<40} {:<10.2f}".format("Request rate configured (RPS):", request_rate))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
    if isinstance(metrics, MultiModalsBenchmarkMetrics):
        print("{:<40} {:<10.2f}".format("Peak concurrent requests:", metrics.max_concurrent_requests))
    if task_type != TaskType.GENERATION or "e2el" in selected_percentile_metrics:
        process_one_metric("e2el", metrics)
    print_text_metrics(task_type, selected_percentile_metrics, metrics)
    if task_type == TaskType.GENERATION:
        print_audio_metrics(selected_percentile_metrics, metrics)
        print_talker_metrics(metrics)
        print_code2wav_metrics(metrics)
    print("=" * 50)


def print_text_metrics(task_type, selected_percentile_metrics, metrics: MultiModalsBenchmarkMetrics):
    print("{s:{c}^{n}}".format(s=" Text Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    if isinstance(metrics, MultiModalsBenchmarkMetrics):
        print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
        print("{:<40} {:<10.2f}".format("Peak output token throughput (tok/s):", metrics.max_output_tokens_per_s))
        print("{:<40} {:<10.2f}".format("Peak concurrent requests:", metrics.max_concurrent_requests))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))

    if task_type == TaskType.GENERATION:
        for metric in selected_percentile_metrics:
            if metric == "e2el":
                continue
            if not metric.startswith("audio"):
                process_one_metric(metric, metrics)


def print_audio_metrics(selected_percentile_metrics, metrics: MultiModalsBenchmarkMetrics):
    print("{s:{c}^{n}}".format(s=" Audio Result ", n=50, c="="))
    print("{:<40} {:<10.2f}".format("Total audio duration generated(s):", metrics.total_audio_duration_s))
    print("{:<40} {:<10}".format("Total audio frames generated:", metrics.total_audio_frames))
    print("{:<40} {:<10.2f}".format("Audio throughput(audio duration/s):", metrics.audio_throughput))
    for metric in selected_percentile_metrics:
        if metric.startswith("audio"):
            process_one_metric(metric, metrics)


def process_one_metric(
    metric_attribute_name: str,
    metrics: MultiModalsBenchmarkMetrics,
):
    metric_header_map = {
        "ttft": "Time to First Token",
        "tpot": "Time per Output Token (excl. 1st token)",
        "itl": "Inter-token Latency",
        "e2el": "End-to-end Latency",
        "audio_ttfp": "Time to First Packet",
        "audio_rtf": "Real Time Factor",
        "audio_duration": "Audio Duration",
        "talker_tpot": "Talker Time per Output Token (excl. 1st token)",
        "talker_itl": "Talker Inter-token Latency",
        "code2wav_tpot": "Code2Wav Time per Output Token (excl. 1st token)",
        "code2wav_itl": "Code2Wav Inter-token Latency",
    }

    header = metric_header_map.get(metric_attribute_name, metric_attribute_name)
    print("{s:{c}^{n}}".format(s=header, n=50, c="-"))

    is_audio_rtf = metric_attribute_name == "audio_rtf"
    is_audio_duration = metric_attribute_name == "audio_duration"

    suffix = "_ms"
    unit_suffix = " (ms)"
    if is_audio_duration:
        suffix = "_s"
        unit_suffix = " (s)"
    elif is_audio_rtf:
        suffix = ""
        unit_suffix = ""
    mean_attr_name = f"mean_{metric_attribute_name}{suffix}"
    mean_value = getattr(metrics, mean_attr_name, 0.0)
    print(f"{f'Mean {metric_attribute_name.upper()}{unit_suffix}:':<40} {mean_value:<10.2f}")

    median_attr_name = f"median_{metric_attribute_name}{suffix}"
    median_value = getattr(metrics, median_attr_name, 0.0)
    print(f"{f'Median {metric_attribute_name.upper()}{unit_suffix}:':<40} {median_value:<10.2f}")

    percentiles_attr_name = f"percentiles_{metric_attribute_name}{suffix}"
    percentiles = getattr(metrics, percentiles_attr_name, [])

    for percentile, value in percentiles:
        p_str = str(int(percentile)) if percentile.is_integer() else str(percentile)
        label = f"P{p_str} {metric_attribute_name.upper()}{unit_suffix}:"
        print(f"{label:<40} {value:<10.2f}")


def print_talker_metrics(metrics: MultiModalsBenchmarkMetrics):
    print("{s:{c}^{n}}".format(s=" Talker Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Total talker input tokens:", metrics.total_talker_input_tokens))
    print("{:<40} {:<10}".format("Total talker output tokens:", metrics.total_talker_output_tokens))
    print("{:<40} {:<10.2f}".format("Talker output throughput (tok/s):", metrics.talker_output_throughput))
    process_one_metric("talker_tpot", metrics)
    process_one_metric("talker_itl", metrics)


def print_code2wav_metrics(metrics: MultiModalsBenchmarkMetrics):
    print("{s:{c}^{n}}".format(s=" Code2Wav Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Total code2wav input tokens:", metrics.total_code2wav_input_tokens))
    process_one_metric("code2wav_tpot", metrics)
    process_one_metric("code2wav_itl", metrics)


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
    task_type,
    selected_percentile_metrics,
    max_concurrency,
    request_rate,
    benchmark_duration,
) -> tuple[BenchmarkMetrics, list[int]]:
    """Calculate the metrics for the benchmark.

    Args:
        input_requests: The input requests.
        outputs: The outputs of the requests.
        dur_s: The duration of the benchmark.
        tokenizer: The tokenizer to use.
        selected_percentiles: The percentiles to select.
        goodput_config_dict: The goodput configuration.

    Returns:
        A tuple of the benchmark metrics and the actual output lengths.
    """
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    audio_ttfps: list[float] = []
    audio_rtfs: list[float] = []
    audio_duration: list[float] = []
    audio_frames: list[int] = []
    talker_tpots: list[float] = []
    talker_itls: list[float] = []
    code2wav_tpots: list[float] = []
    code2wav_itls: list[float] = []
    talker_input_tokens = 0
    talker_output_tokens = 0
    code2wav_input_tokens = 0
    code2wav_output_tokens = 0
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].text_latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            audio_ttfps.append(getattr(outputs[i], "audio_ttfp", 0.0))
            audio_rtfs.append(getattr(outputs[i], "audio_rtf", 0.0))
            audio_duration.append(getattr(outputs[i], "audio_duration", 0.0))
            audio_frames.append(getattr(outputs[i], "audio_frames", 0.0))
            talker_input_tokens += int(getattr(outputs[i], "talker_input_tokens", 0) or 0)
            talker_output_tokens += int(getattr(outputs[i], "talker_output_tokens", 0) or 0)
            code2wav_input_tokens += int(getattr(outputs[i], "code2wav_input_tokens", 0) or 0)
            code2wav_output_tokens += int(getattr(outputs[i], "code2wav_output_tokens", 0) or 0)
            talker_stage_ms = float(getattr(outputs[i], "talker_stage_gen_time_ms", 0.0) or 0.0)
            code2wav_stage_ms = float(getattr(outputs[i], "code2wav_stage_gen_time_ms", 0.0) or 0.0)

            def _stage_latency_s(output, latency_attr: str, stage_ms: float) -> float:
                latency_s = float(getattr(output, latency_attr, 0.0) or 0.0)
                if latency_s <= 0.0 and stage_ms > 0.0:
                    latency_s = stage_ms / 1000.0
                return latency_s

            def _stage_ttft(output, ttft_attr: str, latency_s: float) -> float:
                ttft = float(getattr(output, ttft_attr, 0.0) or 0.0)
                if ttft <= 0.0 and latency_s > 0.0:
                    ttft = latency_s
                return ttft

            talker_latency_s = _stage_latency_s(outputs[i], "talker_latency", talker_stage_ms)
            code2wav_latency_s = _stage_latency_s(outputs[i], "code2wav_latency", code2wav_stage_ms)

            talker_output_len = int(getattr(outputs[i], "talker_output_tokens", 0) or 0)
            code2wav_output_len = int(getattr(outputs[i], "code2wav_output_tokens", 0) or 0)

            talker_ttft = _stage_ttft(outputs[i], "talker_ttft", talker_latency_s)
            code2wav_ttft = _stage_ttft(outputs[i], "code2wav_ttft", code2wav_latency_s)

            talker_tpot = 0.0
            if talker_output_len > 1 and talker_latency_s > 0.0:
                talker_latency_minus_ttft = max(talker_latency_s - talker_ttft, 0.0)
                talker_tpot = talker_latency_minus_ttft / (talker_output_len - 1)

            code2wav_tpot = 0.0
            if code2wav_output_len > 1 and code2wav_latency_s > 0.0:
                code2wav_latency_minus_ttft = max(code2wav_latency_s - code2wav_ttft, 0.0)
                code2wav_tpot = code2wav_latency_minus_ttft / (code2wav_output_len - 1)
            if talker_output_len > 1:
                talker_tpots.append(talker_tpot)
                talker_itl_list = getattr(outputs[i], "talker_itl", None)
                if talker_itl_list:
                    talker_itls += list(talker_itl_list)
                else:
                    talker_itls.append(talker_tpot)
            if code2wav_output_len > 1:
                code2wav_tpots.append(code2wav_tpot)
                code2wav_itl_list = getattr(outputs[i], "code2wav_itl", None)
                if code2wav_itl_list:
                    code2wav_itls += list(code2wav_itl_list)
                else:
                    code2wav_itls.append(code2wav_tpot)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "audio_ttft" in goodput_config_dict:
            valid_metrics.append(audio_ttfps)
            slo_values.append(goodput_config_dict["audio_ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.formatwarning = (
            lambda msg, category, filename, lineno, line=None: f"{filename}:{lineno}: {category.__name__}: {msg}\n"
        )
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration on the benchmark arguments.",
            stacklevel=2,
        )

    # Calculate max output tokens per second metric
    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0

    # Find the time range across all successful requests
    successful_outputs = [output for output in outputs if output.success]
    failed_outputs = [output for output in outputs if not output.success]
    if successful_outputs:
        min_start_time = min(output.start_time for output in successful_outputs)
        max_end_time = max(output.start_time + output.latency for output in successful_outputs)

        # Create second buckets (ceiling to ensure we capture all time)
        duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
        tokens_per_second = np.zeros(duration_seconds)
        concurrent_requests_per_second = np.zeros(duration_seconds)

        for i, output in enumerate(successful_outputs):
            # Calculate token generation timestamp using
            # start_time, ttft, and itl
            token_times = [output.start_time + output.ttft]
            current_time = token_times[0]
            for itl_value in output.itl:
                current_time += itl_value
                token_times.append(current_time)

            # Add tokens to second buckets
            for token_time in token_times:
                second_bucket = int(token_time - min_start_time)
                if 0 <= second_bucket < duration_seconds:
                    tokens_per_second[second_bucket] += 1

            # Track concurrent requests for each second this request was active
            request_start_second = int(output.start_time - min_start_time)
            request_end_second = int((output.start_time + output.latency) - min_start_time)

            for second in range(request_start_second, request_end_second + 1):
                concurrent_requests_per_second[second] += 1

        # Find the maximum tokens per second and corresponding
        # concurrent requests
        if len(tokens_per_second) > 0:
            max_output_tokens_per_s = float(np.max(tokens_per_second))
            max_concurrent_requests = int(np.max(concurrent_requests_per_second))

        if TERM_PLOTLIB_AVAILABLE:
            import termplotlib as tpl

            fig = tpl.figure()
            fig.plot(
                np.arange(len(tokens_per_second)),
                tokens_per_second,
                title="Output tokens per second",
            )
            fig.plot(
                np.arange(len(concurrent_requests_per_second)),
                concurrent_requests_per_second,
                title="Concurrent requests per second",
            )
            fig.show()
        else:
            print("tip: install termplotlib and gnuplot to plot the metrics")

    metrics = MultiModalsBenchmarkMetrics(
        completed=completed,
        failed=len(failed_outputs),
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,  # ttfts is empty if streaming is not supported by the endpoint
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles],
        mean_audio_ttfp_ms=np.mean(audio_ttfps or 0) * 1000,
        std_audio_ttfp_ms=np.std(audio_ttfps or 0) * 1000,
        median_audio_ttfp_ms=np.median(audio_ttfps or 0) * 1000,
        percentiles_audio_ttfp_ms=[(p, np.percentile(audio_ttfps or 0, p) * 1000) for p in selected_percentiles],
        mean_audio_duration_s=np.mean(audio_duration or 0),
        std_audio_duration_s=np.std(audio_duration or 0),
        median_audio_duration_s=np.median(audio_duration or 0),
        percentiles_audio_duration_s=[(p, np.percentile(audio_duration or 0, p)) for p in selected_percentiles],
        total_audio_duration_s=sum(audio_duration),
        total_audio_frames=sum(audio_frames),
        audio_throughput=sum(audio_duration) / dur_s,
        mean_audio_rtf=np.mean(audio_rtfs or 0),
        std_audio_rtf=np.std(audio_rtfs or 0),
        median_audio_rtf=np.median(audio_rtfs or 0),
        percentiles_audio_rtf=[(p, np.percentile(audio_rtfs or 0, p)) for p in selected_percentiles],
        total_talker_input_tokens=talker_input_tokens,
        total_talker_output_tokens=talker_output_tokens,
        talker_output_throughput=talker_output_tokens / dur_s,
        mean_talker_tpot_ms=np.mean(talker_tpots or 0) * 1000,
        std_talker_tpot_ms=np.std(talker_tpots or 0) * 1000,
        median_talker_tpot_ms=np.median(talker_tpots or 0) * 1000,
        percentiles_talker_tpot_ms=[(p, np.percentile(talker_tpots or 0, p) * 1000) for p in selected_percentiles],
        mean_talker_itl_ms=np.mean(talker_itls or 0) * 1000,
        std_talker_itl_ms=np.std(talker_itls or 0) * 1000,
        median_talker_itl_ms=np.median(talker_itls or 0) * 1000,
        percentiles_talker_itl_ms=[(p, np.percentile(talker_itls or 0, p) * 1000) for p in selected_percentiles],
        total_code2wav_input_tokens=code2wav_input_tokens,
        mean_code2wav_tpot_ms=np.mean(code2wav_tpots or 0) * 1000,
        std_code2wav_tpot_ms=np.std(code2wav_tpots or 0) * 1000,
        median_code2wav_tpot_ms=np.median(code2wav_tpots or 0) * 1000,
        percentiles_code2wav_tpot_ms=[(p, np.percentile(code2wav_tpots or 0, p) * 1000) for p in selected_percentiles],
        mean_code2wav_itl_ms=np.mean(code2wav_itls or 0) * 1000,
        std_code2wav_itl_ms=np.std(code2wav_itls or 0) * 1000,
        median_code2wav_itl_ms=np.median(code2wav_itls or 0) * 1000,
        percentiles_code2wav_itl_ms=[(p, np.percentile(code2wav_itls or 0, p) * 1000) for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles],
        max_output_tokens_per_s=max_output_tokens_per_s,
        max_concurrent_requests=max_concurrent_requests,
    )
    print_metrics(
        task_type,
        selected_percentile_metrics,
        max_concurrency,
        request_rate,
        benchmark_duration,
        goodput_config_dict,
        metrics,
    )
    return metrics, actual_output_lens
