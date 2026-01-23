# Production Metrics

## Usage

Users can utilize these metrics in production environments to monitor the health and performance of the vLLM-omni system. Key scenarios include:
- **Performance Monitoring**: Track throughput (e.g., `e2e_avg_tokens_per_s`), latency (e.g., `e2e_total_ms`), and resource utilization to verify that the system meets expected performance standards.
- **Debugging and Troubleshooting**: Use detailed per-request metrics to diagnose issues with specific requests, such as high transfer times or unexpected token counts.
-  **Enable Logging**: Start vLLM-omni with the `--log-stats` flag. This exposes metrics through structured logs.

## Overall Summary

| Field                     | Meaning                                                                                       |
|---------------------------|----------------------------------------------------------------------------------------------|
| `e2e_requests`            | Number of completed requests.                                                                |
| `e2e_wall_time_ms`        | Wall-clock time span from run start to last completion, in ms.                               |
| `e2e_total_tokens`        | Total tokens counted across all completed requests (stage0 input + all stage outputs).       |
| `e2e_avg_time_per_request_ms` | Average wall time per request: `e2e_wall_time_ms / e2e_requests`.                        |
| `e2e_avg_tokens_per_s`    | Average token throughput over wall time: `e2e_total_tokens * 1000 / e2e_wall_time_ms`.      |
| `stage_wall_time_ms`      | Wall-clock time span for each stage, in ms (list format).                                   |

---

## E2E Table (per request)

| Field                     | Meaning                                                               |
|---------------------------|-----------------------------------------------------------------------|
| `e2e_total_ms`            | End-to-end latency in ms.                                             |
| `e2e_total_tokens`        | Total tokens for the request (stage0 input + all stage outputs).      |
| `transfers_total_time_ms` | Sum of transfer edge `total_time_ms` for this request.                |
| `transfers_total_bytes`   | Sum of transfer bytes for this request.                               |


---

## Stage Table (per stage event / request)

| Field               | Meaning                                                                             |
|---------------------|-------------------------------------------------------------------------------------|
| `batch_id`          | Batch index.                                                                        |
| `batch_size`        | Batch size.                                                                         |
| `num_tokens_in`     | Input tokens to the stage.                                                          |
| `num_tokens_out`    | Output tokens from the stage.                                                       |
| `preprocess_time_ms` | Preprocessing time in ms.                                                           |
| `stage_gen_time_ms` | Stage compute time in ms (includes preprocessing time if recorded).                 |

---

## Transfer Table (per edge / request)

| Field                | Meaning                                                                   |
|----------------------|---------------------------------------------------------------------------|
| `size_bytes`         | Total bytes transferred.                                                  |
| `tx_time_ms`         | Sender transfer time in ms.                                               |
| `rx_decode_time_ms`  | Receiver decode time in ms.                                               |
| `in_flight_time_ms`  | In-flight time in ms.                                                     |

## Expectation of the numbers:
e2e_total_tokens = Stage0 's num_tokens_in + other stage's num_tokens_out
transfers_total_time_ms = sum(tx_time_ms + rx_decode_time_ms + in_flight_time_ms) in every edge
