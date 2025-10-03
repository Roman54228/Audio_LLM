# 📈 Performance Testing

**Note**: The following benchmarks focus on the acoustic model, which is the most computationally intensive component of the pipeline.

To achieve state of the art performance of streaming acoustic model and upstream services we recommend several options to use: 

- [`Microsoft ONNX Runtime`](https://github.com/microsoft/onnxruntime/) + [`NVIDIA Triton Inference Server`](https://github.com/triton-inference-server/server) = 🏃💨
An easy-to-use baseline, although not the fastest option.
- [`NVIDIA TensorRT`](https://github.com/NVIDIA/TensorRT/)  + [`NVIDIA Triton inference server`](https://github.com/triton-inference-server/server) = ⚡️🏃💨💨
A much faster (x2-x3 times) alternative, but requires converting the model to the TensorRT format.

## Launching the model server

Before running performance tests, the model must be exported to `ONNX` or `TensorRT` and launched in the Triton Inference Server. See the [Triton Inference Server](docs/triton_inference_server.md) section for detailed instructions.

## Running performance tests

### trtexec

To run performance tests of the acoustic model converted to `TensorRT` you can use a cli utility [**`trtexec`**](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html).

**Key Flags:**

- `--warmUp`: How long to run for warm-up before measuring in ms.
- `--duration`: How many seconds the main measurement should run for in seconds.
- `--iterations`: Alternatively, run for a fixed number of iterations instead of a duration.
- `--avgRuns`: Run the benchmark N times and average the results for more stability.
- `--verbose` or `-v`: Shows more detailed logs about the engine loading and execution.

**Example:**

```bash
trtexec --loadEngine=<engine_file.plan> --warmUp=2000 --duration=10 --avgRuns=5
```

### perf_analyzer

You can use `perf_analyzer` from the Triton SDK container to run performance tests for both `ONNX` and `TensorRT` engines. To do this, first launch the SDK container:

```bash
docker run --rm -it --net host \
  nvcr.io/nvidia/tritonserver:24.02-py3-sdk
```

Note: The SDK version must be `24.02` because `25.06` has [OOM issue](https://github.com/triton-inference-server/perf_analyzer/issues/84)

Then, execute the `perf_analyzer` with streaming ASR-specific options:

```bash
perf_analyzer -u localhost:8001 \
  -i grpc \
  -a \
  -m streaming_acoustic \
  --streaming \
  --sequence-length=50 \
  --measurement-mode=count_windows \
  --measurement-request-count=5000 \
  --request-rate-range=2048:4096:256 \
  --stability-percentage=100 \
  --latency-threshold=100
```

**Key flags:**

- `--sequence-length`: Length of input sequences. sequence-length of 1 corresponds to 300 ms of audio
- `--request-rate-range`: Range of request rates to test
- `--latency-threshold`: Sets the latency threshold in milliseconds

## Results

Example output from `perf_analyzer` for `RTX 3090` and `TensorRT` configuration, batch size `16`:
```bash
Inferences/Second vs. Client Average Batch Latency
Request Rate: 2048.00, throughput: 2045.52 infer/sec, latency 13250 usec
Request Rate: 2304.00, throughput: 2300.63 infer/sec, latency 12819 usec
Request Rate: 2560.00, throughput: 2555.53 infer/sec, latency 13700 usec
Request Rate: 2816.00, throughput: 2804.02 infer/sec, latency 20469 usec
Request Rate: 3072.00, throughput: 2902.08 infer/sec, latency 41838 usec
Request Rate: 3328.00, throughput: 2887.90 infer/sec, latency 38171 usec
Request Rate: 3584.00, throughput: 2811.21 infer/sec, latency 32192 usec
Request Rate: 3840.00, throughput: 2788.82 infer/sec, latency 25986 usec
Request Rate: 4096.00, throughput: 2745.62 infer/sec, latency 27433 usec
```

You can also compute throughput using the formula: `SPS = inferences/sec * chunk size (sec)`. For the example above, this gives a throughput of `3000 * 0.3 = 900 SPS`, while the latency per chunk remains below 100 ms.
