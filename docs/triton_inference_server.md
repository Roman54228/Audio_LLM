# ⚡️ How to export to TensorRT and run model in Triton Inference Server

## How to export model to TensorRT

The TensorRT engine is compiled and optimized for the specific hardware it will run on (e.g., GPU model, CUDA version). Because of this hardware-specific nature, we cannot provide a universal, pre-built engine file. To get the best performance, you must generate the engine directly on the hardware that will be used for inference.

- First, download the onnx model:

```bash
python -m tone download $(pwd)/models/model.onnx --only-acoustic
```

- Convert the downloaded ONNX model to TensorRT format. Choose $BATCH_SIZE to achieve the best throughput with acceptible latency on your device:
```bash
docker run --rm -it \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:25.06-py3 \
  /usr/src/tensorrt/bin/trtexec \
    --onnx=/models/model.onnx \
    --minShapes=signal:1x2400x1,state:1x219729 \
    --optShapes=signal:$BATCH_SIZEx2400x1,state:$BATCH_SIZEx219729 \
    --maxShapes=signal:$BATCH_SIZEx2400x1,state:$BATCH_SIZEx219729 \
    --builderOptimizationLevel=5 \
    --stronglyTyped \
    --useSpinWait \
    --noDataTransfers \
    --saveEngine=/models/model.plan
```

- `1` in `minShapes` stands for minimal possible batch size.
- `2400` denotes the length of input tensor. Formula: `length = sample rate (Hz) * chunk size (sec)`. For our model `8000 * 0.3 = 2400`
- `219729` denotes the overall size of a hidden state generated for the previous audio segment. 
- After conversion `trtexec` will report performance metrics of obtained TensorRT engine.

## How to deploy model in Triton Inference Server

Before deploy, the model must be prepared and saved in a format compatible with Triton. Create a directory structure as follows:

```
models
└── streaming_acoustic
    ├── 1
    │   └── model.onnx or model.plan
    └── config.pbtxt
```

Copy the required `config.pbtxt` file from the `configs` folder:
- `configs/streaming_acoustic/config.pbtxt` for ONNX Runtime;
- `configs/streaming_acoustic_trt/config.pbtxt` for TensorRT.

Then prepare the model:

- For ONNX Runtime: simply download the model:
```bash
python -m tone download $(pwd)/models/streaming_acoustic/1 --only-acoustic
```

- For TensorRT: convert the downloaded ONNX model to TensorRT format with instructions above. Any version of TensorRT 10.x is supported.

Once the model is prepared, launch the Triton Server using Docker:

```bash
docker run --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:25.06-py3 \
  tritonserver --model-repository=/models
```
