# ⚡️ Экспорт в TensorRT и запуск модели в Triton Inference Server

## Как экспортировать модель в TensorRT

TensorRT engine компилируется и оптимизируется под конкретную аппаратную и программную конфигурацию, на которой он будет использоваться (например, модель GPU, версия CUDA). Из-за такой аппаратной зависимости мы не можем предоставить универсальный готовый engine файл. Для получения лучшей производительности вам необходимо самостоятельно скомпилировать engine на вашей целевой конфигурации, которая будет использоваться для инференса.

- Сначала скачайте ONNX модель:

```bash
python -m tone download $(pwd)/models/model.onnx --only-acoustic
```

- Скомпилируйте ONNX модель в TensorRT формат. Выберите $BATCH_SIZE для достижения наилучшей пропускной способности с приемлемой задержкой на вашем устройстве:
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

- `1` в `minShapes` обозначает минимально допустимый размер батча.
- `2400` обозначает длину входного тензора. Формула для вычисления длины: `длина = частота дискретизации (Hz) * размер чанка (сек)`. Для нашей модели `8000 * 0.3 = 2400`.
- `219729` обозначает общий размер скрытых состояний, сгенерированных для предыдущего фрагмента аудио. 
- После компиляции `trtexec` предоставит отчёт с метриками производительности полученного TensorRT engine.

## Как разворачивать модель в Triton Inference Server

Для повышенной производительности разверните акустическую модель T-one в Triton. Для этого модель должна быть приведена к совместимому формату. Создайте структуру папок следующего вида:

```
models
└── streaming_acoustic
    ├── 1
    │   └── model.onnx или model.plan
    └── config.pbtxt
```

Скопируйте необходимый файл `config.pbtxt` из папки `configs`:
- `configs/streaming_acoustic/config.pbtxt` для ONNX Runtime;
- `configs/streaming_acoustic_trt/config.pbtxt` для TensorRT.

Затем подготовьте модель:

- Для ONNX Runtime: просто загрузите модель:
```bash
python -m tone download $(pwd)/models/streaming_acoustic/1 --only-acoustic
```

- Для TensorRT: конвертируйте загруженную модель из формата ONNX в формат TensorRT по инструкциям выше. Поддерживается любая версия TensorRT 10.x.

После подготовки модели запустите сервер Triton с помощью Docker:

```bash
docker run --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:25.06-py3 \
  tritonserver --model-repository=/models
```
