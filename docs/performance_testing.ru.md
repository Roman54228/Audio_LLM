# 📈 Тестирование производительности

**Предисловие**: Мы оцениваем производительность только акустической модели, так как она является наиболее ресурсоемкой частью всего пайплайна.

Для достижения максимальной производительности потоковой акустической модели и использующих её сервисов мы рекомендуем использовать следующие варианты:

- [`Microsoft ONNX Runtime`](https://github.com/microsoft/onnxruntime/) + [`Nvidia Triton Inference Server`](https://github.com/triton-inference-server/server) = 🏃💨
Простая в использовании, но не самая быстрая конфигурация — используется в качестве бейзлайна.
- [`NVIDIA TensorRT`](https://github.com/NVIDIA/TensorRT/) + [`NVIDIA Triton inference server`](https://github.com/triton-inference-server/server) = ⚡️🏃💨💨
Гораздо более быстрая альтернатива (в 2-3 раза), но требует конвертации модели в формат TensorRT.

## Запуск сервера с моделью

Перед запуском тестов производительности модель должна быть экспортирована в `ONNX` или `TensorRT` и запущена в Triton Inference Server. Смотрите раздел [Triton Inference Server](docs/triton_inference_server.ru.md) с подробными инструкциями.

## Замеры производительности

### trtexec

Для запуска тестов производительности акустической модели в формате `TensorRT` можно использовать cli утилиту [**`trtexec`**](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html).

**Ключевые флаги:**

- `--warmUp`: Длительность "прогрева" в миллисекундах перед началом измерений.
- `--duration=`: Продолжительность основного измерения в секундах.
- `--iterations`: В качестве альтернативы, запустить фиксированное количество итераций вместо выполнения по времени.
- `--avgRuns`: Запустить бенчмарк N раз и усреднить результаты для большей стабильности.
- `--verbose` или `-v`: Выводит подробные логи о загрузке и выполнении engine.

**Пример:**

```bash
trtexec --loadEngine=<engine_file.plan> --warmUp=2000 --duration=10 --avgRuns=5
```

### perf_analyzer

Вы можете использовать `perf_analyzer` из Triton SDK контейнера для запуска тестов производительности как для `ONNX`, так и для `TensorRT`. Для этого запустите контейнер с SDK:

```bash
docker run --rm -it --net host \
  nvcr.io/nvidia/tritonserver:24.02-py3-sdk
```

Обратите внимание, что версия SDK — `24.02`, поскольку в версии `25.06` имеется [проблема с OOM](https://github.com/triton-inference-server/perf_analyzer/issues/84)

Затем выполните команду `perf_analyzer` с параметрами, подходящими для потокового ASR:

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

**Ключевые флаги:**

- `--sequence-length`: Длина входных последовательностей. sequence-length = 1 соответствует 300 мс аудио
- `--request-rate-range`: Диапазон частот запросов для тестирования
- `--latency-threshold`: Устанавливает пороговое значение задержки в миллисекундах

## Результаты

Пример вывода `perf_analyzer` для конфигурации с `RTX 3090` и `TensorRT`, размер батча `16`:
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

Вы можете также вычислить пропускную способность по формуле: `SPS = inferences/sec * chuck size (sec)`. Для приведённого выше примера это даёт пропускную способность `3000 * 0.3 = 900 SPS`, при этом задержка на один чанк остаётся ниже 100 мс.
