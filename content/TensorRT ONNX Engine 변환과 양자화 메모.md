---
title: TensorRT ONNX Engine 변환과 양자화 메모
category:
  - AI
  - Inference
  - Model Optimization
  - TensorRT
tags:
  - AI/Inference
  - AI/ModelOptimization
  - TensorRT
  - TensorRT/Engine
  - TensorRT/Builder
  - TensorRT/Runtime
  - TensorRT/Quantization
  - TensorRT/Calibration
  - ONNX
  - ONNX/Graph
  - ONNX/Visualization
  - CUDA/Kernel
  - GPU/Memory
aliases:
  - TensorRT Engine Build
  - ONNX to TensorRT Engine
  - TensorRT Quantization Calibration
created: 2026-07-07
source: Codex 대화 정리
---

# TensorRT ONNX Engine 변환과 양자화 메모

관련 노트: [[AI 모델 추론 시스템 아키텍처 및 커널 최적화]]

이 문서는 PyTorch 모델, ONNX, TensorRT engine 사이의 관계와, TensorRT build 과정에서 일어나는 최적화/양자화/시각화 가능 범위를 정리한 메모다.

---

## 1. 기본 변환 흐름

일반적인 배포 흐름은 다음과 같이 이해하면 된다.

```text
.pt / PyTorch model
  -> export
.onnx
  -> TensorRT build
.engine / .plan
  -> TensorRT runtime deserialize
inference 실행
```

핵심은 `.onnx`와 `.engine`의 성격이 다르다는 점이다.

- `.onnx`: 프레임워크 중립적인 모델 표현이다. 연산 그래프, tensor, attribute, weight initializer를 담는다.
- `.engine` / `.plan`: TensorRT builder가 특정 GPU, CUDA, TensorRT 환경에 맞춰 만든 하드웨어 종속 실행 계획이다.

즉, `.onnx`는 단순히 NN 구조만 있는 파일이 아니다. 일반적으로 weight도 initializer 형태로 포함한다. 아주 큰 모델은 ONNX external data 방식으로 weight를 별도 파일에 둘 수 있지만, 개념적으로는 ONNX model의 일부다.

---

## 2. TensorRT build 단계에서 일어나는 일

TensorRT builder는 ONNX를 읽어서 다음을 수행한다.

1. 그래프 최적화
   - 예: Conv + BatchNorm + ReLU 같은 연산을 fusion할 수 있다.

2. CUDA kernel / tactic 선택
   - 같은 convolution이라도 GPU 아키텍처, 입력 shape, precision, workspace 조건에 따라 빠른 구현이 다르다.
   - TensorRT는 layer별로 가능한 tactic을 평가하거나 선택해 engine에 반영한다.

3. precision 처리
   - TensorRT 10.x까지는 `--fp16`, `--int8` 같은 builder flag로 reduced precision 사용을 지시하는 흐름이 흔했다.
   - TensorRT 11.x에서는 strongly typed network가 기본이 되면서 이런 precision flag와 implicit quantization 흐름이 제거됐다.

4. 메모리 계획
   - activation buffer, scratch/workspace, persistent memory 사용 방식을 정한다.
   - inference 중간 tensor의 lifetime을 보고 memory reuse를 최적화한다.

5. engine serialization
   - 최적화된 graph, tactic 정보, plugin 정보, 보통 weight까지 포함한 serialized binary를 만든다.

중요한 오해 포인트:

> inference 직전에 weight 파일을 따로 dump하는 것이 일반 흐름은 아니다.

일반적인 engine은 build 시점에 weight를 포함한다. runtime에서는 engine을 deserialize하면서 weight와 실행에 필요한 상태를 host/GPU memory에 준비한다. 예외는 weight stripping, refit, weight streaming 같은 특수 기능을 사용한 경우다.

---

## 3. ONNX 시각화

`.onnx` 파일은 시각화할 수 있다. 가장 흔한 도구는 Netron이다.

```text
.onnx -> Netron
```

Netron에서 확인할 수 있는 것:

- input/output tensor 이름과 shape
- Conv, Relu, Resize, MatMul 같은 node/op 흐름
- weight/constant initializer
- layer attribute
- 일부 subgraph/control-flow 구조

단, Netron에서 보는 것은 ONNX graph 기준이다. TensorRT가 build하면서 적용한 layer fusion, tactic 선택, CUDA kernel 선택, workspace 계획은 ONNX만 열어서는 보이지 않는다.

참고 링크:

- [Netron](https://netron.app)
- [Netron GitHub](https://github.com/lutzroeder/netron)
- [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html)

---

## 4. TensorRT engine 시각화와 inspect

TensorRT `.engine` 파일은 ONNX처럼 Netron에 바로 넣어서 그래프를 보는 표준 포맷이 아니다. 대신 TensorRT의 Engine Inspector와 `trtexec`를 사용해 layer 정보를 JSON으로 추출할 수 있다.

```bash
trtexec --loadEngine=model.engine --dumpLayerInfo --exportLayerInfo=model.trt.json
```

이 JSON에는 다음 정보가 포함될 수 있다.

- TensorRT layer 이름
- layer type
- input/output tensor shape
- datatype / format
- 선택된 tactic 이름
- fusion된 ONNX layer metadata
- binding / I/O tensor 정보

더 상세한 정보를 보고 싶으면 engine build 시점에 detailed profiling metadata를 포함해야 한다.

```bash
trtexec --onnx=model.onnx \
  --saveEngine=model.engine \
  --profilingVerbosity=detailed
```

이후 `--exportLayerInfo`로 뽑은 `.trt.json`은 NVIDIA Nsight Deep Learning Designer에서 열어 engine computation graph로 볼 수 있다.

```text
.engine
  -> trtexec --exportLayerInfo
.trt.json
  -> Nsight Deep Learning Designer
```

참고 링크:

- [TensorRT Engine Tools and Debugging](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-tools.html)
- [TensorRT trtexec command-line programs](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html)

---

## 5. 양자화와 calibration

질문 핵심: engine을 build하면서 quantization이나 calibration을 할 수 있는가?

답은 TensorRT 버전에 따라 다르다.

### 5.1 TensorRT 10.x까지의 전통적 흐름

TensorRT 10.x 이하에서는 build 중 INT8 calibration 흐름이 흔했다.

```text
FP32 ONNX
  -> TensorRT build 중 INT8 calibration
  -> INT8 engine 생성
```

예시:

```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --int8 --calib=calibration.cache
```

이때 calibration은 대표 입력 데이터를 흘려 activation의 dynamic range와 scale을 잡는 과정이다. weight를 다시 학습하는 것이 아니라, FP 값을 INT8 범위로 매핑하기 위한 보정값을 계산하는 단계다.

### 5.2 TensorRT 11.x 흐름

TensorRT 11.x에서는 `--int8`, `--calib`, `--fp16` 같은 precision flag 기반 흐름이 제거됐다. 최신 방향은 build 전에 모델을 명시적으로 quantized ONNX로 만드는 것이다.

```text
FP32 ONNX
  -> ModelOpt PTQ/QAT: calibration + Q/DQ 삽입
  -> quantized ONNX
  -> TensorRT build
  -> quantized engine
```

예시:

```bash
python -m modelopt.onnx.quantization \
  --onnx_path model.onnx \
  --calibration_data data.npz

trtexec --onnx=model_quantized.onnx --saveEngine=model.engine
```

TensorRT는 ONNX 안의 `QuantizeLinear` / `DequantizeLinear` 노드를 읽고, 그 semantics를 유지하는 방향으로 engine을 최적화한다.

### 5.3 FP16과 INT8 calibration의 차이

FP16 사용과 INT8 calibration은 다르다.

- FP16: 보통 FP16 kernel이나 FP16 tensor type을 사용할 수 있게 하는 mixed precision 최적화에 가깝다.
- INT8 calibration: 대표 데이터로 activation scale을 계산해 quantization error를 줄이는 보정 과정이다.

TensorRT 11.x에서는 FP16도 기존처럼 단순 builder flag로 켜는 방식이 아니라, ModelOpt AutoCast 등으로 모델 자체를 mixed precision 형태로 준비하는 방향이다.

참고 링크:

- [TensorRT Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [TensorRT 10.x to 11.x trtexec migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html)

---

## 6. 확인 방법

### 6.1 ONNX에 weight가 있는지 확인

How to check:

- Netron으로 `.onnx`를 열고 initializer/constant tensor를 확인한다.
- 파일 크기가 weight 크기와 비슷한지 본다.

Expected result:

- 일반적인 ONNX 파일이면 graph뿐 아니라 weight initializer가 보인다.
- external data 방식이면 ONNX 파일 옆에 별도 weight data 파일이 있을 수 있다.

### 6.2 engine 상세 정보 확인

How to check:

```bash
trtexec --loadEngine=model.engine --dumpLayerInfo --exportLayerInfo=model.trt.json
```

Expected result:

- JSON에 layer별 TensorRT 정보가 출력된다.
- 내용이 layer name 위주로만 빈약하면 engine을 `--profilingVerbosity=detailed`로 다시 build해야 한다.

### 6.3 build-time calibration인지 확인

How to check:

- 코드나 명령에서 `--int8`, `--calib`, `IInt8Calibrator`를 찾는다.
- ONNX 내부에 `QuantizeLinear` / `DequantizeLinear` 노드가 있는지 확인한다.

Expected result:

- `--int8`, `--calib`, `IInt8Calibrator`가 있으면 TensorRT 10.x식 build-time calibration 흐름일 가능성이 높다.
- ONNX에 Q/DQ 노드가 있고 build 명령에 precision flag가 없다면 calibration은 ONNX 생성 전 단계에서 끝난 구조다.

---

## 7. 요약

```text
ONNX
  = portable graph + usually weights
  = Netron으로 시각화 가능

TensorRT engine
  = hardware/software-specific optimized binary
  = TensorRT runtime에서 deserialize 후 실행
  = trtexec + Engine Inspector로 inspect 가능

Quantization
  TensorRT 10.x: build 중 INT8 calibration 가능
  TensorRT 11.x: ModelOpt 등으로 Q/DQ ONNX를 먼저 만들고 build
```
