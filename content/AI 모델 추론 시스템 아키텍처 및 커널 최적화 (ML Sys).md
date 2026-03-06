## 1. 훈련(Training)과 추론(Inference)의 시스템적 차이

최근 AI 업계의 트렌드는 거대 모델의 사전 학습에서 **추론 연산 최적화 및 서빙**으로 이동 중이다. 훈련과 추론은 시스템 요구사항과 VRAM 사용 구조가 근본적으로 다르다.

- **훈련 (Training):** 순전파(Forward) 후 역전파(Backward)를 통해 가중치(Weights)의 기울기(Gradient)를 업데이트. 이전 활성화(Activation) 값들을 모두 저장해야 하므로 막대한 VRAM과 처리량(Throughput) 중심의 대역폭이 필요.
    
- **추론 (Inference):** 고정된(Read-Only) 가중치를 바탕으로 순전파 연산(행렬 곱셈)만 수행. 실시간 응답을 위한 **지연 시간(Latency)** 최소화와, 토큰 생성 시 누적되는 **KV Cache**의 메모리 단편화 관리가 핵심.
    
- **추론 시간 확장 (Inference-time Scaling):** o1 모델 등에서 사용되는 기법으로, 추론 시에 새로운 가중치를 사용하는 것이 아니라 고정된 가중치 위에서 탐색 알고리즘(예: Tree of Thoughts, MCTS)을 통해 순전파를 반복하여 논리적 검증(CoT)을 수행하는 아키텍처.
    

---

## 2. 양자화 (Quantization) 아키텍처

학습된 모델의 FP16/FP32 가중치와 활성화 값을 INT8/INT4 등 저정밀도 이산 공간으로 매핑하여 VRAM 요구량과 메모리 I/O 병목을 줄이는 기법.

### 2.1. 수학적 원리와 한계

양자화는 토큰 ID의 충돌이 아니라 실수 매핑 과정에서의 **정밀도 손실(Precision Loss)**을 유발한다. 이를 보정하기 위해 스케일(Scale)과 제로 포인트(Zero-point)를 계산한다.

- $S = \frac{x_{max} - x_{min}}{q_{max} - q_{min}}$
    
- $Z = \text{round}\left(q_{min} - \frac{x_{min}}{S}\right)$
    
- $x_q = \text{round}(\frac{x_f}{S}) + Z$
    

### 2.2. 핵심 최적화 기법 (PTQ 기준)

1. **그룹 단위 양자화 (Group-wise Quantization):** 행렬 전체가 아닌 64~128개 파라미터 단위로 $S, Z$를 독립 계산하여 국소적 분포 보존.
    
2. **아웃라이어 보존 (Mixed Precision):** 크기가 비정상적으로 큰 극단값(Outlier)은 FP16을 유지하고 나머지만 INT8로 처리(LLM.int8).
    


``` python
import numpy as np

def quantize_tensor(x: np.ndarray, q_min: int = -128, q_max: int = 127):
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (q_max - q_min) if (x_max - x_min) != 0 else 1e-9
    zero_point = np.round(q_min - (x_min / scale))
    x_q = np.clip(np.round((x / scale) + zero_point), q_min, q_max).astype(np.int8)
    return x_q, scale, zero_point

def mixed_precision_quantize_with_outliers(x: np.ndarray, threshold: float = 6.0):
    outlier_mask = np.abs(x) > threshold
    x_outliers_fp16 = np.zeros_like(x, dtype=np.float16)
    x_outliers_fp16[outlier_mask] = x[outlier_mask].astype(np.float16)
    
    x_normal = np.copy(x)
    x_normal[outlier_mask] = 0.0 
    x_q_int8, scale, zp = quantize_tensor(x_normal)
    return x_q_int8, x_outliers_fp16, outlier_mask, scale, zp
```

### 2.3. W4A16 vs W8A8 구조의 차이

- **W4A16:** 가중치(Weight)만 INT4로 저장하고 활성화(Activation)는 FP16 유지. VRAM I/O를 극대화하기 위해 하드웨어 연산기(Tensor Core) 통과 전 SRAM에서 INT4 $\rightarrow$ FP16으로 **역양자화(Dequantization)** 수행 필수.
    
- **W8A8:** 가중치와 활성화 모두 INT8 적용. 연산 자체는 극단적으로 빠르나, INT8 연산 결과는 오버플로우를 막기 위해 **INT32로 누산(Accumulation)**됨. 다음 레이어로 가기 위해 FP16 스케일링 후 아웃라이어를 처리하며 다시 INT8로 재양자화하는 까다로운 라우팅 파이프라인 존재.
    

### 2.4. AWQ 양자화 생성 파이프라인 (AutoAWQ)

단순 압축이 아닌, 실제 캘리브레이션 데이터를 순전파시켜 **활성화 값 기준의 아웃라이어 채널을 스캔하고 스케일링 보정**을 수행하는 오프라인 로직.


``` python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "Qwen/Qwen2.5-7B-Instruct" 
quant_save_path = "./qwen-7b-custom-awq" 

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 캘리브레이션 및 활성화 기반 아웃라이어 스케일링 보정
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_save_path)
tokenizer.save_pretrained(quant_save_path)
```

---

## 3. 커널 퓨전 (Kernel Fusion)

VRAM I/O 속도가 하드웨어 연산 속도를 따라가지 못하는 **메모리 장벽(Memory Wall)**을 해결하기 위한 GPU 커널 레벨의 최적화.

역양자화, 행렬 곱셈, 활성화 함수 적용을 각각 별도의 커널로 실행하지 않고, 데이터를 SRAM에 한 번 로드하여 레지스터 단에서 모두 처리 후 VRAM에 최종 저장.


``` python
import triton
import triton.language as tl

@triton.jit
def fused_dequant_matmul_relu_kernel(
    x_ptr, w_int4_ptr, scale_ptr, output_ptr, M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # 1. VRAM -> SRAM 데이터 로드 (유일한 Read)
        x_block = tl.load(x_ptr + ...) 
        w_int4_block = tl.load(w_int4_ptr + ...) 
        scale_block = tl.load(scale_ptr + ...)

        # 2. 역양자화 (SRAM 내부)
        w_fp16_block = w_int4_block.to(tl.float16) * scale_block
        # 3. 행렬곱 누산 (SRAM 내부)
        accumulator += tl.dot(x_block, w_fp16_block)

    # 4. 활성화 함수 (SRAM 내부)
    accumulator = tl.where(accumulator > 0, accumulator, 0.0)

    # 5. 최종 결과 VRAM 저장 (유일한 Write)
    tl.store(output_ptr + ..., accumulator.to(tl.float16))
```

---

## 4. 어텐션 연산 최적화 (Memory IO & Scheduling)

### 4.1. FlashAttention-1: Tiling과 Online Softmax

$O(N^2)$ 크기의 어텐션 행렬을 VRAM에 기록하지 않고, 블록 단위로 쪼개어(Tiling) SRAM에서 계산. Softmax 연산의 수치적 한계를 해결하기 위해 로컬 최댓값($m$)과 지수합($l$)을 업데이트하며 글로벌 스케일을 실시간으로 보정하는 **Online Softmax** 적용.


``` python
import numpy as np

def flash_attention_simulated(Q, K, V, block_size=2):
    N, d = Q.shape
    O, l = np.zeros((N, d)), np.zeros((N, 1))
    m = np.full((N, 1), -np.inf)
    scale = 1.0 / np.sqrt(d)
    
    for j in range(0, N, block_size):
        K_j, V_j = K[j:j+block_size], V[j:j+block_size]
        S_ij = np.dot(Q, K_j.T) * scale 
        
        # Online Softmax 로컬 통계량 갱신 및 Rescaling
        m_j = np.max(S_ij, axis=1, keepdims=True)
        m_new = np.maximum(m, m_j)
        
        P_ij = np.exp(S_ij - m_new)
        l_j = np.sum(P_ij, axis=1, keepdims=True)
        correction_factor = np.exp(m - m_new)
        l_new = l * correction_factor + l_j
        
        O = (O * l * correction_factor + np.dot(P_ij, V_j)) / l_new
        m, l = m_new, l_new
        
    return O
```

### 4.2. FlashAttention-2: Loop Order & Sequence Parallelism

- **문제점:** FA1은 바깥쪽 루프가 $K, V$이고 안쪽 루프가 $Q$여서 임시 $O_i$ 결과를 VRAM에 빈번히 읽고 쓰는 오버헤드 발생.
    
- **개선:** $Q$를 바깥쪽 루프로 스와핑하여 SRAM 내부에서 임시값을 확정 짓도록 변경. 또한 $Q$ 시퀀스 길이를 병렬화의 축으로 삼아 GPU의 SM(Streaming Multiprocessor) Occupancy를 극대화.
    

### 4.3. Flash-Decoding (추론 최적화)

디코딩 시점에는 $Q$ 시퀀스 길이가 1로 고정되므로 FA2의 병렬화가 무력화됨. 대신 끝없이 길어지는 **KV Cache 자체(Sequence Dimension)를 분할(Split-K)**하여 여러 스레드 블록에 할당. 연산 후 단일 Reduction 커널을 통해 글로벌 통계량 기준으로 최종 값을 수학적 동기화.

---

## 5. 페이징 기반 메모리 관리 (PagedAttention)

서버 환경에서 여러 요청의 KV Cache가 연속된 VRAM을 요구하며 발생하는 내부/외부 단편화(Fragmentation)를 해결.

운영체제의 가상 메모리 페이징 기법을 도입하여, 물리적 VRAM 블록(Block)을 작게 쪼개고 요청 토큰들을 논리적인 블록 테이블(Block Table)에 불연속적으로 매핑.

- **효과:** 메모리 단편화 소멸 (유휴 공간 4% 미만 유지), CoW(Copy-on-Write) 및 시스템 프롬프트 공유 가능. 처리량(Throughput) 비약적 상승.
    

---

## 6. vLLM 기반 서빙 파이프라인 (추론 엔진)

### 6.1. 비동기 추론 서버 (Docker Compose)

W4A16 커널 퓨전과 PagedAttention을 강제로 활성화하여 띄우는 OpenAI 규격의 로컬 서버 설정.


``` yaml
version: '3.8'
services:
  vllm-server:
    image: vllm/vllm-openai:latest
    container_name: vllm-inference
    runtime: nvidia
    ports:
      - "8000:8000"
    command: >
      --model Qwen/Qwen2.5-7B-Instruct-AWQ
      --quantization awq
      --gpu-memory-utilization 0.9 # PagedAttention KV Cache Pool 확보
      --max-num-batched-tokens 8192
      --enforce-eager
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 6.2. 오프라인 추론 및 프로파일링 (In-Process)

HTTP 지연시간을 배제하고 커널 레벨의 VRAM 할당 동작 및 Continuous Batching 효율을 테스트.


``` python
from vllm import LLM, SamplingParams
import time

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    quantization="awq",
    gpu_memory_utilization=0.9, 
    max_model_len=4096,
    enforce_eager=True 
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
prompts = ["Explain Kernel Fusion.", "What is PagedAttention?"] * 10 

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
total_time = time.time() - start_time

print(f"Raw Throughput: { (256 * len(prompts)) / total_time :.2f} tokens/s")
```

---

## 🗂️ Categories & Tags

- **#AI_Architecture**
    
- **#Machine_Learning_Systems_MLSys**
    
- **#Quantization**
    
- **#CUDA_Optimization**
    
- **#Memory_Management**
    
- **#vLLM**