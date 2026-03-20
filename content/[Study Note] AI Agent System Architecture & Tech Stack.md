---
title: AI Agent System Architecture & Tech Stack
category: AI, Tech Stack, AI Agent, Agent
tags:
  - AI
  - Agent
  - Architecture
---

## 1. AI 에이전트(Agent) 개요

AI 에이전트는 대규모 언어 모델(LLM)을 단순한 텍스트 생성기가 아닌 **'추론 엔진(Reasoning Engine)'**으로 활용하는 시스템이다. 사용자의 요청을 분석하고, 스스로 계획을 수립하며, 외부 시스템(API, DB 등)과 상호작용하여 자율적으로 목표를 달성하는 **상태 머신(State Machine)**이자 **이벤트 루프(Event Loop)** 형태로 동작한다.

## 2. 핵심 아키텍처 및 구성 요소 (4 Core Modules)

### A. 제어 루프 및 오케스트레이션 (Control Loop & Orchestration)

단방향 Request-Response가 아닌, 종료 조건이 충족될 때까지 순환하는 제어 흐름이다.

- **동작 주기:** 요청 수신 $\rightarrow$ 추론 $\rightarrow$ 도구 선택 $\rightarrow$ 행동 실행 $\rightarrow$ 관찰 $\rightarrow$ 재추론.
    
- **백엔드 설계 시 고려사항 (비효율 및 리스크):**
    
    - **무한 루프 방지:** 하드코딩된 최대 반복 횟수(Max Steps) 제한 및 스텝별 타임아웃 설정 필수.
        
    - **비용 및 지연 최적화:** 매 루프마다 전체 컨텍스트가 전송되므로 토큰 비용과 Latency가 급증함. 라우팅을 담당하는 경량 모델과 복잡한 추론을 담당하는 무거운 모델을 분리하는 아키텍처가 필요하다.
        

### B. 도구 사용 및 I/O 제어 (Tool Calling / Function Calling)

에이전트가 외부 세계(DB, 결제 시스템, 메일 등)에 개입하는 인터페이스.

- **동작 방식:** 사용 가능한 함수 명세(이름, 설명, 파라미터 타입)를 LLM에 주입하면, LLM이 실행할 함수와 JSON 인자값을 반환한다.
    
- **백엔드 설계 시 고려사항 (에러 핸들링):**
    
    - LLM의 파라미터 타입 오류나 존재하지 않는 함수 호출(Hallucination) 방어.
        
    - Pydantic 등을 통한 스키마 검증 필수.
        
    - 검증 실패 시 단순 에러 종료가 아닌, **에러 메시지를 LLM에 피드백으로 반환하여 스스로 파라미터를 수정하게 유도하는 복구 루프(Error Feedback Loop)** 구현이 핵심이다.
        

### C. 계획 및 인지 아키텍처 (Planning & Cognitive Architecture)

- **ReAct (Reasoning and Acting):** 생각 $\rightarrow$ 행동 $\rightarrow$ 관찰의 단일 경로 순차 처리. 단순하지만 복잡한 문제 처리에 한계가 있음.
    
- **Plan-and-Solve / Work Tree:** 요청을 분석하여 하위 태스크(Sub-tasks)들의 DAG(Directed Acyclic Graph)를 생성. 워커 노드들이 트리를 탐색하며 병렬/직렬로 도구를 실행하여 환각을 최소화함.
    
- **Reflection (자가 검증):** 최종 결과 반환 전, 초기 요청 충족 여부를 에이전트 스스로 비판(Critique)하고 수정하는 파이프라인.
    

### D. 메모리 및 상태 관리 (Memory & State Management)

- **단기 메모리:** 현재 세션의 컨텍스트 윈도우(메시지 배열).
    
- **장기 메모리:** 과거 기록, 문서 등을 Vector DB 등에 저장하고 의미 기반 검색(RAG)으로 필요할 때 주입.
    
- **백엔드 설계 시 고려사항:** 대화가 길어질수록 발생하는 'Lost in the middle' (핵심 정보 누락) 현상 방지. 백엔드에서 오래된 대화를 비동기로 요약 처리하거나, 최근 $N$개 메시지만 유지하는 슬라이딩 윈도우 로직 적용이 필요하다.
    

## 3. 실무 도입 기술 스택 (Tech Stacks)

### A. 프레임워크 (Frameworks)

- **LangChain:** 프로토타이핑에 유리하나, 백엔드 프로덕션 적용 시 과도한 추상화(Over-abstraction)로 인해 에러 트레이싱과 커스텀 확장이 불리하다.
    
- **LangGraph:** 노드와 엣지로 구성된 그래프 형태로 에이전트의 상태(State)와 순환(Cyclic) 워크플로우를 정의. 조건부 분기, 롤백 등 백엔드의 상태 머신 제어와 유사하여 현재 복잡한 에이전트 설계의 표준으로 자리 잡음.
    
- **DSPy:** 개발자의 수동 프롬프트 엔지니어링 대신, 모델 스스로 프롬프트를 최적화(컴파일)하는 선언적 프레임워크.
    

### B. 데이터 인프라 및 LLMOps

- **LlamaIndex:** RAG 파이프라인(수집, 청킹, 임베딩, 검색) 구축 특화 프레임워크.
    
- **Vector DB:** Milvus, Qdrant 등이 주류이나, 인프라 복잡도를 낮추려면 기존 Redis의 RediSearch 모듈 활용이 아키텍처 단순화에 효율적이다.
    
- **LangSmith / Phoenix:** LLM 애플리케이션의 APM. 노드별 프롬프트 입출력, 지연 시간, 토큰 비용 등을 트레이싱하여 시스템 병목을 파악하는 필수 관측성(Observability) 도구.