

AI Agent가 터미널 실행이나 파일 수정과 같은 실제 OS 환경의 작업을 수행할 수 있는 것은 LLM이 직접 OS에 접근하기 때문이 아니라, **도구 사용(Tool Use)** 메커니즘과 이를 제어하는 **에이전트 런타임**의 결합을 통해 이루어집니다. 

이 문서는 AI Agent의 핵심 원리부터 엔터프라이즈 백엔드 환경에서의 프로덕션 레벨 구현까지의 기술적 흐름을 정리합니다.

---

## 1. AI Agent의 핵심 동작 원리

### 1.1. Tool Use (Function Calling)
LLM 자체는 텍스트를 생성하는 모델이지만, 특정 상황에서 미리 정의된 함수를 호출하도록 훈련되어 있습니다.

* **정의:** 개발자는 에이전트에게 "파일을 읽으려면 `read_file(path)` 함수를 쓰고, 터미널 명령을 실행하려면 `execute_command(cmd)`를 써라"라고 가이드(System Prompt)와 함께 API 명세를 제공합니다.
* **프로세스:**
    1. 사용자가 "현재 디렉토리의 파일 목록을 보여줘"라고 요청합니다.
    2. LLM은 직접 명령을 실행하는 대신, `{"name": "execute_command", "arguments": {"cmd": "ls -al"}}`라는 **구조화된 데이터(JSON)**를 출력합니다.
    3. 이 출력을 받은 **에이전트 런타임(Python 백엔드 등)**이 실제로 OS 환경에서 해당 명령을 실행합니다.
    4. 실행 결과(표준 출력)를 다시 LLM에게 텍스트로 전달하면, LLM이 이를 해석해 사용자에게 답변합니다.

### 1.2. 에이전트 런타임 (Sandboxed Environment)
에이전트는 독립적인 실행 환경을 가집니다. 외부 환경에 대한 제어권을 가지므로 보안과 격리가 필수적입니다.

* **Docker Container:** 에이전트가 실행하는 모든 명령이 호스트 시스템에 영향을 주지 않도록 격리된 컨테이너 내부에서 수행됩니다.
* **ReAct(Reasoning and Acting) 루프:** LLM이 '생각(Thought) -> 행동(Action) -> 관찰(Observation)'의 단계를 반복하며 작업을 완수할 때까지 루프를 제어하는 프레임워크가 작동합니다.

### 1.3. 구조적 차이: LLM vs AI Agent

| 구분 | LLM (Chat) | AI Agent |
| :--- | :--- | :--- |
| **목적** | 질문에 대한 텍스트 응답 생성 | 주어진 목표(Goal) 달성 및 실행 |
| **입출력** | Text in / Text out | Text in / Action out (API 호출 등) |
| **권한** | 없음 (샌드박스 내부 텍스트 생성) | 환경 제어권 (File I/O, Network, Shell) |
| **상태 유지** | 대화 기록 중심 | 작업 상태 및 환경 변화 추적 |



---

## 2. 프레임워크 동작 과정 (LangGraph 기반)

과거의 Agent Executor 방식과 달리, 최근의 엔터프라이즈 환경에서는 제어 흐름, 에러 핸들링, 상태 관리를 명시적으로 설계할 수 있는 **상태 머신(State Machine)** 기반의 그래프 접근법(예: LangGraph)을 표준으로 사용합니다.



### 2.1. LangGraph 실행 파이프라인
에이전트 시스템은 노드(Node)와 엣지(Edge)로 구성된 순환 그래프로 동작합니다.

1. **상태(State) 초기화:** 에이전트와 사용자의 전체 대화 기록(메시지 리스트)을 담는 전역 상태 객체가 생성됩니다.
2. **Agent Node (LLM 추론):** * LLM이 현재까지의 상태를 입력받아 분석합니다.
    * 외부 도구가 필요하다면 `tool_calls` 객체(실행할 함수 이름과 파라미터)를 포함한 메시지를 반환합니다. 도구가 필요 없다면 최종 텍스트 응답을 반환합니다.
3. **Conditional Edge (분기점):**
    * Agent Node의 출력에 `tool_calls`가 포함되어 있는지 검사합니다.
    * 있다면 **Tool Node**로 라우팅하고, 없다면 **END**로 라우팅하여 루프를 종료합니다.
4. **Tool Node (도구 실행):**
    * LLM이 생성한 파라미터를 파싱하여 실제 등록된 함수(예: 쉘 명령어 실행)를 실행합니다.
    * 실행 결과(표준 출력 또는 에러 메시지)를 `ToolMessage` 형태로 래핑하여 전체 상태에 추가합니다.
5. **순환:** 상태가 업데이트된 후 다시 Agent Node로 돌아가, LLM이 도구 실행 결과를 바탕으로 다음 행동을 결정합니다.

---

## 3. 작업 완료 판단 및 예외 통제 메커니즘

에이전트가 작업의 완료 여부를 판단하는 핵심 주체는 프레임워크가 아니라 **LLM 자체**입니다. 프레임워크는 LLM의 출력 형태를 파싱하여 조건부 라우팅(Conditional Routing)을 수행할 뿐입니다.

### 3.1. 작업 완료 판단 프로세스


1. **System Prompt 지시:** 목표 달성 시 최종 답변을 제공하라는 명시적 가이드 주입.
2. **상태 평가 (State Evaluation):** 도구 실행 후 누적된 `ToolMessage` 기록을 LLM이 다시 입력받음.
3. **추론 (Reasoning):** * **불충분:** 추가 정보/작업이 필요하다고 판단하여 새로운 `tool_call` 생성.
    * **충분:** 요청이 충족되었다고 판단하여 일반 텍스트(최종 답변) 생성.
4. **라우팅 (Routing):** 백엔드 코드가 LLM 응답에 `tool_calls` 속성이 비어있는지 검사 후 종료 노드(`END`)로 이동.

### 3.2. 백엔드 관점의 아키텍처 방어 로직 (비판적 검토)
LLM의 확률적 추론에 전적으로 의존하는 구조는 시스템 불안정성을 초래하므로, 다음과 같은 백엔드 엔지니어링 통제가 필수적입니다.

* **무한 루프 방지 (Recursion Limit):** 권한 없음이나 파일 부재 등으로 도구 실행이 계속 실패할 때, LLM이 무한히 재시도하는 것을 막기 위해 에이전트 런타임에 명시적인 `max_steps` 제약을 두어야 합니다.
* **환각(Hallucination)에 의한 조기 종료 통제:** 터미널 명령이 실패했음에도 LLM이 임의로 "수정 완료"라고 답변하며 종료하는 것을 막기 위해, 도구 실행 결과에 대한 하드 코딩된 정규식/상태 코드 검증 로직이 필요합니다.
* **구조화된 종료 강제 (Structured Output):** LLM이 단순 텍스트로 답변하며 종료하는 대신, 완료 판단 자체를 하나의 도구(예: `SubmitFinalAnswer`)로 정의하여 이 도구를 호출해야만 루프가 종료되도록 강제해야 합니다. 이를 통해 백엔드는 항상 정형화된 JSON 응답을 보장받을 수 있습니다.

---

## 4. 프로덕션 레벨 에이전트 구현 (Python)

아래 코드는 상태 관리, 타입 안정성, 무한 루프 제어, 그리고 구조화된 종료(Structured Termination)가 제어된 Google Style Guide 기반의 LangGraph 에이전트 구현체입니다.


### 4.1. 주요 설계 원칙 적용
1. **명시적인 타입 검증 (Pydantic):** `FinalAnswerSchema`를 통해 LLM이 도구 파라미터를 누락하거나 틀리게 전달할 경우 프레임워크 단에서 즉각적인 오류 반환 및 자가 정정 유도.
2. **구조화된 종료:** `submit_final_answer` 도구 호출을 통해서만 `is_finished = True` 상태 값을 변경하여 확정적 종료 보장.
3. **애플리케이션 레벨의 루프 제어:** State 객체 내부에 `step_count`를 기록하여 프레임워크 크래시 대신 클라이언트에게 정제된 Timeout/에러 JSON 응답 가능.
4. **Sandbox API 분리:** 호스트 쉘(`subprocess`) 실행 안티 패턴을 피하고, 원격 Sandbox API(gRPC)로 명령을 위임하는 구조.

### 4.2. Core Implementation

```python
import json
import logging
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------
class AgentState(BaseModel):
    """에이전트의 전체 실행 상태를 관리하는 데이터 모델."""
    messages: Annotated[List[BaseMessage], add_messages]
    step_count: int = 0
    max_steps: int = Field(default=10, description="최대 루프 허용 횟수")
    is_finished: bool = False

# ---------------------------------------------------------------------------
# 2. Structured Output & Tools 
# ---------------------------------------------------------------------------
class FinalAnswerSchema(BaseModel):
    """최종 답변 제출을 강제하기 위한 스키마."""
    status: Literal["success", "failure"] = Field(
        ..., description="작업의 최종 성공 여부"
    )
    result_data: str = Field(
        ..., description="사용자에게 전달할 최종 결과물 또는 실패 사유"
    )

@tool
def execute_secure_terminal_command(command: str) -> str:
    """격리된 샌드박스 환경에서 터미널 명령어를 실행합니다."""
    logger.info(f"요청된 명령어: {command}")
    
    # [Architecture Note] 실제 호스트 실행 금지. gRPC 원격 실행으로 대체 필요.
    if "rm " in command or "sudo " in command:
        return "Error: Permission denied. Dangerous commands are blocked."
    
    return f"Executed in sandbox: {command}\nOutput: Success."

@tool(args_schema=FinalAnswerSchema)
def submit_final_answer(status: str, result_data: str) -> str:
    """작업이 완료되었을 때 반드시 호출해야 하는 종료 도구."""
    return f"Final answer submitted with status: {status}"

# ---------------------------------------------------------------------------
# 3. Node Functions
# ---------------------------------------------------------------------------
def call_model(state: AgentState) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [execute_secure_terminal_command, submit_final_answer]
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke(state.messages)
    
    return {
        "messages": [response],
        "step_count": state.step_count + 1
    }

def execute_tools(state: AgentState) -> Dict[str, Any]:
    last_message = state.messages[-1]
    if not last_message.tool_calls:
        return {}

    new_messages = []
    is_finished = False

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        try:
            if tool_name == "execute_secure_terminal_command":
                result = execute_secure_terminal_command.invoke(tool_args)
            elif tool_name == "submit_final_answer":
                result = submit_final_answer.invoke(tool_args)
                is_finished = True 
            else:
                result = f"Error: Unknown tool {tool_name}"
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            result = f"Execution failed: {str(e)}"
            
        new_messages.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

    return {"messages": new_messages, "is_finished": is_finished}

# ---------------------------------------------------------------------------
# 4. Routing Logic
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    # 방어 로직 1: 최대 스텝 초과 시 강제 종료
    if state.step_count >= state.max_steps:
        logger.warning("Max steps reached. Forcing termination.")
        return END
        
    # 방어 로직 2: 정상적인 완료 프로세스 (SubmitFinalAnswer 호출됨)
    if state.is_finished:
        logger.info("Agent submitted final answer. Terminating cleanly.")
        return END
        
    last_message = state.messages[-1]
    
    # 도구 호출이 있으면 도구 노드로 라우팅
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
        
    # 에러 상황: 도구 호출도 없고 완료도 안 된 상태에서 LLM이 멈춤
    logger.error("LLM stopped without calling tools or submitting answer.")
    return END

# ---------------------------------------------------------------------------
# 5. Graph Assembly & Execution
# ---------------------------------------------------------------------------
def build_agent() -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

if __name__ == "__main__":
    agent_app = build_agent()
    
    system_prompt = (
        "당신은 시스템 관리 에이전트입니다. 주어진 목표를 해결하기 위해 도구를 사용하세요. "
        "모든 작업이 끝나면 반드시 'submit_final_answer' 도구를 호출하여 종료를 선언하십시오. "
        "일반 텍스트로 종료를 선언하지 마십시오."
    )
    
    initial_state = AgentState(
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content="프로젝트 루트 디렉토리의 파일 목록을 확인하고, 결과가 정상인지 보고해줘.")
        ]
    )
    
    for event in agent_app.stream(initial_state):
        for key, value in event.items():
            logger.info(f"Node [{key}] output processed.")
```

🗂️ Categories & Tags

- #AI 
	
- #Agent 
	
- #LLM 
	
- #Backend 
	
- #Architecture 
	
- #LangGraph

## 📊 Graph

- [[AI 모델 추론 시스템 아키텍처 및 커널 최적화 (ML Sys)]]