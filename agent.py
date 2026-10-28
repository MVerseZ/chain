import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tavily import TavilyClient

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

print(
    "OPENROUTER_API_KEY:", "OK" if os.environ.get("OPENROUTER_API_KEY") else "НЕ НАЙДЕН"
)
print("TAVILY_API_KEY:", "OK" if os.environ.get("TAVILY_API_KEY") else "НЕ НАЙДЕН")

llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    request_timeout=120.0,
    max_retries=5,
)

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: NotRequired[str]
    intent: NotRequired[Literal["search", "answer"]]
    search_results: NotRequired[list[dict]]
    final_answer: NotRequired[str]
    standalone_question: NotRequired[str]
    search_query: NotRequired[str]
    relevance: NotRequired[str]
    retry_count: NotRequired[int]
    max_retries: NotRequired[int]


MAX_RETRIES = 2
KEEP_LAST_MESSAGES = 4
SUMMARIZE_THRESHOLD = 6

# --- Все узлы теперь async ---


async def summarize_history(state: AgentState) -> dict:
    messages = state.get("messages", [])
    if len(messages) <= SUMMARIZE_THRESHOLD:
        return {}

    old_messages = messages[:-KEEP_LAST_MESSAGES]
    old_text = ""
    for msg in old_messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        old_text += f"{role}: {msg.content}\n"

    existing_summary = state.get("summary", "")
    if existing_summary:
        prompt = f"Обнови резюме диалога.\n\nСуществующее резюме:\n{existing_summary}\n\nНовые сообщения:\n{old_text}\n\nВерни только резюме."
    else:
        prompt = f"Сожми диалог в краткое резюме.\n\nДиалог:\n{old_text}\n\nВерни только резюме."

    new_summary = (await llm.ainvoke(prompt)).content.strip()
    removals = [RemoveMessage(id=m.id) for m in old_messages if getattr(m, "id", None)]
    return {"summary": new_summary, "messages": removals}


async def classify_intent(state: AgentState) -> dict:
    history_text = ""
    for msg in state["messages"][-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    last_question = state["messages"][-1].content
    summary_context = ""
    if state.get("summary"):
        summary_context = f"\nРезюме предыдущего диалога:\n{state['summary']}\n"

    prompt = f"""Твоя задача — две вещи:

1. Определи, нужен ли для ответа поиск в интернете.
   - SEARCH нужен, если ответ зависит от СВЕЖИХ данных: погода, курсы валют, новости, цены, расписания, события, «что сейчас», «сегодня», «в этом году».
   - ANSWER достаточен, если это ОБЩЕИЗВЕСТНЫЙ факт, определение, история, география, математика, перевод — то, что не меняется со временем.

2. Перепиши последний вопрос так, чтобы он был понятен БЕЗ контекста диалога. Замени неявные отсылки ("там", "туда", "у него", "а также", "а что насчёт") на конкретные объекты из истории.

{summary_context}
История последних сообщений:
{history_text}

Последний вопрос: {last_question}

ПРИМЕРЫ:
- "Какая погода в Москве?" → INTENT: search
- "Столица Франции?" → INTENT: answer
- "Кто написал Войну и мир?" → INTENT: answer
- "Какой курс доллара сегодня?" → INTENT: search
- "Сколько будет 2+2?" → INTENT: answer
- "А какая там погода?" (после разговора про Париж) → INTENT: search, STANDALONE: "Какая погода в Париже?"

Ответь СТРОГО в таком формате (две строки, без пояснений):
INTENT: search
STANDALONE: <самостоятельный вопрос>

или

INTENT: answer
STANDALONE: <самостоятельный вопрос>"""

    response = (await llm.ainvoke(prompt)).content.strip()

    intent = "answer"
    standalone = last_question

    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("INTENT:"):
            value = line.split(":", 1)[1].strip().lower()
            intent = "search" if "search" in value else "answer"
        elif line.upper().startswith("STANDALONE:"):
            standalone = line.split(":", 1)[1].strip()

    if not standalone:
        standalone = last_question

    return {
        "intent": intent,
        "standalone_question": standalone,
        "search_query": standalone,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
    }


async def search_web(state: AgentState) -> dict:
    query = (
        state.get("search_query")
        or state.get("standalone_question")
        or state["messages"][-1].content
    )
    # Tavily синхронный — выносим в поток, чтобы не блокировать event loop
    results = await asyncio.to_thread(tavily.search, query=query, max_results=3)
    return {"search_results": results.get("results", [])}


async def grade_results(state: AgentState) -> dict:
    question = state.get("standalone_question") or state["messages"][-1].content
    results = state.get("search_results", [])
    if not results:
        return {"relevance": "not_relevant"}

    snippets = "\n\n".join([r.get("content", "")[:300] for r in results])
    prompt = f"Оцени релевантность результатов.\n\nВопрос: {question}\n\nРезультаты:\n{snippets}\n\nОтветь: 'relevant' или 'not_relevant'."
    response = (await llm.ainvoke(prompt)).content.strip().lower()
    relevance = (
        "relevant"
        if "relevant" in response and "not" not in response
        else "not_relevant"
    )
    return {"relevance": relevance}


async def rewrite_query(state: AgentState) -> dict:
    question = state.get("standalone_question") or state["messages"][-1].content
    retry_count = state.get("retry_count", 0) + 1
    prompt = (
        f"Перепиши вопрос для поиска.\n\nВопрос: {question}\n\nВерни только вопрос."
    )
    new_query = (await llm.ainvoke(prompt)).content.strip()
    return {"search_query": new_query, "retry_count": retry_count}


async def generate_answer(state: AgentState) -> dict:
    context = ""
    if state.get("search_results"):
        snippets = [r.get("content", "") for r in state["search_results"]]
        context = "\n\n".join(snippets)

    summary_section = ""
    if state.get("summary"):
        summary_section = f"\nКонтекст диалога:\n{state['summary']}\n"

    question = state.get("standalone_question") or state["messages"][-1].content

    system_prompt = f"""Ты — дружелюбный ассистент.
{summary_section}
Контекст из поиска: {context if context else "Нет контекста."}

ПРАВИЛА:
- Отвечай только на заданный вопрос.
- Используй только факты из контекста.
- Если есть восход/закат — используй точно как написано.
- Включи все погодные данные из контекста (температура, ветер, влажность, давление, восход/закат).
- Стиль: живо, с **жирным** для цифр, кратко."""

    messages_for_llm = [("system", system_prompt), ("human", question)]
    answer = (await llm.ainvoke(messages_for_llm)).content
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


async def fix_sun_times(state: AgentState) -> dict:
    answer = state.get("final_answer", "")
    sunrise_match = re.search(
        r"(?:Рассвет|Восход)[^\d]*(\d{1,2}:\d{2})", answer, re.IGNORECASE
    )
    sunset_match = re.search(r"(?:Закат)[^\d]*(\d{1,2}:\d{2})", answer, re.IGNORECASE)
    if not (sunrise_match and sunset_match):
        return {}

    sunrise_time = sunrise_match.group(1)
    sunset_time = sunset_match.group(1)
    sunrise_h = int(sunrise_time.split(":")[0])
    sunset_h = int(sunset_time.split(":")[0])
    if not (sunrise_h >= 12 and sunset_h < 12):
        return {}

    positions = [
        (sunrise_match.start(1), sunrise_match.end(1), sunset_time),
        (sunset_match.start(1), sunset_match.end(1), sunrise_time),
    ]
    positions.sort(key=lambda x: x[0], reverse=True)
    fixed = answer
    for start, end, replacement in positions:
        fixed = fixed[:start] + replacement + fixed[end:]

    logger.info("fix_sun_times: исправлены перепутанные рассвет/закат")
    return {"final_answer": fixed}


def route_after_classification(state: AgentState) -> Literal["search", "answer"]:
    return state["intent"]


def route_after_grading(state: AgentState) -> Literal["rewrite", "answer"]:
    relevance = state.get("relevance", "not_relevant")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)
    if relevance == "relevant":
        return "answer"
    if retry_count < max_retries:
        return "rewrite"
    return "answer"


def build_graph(checkpointer):
    """Собирает и компилирует граф с переданным чекпойнтером."""
    builder = StateGraph(AgentState)
    builder.add_node("summarize", summarize_history)
    builder.add_node("classify", classify_intent)
    builder.add_node("search", search_web)
    builder.add_node("grade", grade_results)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("answer", generate_answer)
    builder.add_node("fix_sun", fix_sun_times)

    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "classify")
    builder.add_conditional_edges(
        "classify", route_after_classification, {"search": "search", "answer": "answer"}
    )
    builder.add_edge("search", "grade")
    builder.add_conditional_edges(
        "grade", route_after_grading, {"rewrite": "rewrite", "answer": "answer"}
    )
    builder.add_edge("rewrite", "search")
    builder.add_edge("answer", "fix_sun")
    builder.add_edge("fix_sun", END)

    return builder.compile(checkpointer=checkpointer)


def extract_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""
