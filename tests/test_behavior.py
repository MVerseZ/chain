# tests/test_behavior.py
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent import build_graph


@pytest.fixture
def graph():
    """Свежий граф с in-memory чекпойнтером для каждого теста."""
    return build_graph(MemorySaver())


async def run_and_collect(graph, question: str, thread_id: str):
    """
    Запускает граф и собирает:
      - список выполненных узлов (по stream_mode="updates")
      - финальное состояние
    """
    config = {"configurable": {"thread_id": thread_id}}
    visited = []

    async for event in graph.astream(
        {"messages": [HumanMessage(content=question)]},
        config,
        stream_mode="updates",
    ):
        # event — словарь {node_name: state_update}
        for node_name in event:
            visited.append(node_name)  # noqa: PERF402

    final_state = await graph.aget_state(config)
    return final_state.values, visited


@pytest.mark.asyncio
async def test_weather_question_triggers_search(graph):
    """Вопрос о погоде должен приводить к вызову узла search."""
    _, visited = await run_and_collect(graph, "Какая сейчас погода в Москве?", "t1")
    assert "search" in visited, f"Ожидался search, но visited = {visited}"


@pytest.mark.asyncio
async def test_capital_question_does_not_search(graph):
    """Вопрос о столице не должен вызывать поиск."""
    _, visited = await run_and_collect(graph, "Столица Франции?", "t2")
    assert "search" not in visited, (
        f"Поиск не должен был вызываться, но visited = {visited}"
    )
    assert "answer" in visited, f"Ожидался answer, но visited = {visited}"


@pytest.mark.asyncio
async def test_standalone_question_is_rewritten(graph):
    """
    «А какая там погода?» после вопроса о Париже должно
    превратиться в вопрос про Париж и вызвать поиск.
    """
    # 1. Контекст: спрашиваем про Париж
    await run_and_collect(graph, "Столица Франции?", "t3")

    # 2. Задаём вопрос с отсылкой «там»
    final_state, visited = await run_and_collect(graph, "А какая там погода?", "t3")

    assert "search" in visited, (
        f"Standalone-вопрос должен вызывать поиск, visited = {visited}"
    )

    standalone = final_state.get("standalone_question", "")
    assert "Париж" in standalone or "Paris" in standalone, (
        f"Standalone-вопрос не переписан: {standalone!r}"
    )


@pytest.mark.asyncio
async def test_sun_fix_node_runs(graph):
    """Узел fix_sun должен выполняться после answer на каждом запросе."""
    _, visited = await run_and_collect(graph, "Какая сейчас погода в Москве?", "t4")
    assert "fix_sun" in visited, f"Ожидался fix_sun, visited = {visited}"
