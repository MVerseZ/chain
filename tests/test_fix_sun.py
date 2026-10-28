# tests/test_fix_sun.py
import pytest
from agent import fix_sun_times


def make_state(answer: str) -> dict:
    return {"final_answer": answer}


async def test_swaps_reversed_sun_times():
    """Если рассвет вечером, а закат утром — значения меняются местами."""
    state = make_state("Восход в 17:16, закат в 07:11.")
    result = await fix_sun_times(state)

    assert "final_answer" in result
    fixed = result["final_answer"]
    assert "Восход в 07:11" in fixed
    assert "закат в 17:16" in fixed


async def test_leaves_correct_sun_times_untouched():
    """Корректные значения не должны меняться."""
    state = make_state("Восход в 07:11, закат в 17:16.")
    result = await fix_sun_times(state)
    assert result == {}


async def test_ignores_answer_without_sun_times():
    """Если в ответе нет времени восхода/заката — ничего не делаем."""
    state = make_state("Сегодня ясно, +4°C.")
    result = await fix_sun_times(state)
    assert result == {}


async def test_handles_only_sunrise():
    """Если есть только рассвет — не падаем и ничего не меняем."""
    state = make_state("Восход в 17:16.")
    result = await fix_sun_times(state)
    assert result == {}