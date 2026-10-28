import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import build_graph, extract_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "checkpoints.db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск: инициализация AsyncSqliteSaver...")
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer)
        app.state.checkpointer = checkpointer
        logger.info("Граф скомпилирован с AsyncSqliteSaver.")
        yield
    logger.info("Остановка: соединение с SQLite закрыто.")


app = FastAPI(title="LangGraph Weather Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse_pack(data: dict, event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_agent_response(request: Request, user_input: str, thread_id: str):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    streamed = ""
    try:
        async for chunk, metadata in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="messages",
        ):
            # Только токены узла answer (без classify/grade/rewrite)
            if metadata.get("langgraph_node") != "answer":
                continue

            text = extract_text(chunk)
            if not text:
                continue

            # Дедупликация: LangGraph иногда присылает полный накопленный текст отдельным чанком
            if text == streamed:
                continue
            if text.startswith(streamed) and len(text) > len(streamed):
                new_part = text[len(streamed):]
                streamed = text
                if new_part:
                    yield sse_pack({"token": new_part})
                continue

            streamed += text
            yield sse_pack({"token": text})

        # После завершения графа берём финальное состояние (уже после fix_sun_times)
        final_state = await graph.aget_state(config)
        final_answer = final_state.values.get("final_answer", streamed)

        # Отправляем done с финальным (возможно исправленным) ответом
        yield sse_pack(
            {"status": "done", "final_answer": final_answer},
            event="done",
        )

    except Exception as e:
        logger.error(f"Ошибка в потоке: {e}")
        yield sse_pack({"error": str(e)}, event="error")


@app.get("/chat/stream")
async def chat_stream(request: Request, question: str, thread_id: str = "default-user"):
    return StreamingResponse(
        stream_agent_response(request, question, thread_id),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)