"""This module is a minimal representation of an AI MVP."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)

app = FastAPI(title="MVP AI Service")


DOCUMENTS = [
    """
    Возврат товара возможен в течение 14 дней после покупки.
    Товар должен сохранять товарный вид и оригинальную упаковку.
    """,
    """
    Служба поддержки работает с понедельника по пятницу
    с 09:00 до 18:00.
    """,
    """
    Доставка обычно занимает от 2 до 5 рабочих дней.
    При отправке клиент получает номер для отслеживания.
    """,
]


class UserRequest(BaseModel):
    """A class for saving Requests from a user.

    Save a request in a str format.
    """

    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    """Response model for the ask endpoint."""

    question: str
    context: str | None
    answer: str


def search_documents(question: str) -> str | None:
    """Простейший поиск по совпадающим словам.

    Возвращает документ с наибольшим количеством совпадений.
    """
    question_words = {
        word.strip(".,!?;:()[]{}\"'").lower()
        for word in question.split()
        if len(word.strip(".,!?;:()[]{}\"'")) >= 3
    }

    best_document: str | None = None
    best_score = 0

    for document in DOCUMENTS:
        document_lower = document.lower()

        score = sum(1 for word in question_words if word in document_lower)

        if score > best_score:
            best_score = score
            best_document = document.strip()

    return best_document


def generate_answer(question: str, context: str | None) -> str:
    """Generate an answer using the LLM and optional document context."""
    if context is None:
        context_text = """
        В базе знаний не найдено подходящей информации.
        Не выдумывай факты. Если точного ответа нет,
        прямо скажи, что информации недостаточно.
        """
    else:
        context_text = f"""
        Информация из базы знаний:

        {context}
        """

    response = client.responses.create(
        model=model_name,
        instructions=(
            "Ты помощник службы поддержки. "
            "Отвечай понятно, кратко и на языке пользователя. "
            "Используй предоставленный контекст. "
            "Не придумывай информацию, которой нет в контексте. "
            "Если контекста недостаточно, честно скажи об этом."
        ),
        input=f"""
        Вопрос пользователя:
        {question}

        {context_text}
        """,
    )

    return response.output_text


@app.get("/")
def index() -> dict[str, str]:
    """Return basic service information."""
    return {
        "message": "MVP AI Service is running",
        "usage": "Send POST request to /ask",
    }


@app.post("/ask", response_model=AskResponse)
def ask_user(request: UserRequest) -> AskResponse:
    """Process a user question and return an AI-generated answer."""
    try:
        context = search_documents(request.question)
        answer = generate_answer(request.question, context)
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="Не удалось получить ответ от AI-модели.",
        ) from error

    return AskResponse(
        question=request.question,
        context=context,
        answer=answer,
    )
