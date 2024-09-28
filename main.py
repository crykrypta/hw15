from fastapi import FastAPI, Query, HTTPException

from fastapi_models import LLMQuery
from LLM import LLM

# Инициализация класса LLM
llm = LLM('Simble.txt')

# Инициализация FastAPI
app = FastAPI()


# Маршруты FastAPI:

# Root маршрут
@app.get('/')
async def get_root():
    return {'message': 'Hello World'}


# /help маршрут
@app.get('/help')
async def get_help():
    return {'message': 'Help Page'}


# Маршрут - получение ответа от модели
@app.get('/llm/get_answer')
async def get_answer(query: LLMQuery):
    return {'answer': llm.get_answer(query.text)}


# ДЗ Lite (API-Калькулятора)

# Сложение
@app.get('/calc/sum')
async def sum(
    a: float = Query(..., description='Первое число для сложения'),
    b: float = Query(..., description='Второе число для сложения')
):
    return {'answer': a + b}


# Вычитание
@app.get('/calc/subtract')
async def subtract(
    a: float = Query(..., description='Первое число для вычитания'),
    b: float = Query(..., description='Второе число для вычитания')
):
    return {'answer': a - b}


# Умножение
@app.get('/calc/multiply')
async def multiply(
    a: float = Query(..., description='Первое число для умножения'),
    b: float = Query(..., description='Второе число для умножения')
):
    return {'answer': a * b}


# Деление
@app.get('/calc/divide')
async def divide(
    a: float = Query(..., description='Первое число для деления'),
    b: float = Query(..., description='Второе число для деления')
):
    if b == 0:
        raise HTTPException(status_code=400, detail='Деление на ноль')
    return {'answer': a / b}
