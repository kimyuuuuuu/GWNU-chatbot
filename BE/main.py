from dotenv import load_dotenv
from fastapi import FastAPI, Query

from chatbot import process_question

load_dotenv()
app = FastAPI()


@app.get("/")
def read_root():
    return {"hello": "this is fastapi chatbot server \n have a good dev!"}


@app.get("/chat/sync")
def chat_sync(question: str = Query(None, min_length=3, max_length=100)):
    response, retrieve_docs = process_question(is_sync=True, user_question=question)
    return {"response": response}


@app.get("/chat/async")
def chat_async(question: str = Query(None, min_length=3, max_length=100)):
    response, retrieve_docs = process_question(is_sync=False, user_question=question)
    return {"response": response}
