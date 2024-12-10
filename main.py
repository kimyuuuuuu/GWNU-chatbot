from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from chatbot import process_question, process_question_streaming

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


@app.get("/chat/streaming_sync")
def chat_streaming_sync(question: str = Query(None, min_length=3, max_length=100)):
    def event_stream():
        try:
            response, retrieve_docs = process_question_streaming(
                is_sync=True, user_question=question
            )

            for chunk in response:
                if len(chunk) > 0:
                    yield f"data: {chunk} "
        except Exception as e:
            yield f"error: {str(e)}"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/chat/streaming_async")
def chat_streaming_async(question: str = Query(None, min_length=3, max_length=100)):
    def event_stream():
        try:
            response, retrieve_docs = process_question_streaming(
                is_sync=False, user_question=question
            )

            for chunk in response:
                if len(chunk) > 0:
                    yield f"data: {chunk} "
        except Exception as e:
            yield f"error: {str(e)}"

    return StreamingResponse(event_stream(), media_type="text/event-stream")