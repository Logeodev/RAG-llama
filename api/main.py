from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from rag import RAG_Agent, embed_file_to_chroma
import os

app = FastAPI()

@app.get("/")
def serve_upload_form():
    return FileResponse(os.path.join(os.path.dirname(__file__), "uploadForm.html"))

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except Exception as e:
        return {"status": "error", "detail": f"Unable to decode file as UTF-8: {str(e)}"}
    result = embed_file_to_chroma(file.filename, text)
    return {"status": "embedded", "result": result}

@app.get("/rag")
def rag_query(q: str):
    return RAG_Agent().run(q)