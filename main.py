from fastapi import FastAPI
from app.api.ai_route import router as ai_router

app=FastAPI(title="Content Research Agent")

app.include_router(ai_router)

@app.get("/")
def health():
    return {"status": "ok"}