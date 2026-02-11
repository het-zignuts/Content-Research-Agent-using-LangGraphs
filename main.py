from fastapi import FastAPI
from app.api.ai_route import router as ai_router

# Instantiate the FastAPI application.
app=FastAPI(title="Content Research Agent")

# adding the AI research router. 
app.include_router(ai_router)

# A simple health check endpoint at rooot url.
@app.get("/")
def health():
    return {"status": "ok"}