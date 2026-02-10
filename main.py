from fastapi import FastAPI

app=FastAPI(title="Content Research Agent")


@app.get("/")
def health():
    return {"status": "ok"}