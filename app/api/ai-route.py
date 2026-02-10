from app.graph.graph import build_graph
from fastapi import APIRouter, Query, UploadFile, File
from typing import List
from app.config.config import Config
from app.ingestion.ingestion import ingest_docs
import os
import shutil
import uuid
from app.utils.utils import cleanup_session
from fastapi.responses import FileResponse

router=APIRouter(prefix="/ai", tags=["AI Content Research Agent"])

@router.post("/ai-research")
def ai_research(query: str = Query(..., description="Research query to ask the agent"), files: List[UploadFile] = File(...)):
    session_id=str(uuid.uuid4()) # generate a unique session ID for this research session
    try:
        UPLOAD_PATH=f"{Config.UPLOAD_PATH}/{session_id}"
        os.makedirs(UPLOAD_PATH, exist_ok=True) # create a directory for this session's uploads
        file_paths=[]
        for file in files:
            file_name=f"{file.filename}_{str(uuid.uuid4())}" # generate a unique filename to avoid conflicts
            file_path=os.path.join(UPLOAD_PATH, file_name) # generate the full file path for storage
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer) # save the uploaded file to disk
            file_paths.append(file_path)
        
        ingest_docs(session_id, file_paths) # ingest the uploaded documents (load, chunk, embed)

        graph=build_graph() # build the LangGraph for this session and query
        response=graph.invoke({"session_id": session_id, "query": query, "report_md": None, "answer": None}) # invoke the graph with the session ID and user query to get the response

        response=response.model_dump() # convert the response to a dictionary for JSON serialization
        if response["answer"] is None:
            response["answer"]="Sorry, I could not find an answer to your question based on the provided documents." # provide a default answer if the graph did not generate one
        if response["report_md"]:
            report_filename=f"report_{session_id}.md"
            report_path=os.path.join(Config.REPORTS_PATH, report_filename)
            with open(report_path, "w") as f:
                f.write(response["report_md"]) # save the report markdown to a file for download
            response["report_url"]=f"/reports/download/{report_filename}" # include the report URL in the response for the frontend to access
        return response
    
    except Exception as e:
        return {"error": str(e)} # return any exceptions as error messages in the response

    finally:
        cleanup_session(session_id) # clean up uploaded files and vector DB for this session to free up space

@router.get("/reports/download/{report_filename}")
def download(report_filename: str):
    return FileResponse(
        f"{Config.REPORT_STORE_PATH}/{report_filename}",
        media_type="text/markdown",
        filename=report_filename
    )
