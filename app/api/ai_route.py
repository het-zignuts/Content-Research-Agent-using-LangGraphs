from app.graph.graph import build_graph
from fastapi import APIRouter, Query, UploadFile, File, HTTPException, status
from typing import List, Dict
from app.config.config import Config
from app.ingestion.ingestion import ingest_docs
import os
import shutil
import uuid
from app.utils.utils import cleanup_session
from fastapi.responses import FileResponse
from app.schemas.schemas import AgentResponse

# A router for handling AI research requests, which includes uploading documents, invoking the LangGraph for processing, and returning the response along with any generated reports. 
# It also includes a route for downloading generated reports.
router = APIRouter(prefix="/ai", tags=["AI Content Research Agent"])


@router.post("/ai-research", response_model=AgentResponse, status_code=status.HTTP_200_OK)
def ai_research(query: str = Query(..., description="Research query to ask the agent"), files: List[UploadFile] = File(...)):
    """
    Endpoint to handle AI research requests. It accepts a research query and a list of files to be ingested.
    """
    session_id = str(uuid.uuid4())  # generate a unique session ID for this research session
    try:
        UPLOAD_PATH = f"{Config.UPLOAD_PATH}/{session_id}"  # create a unique upload path for this session to store the uploaded files
        os.makedirs(UPLOAD_PATH, exist_ok=True)  # create a directory for this session's uploads
        file_paths: List[str] = []
        for file in files:
            file_name = f"{str(uuid.uuid4())}_{file.filename}"  # generate a unique filename to avoid conflicts
            file_path = os.path.join(UPLOAD_PATH, file_name)  # generate the full file path for storage
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)  # save the uploaded file to disk
            file_paths.append(file_path)
        
        ingest_docs(session_id, file_paths)  # ingest the uploaded documents (load, chunk, embed)

        graph = build_graph()  # build the LangGraph for this session and query
        response = graph.invoke({"session_id": session_id, "query": query, "report_md": None, "answer": None})  # invoke the graph with the session ID and user query to get the response

        resp_status=200
        resp_message="Agent Answered the Query"

        # ensure there's always an answer
        if response.get("answer") is None:
            response["answer"] = "Sorry, I could not find an answer to your question based on the provided documents."
            resp_status=400
            resp_message="The agent couldn't successfully answer the query."
        # save the report markdown if generated and add a download URL
        if response.get("report_md"):
            report_filename = f"report_{session_id}.md"
            report_path = os.path.join(Config.REPORT_STORE_PATH, report_filename)
            with open(report_path, "w") as f:
                f.write(response["report_md"])  # save the report markdown to a file for download
            response["report_url"] = f"/reports/download/{report_filename}"  # include the report URL in the response for the frontend to access
            
        agent_body = AgentResponse(status=resp_status, message=resp_message, data=response)
        return agent_body
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # Catch the ValueError from tool_selector_node
    except Exception as e:
        raise e  # raise any exceptions that occur during processing to be handled by FastAPI's error handlers
    finally:
        cleanup_session(session_id)  # clean up uploaded files and vector DB for this session to free up space, executed before returning the response


@router.get("/reports/download/{report_filename}")
def download(report_filename: str):
    """
    Endpoint to download generated reports. It serves the markdown report file for the given filename.
    """
    return FileResponse(
        f"{Config.REPORT_STORE_PATH}/{report_filename}",  # serve the report file from the report store directory
        media_type="text/markdown",  # set the media type to markdown for proper handling by the browser or client
        filename=report_filename  # set the filename for the downloaded file
    )