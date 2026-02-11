# Content Research Agent:
An intelligent, session-based Content Research Agent built using LangGraph, Groq LLM, FastAPI, and RAG architecture.


## Overview:
This system allows users to upload multiple documents and perform advanced research tasks such as:
- Summarization
- Question Answering
- Document Comparison
- Information Extraction
- Insight Generation
It uses a multi-node LangGraph workflow, to achieve the above functionalities.


## Project Structure:

```text
Content-Research-Agent-using-LangGraphs
├── app
│   ├── api
│   │   ├── __init__.py
│   │   └── ai_route.py
│   ├── config
│   │   ├── __init__.py
│   │   └── config.py
│   ├── db
│   │   ├── __init__.py
│   │   └── vector_db.py
│   ├── graph
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── nodes
│   │   │   ├── __init__.py
│   │   │   ├── compare.py
│   │   │   ├── extract.py
│   │   │   ├── insight.py
│   │   │   ├── qna.py
│   │   │   ├── tool_selector.py
│   │   │   └── summarize.py
│   │   └── state.py
│   ├── ingestion
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── chunker.py
│   │   ├── embed.py
│   │   └── loader.py
│   ├── llms
│   │   ├── __init__.py
│   │   └── groq.py
│   └── utils
│       ├── __init__.py
│       └── utils.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```


## Project and environment setup:

1. Clone the project repo folder using the following command:
```git
git clone https://github.com/het-zignuts/Content-Research-Agent-using-LangGraphs.git
```

2. Inside the repo folder, create and activate the python environment using the following commands:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

3. Once activated, install the dependencies as:
```python
pip install -r requirements.txt
```

4. Create a .env file in the folder with the following fields (replace the value-placeholders with actual values):
```.env
GROQ_API_KEY=<your-groq-api-key>
MODEL=<your-llm-model-name>
VECTOR_DB_DIR=app/data/vector_dbs
UPLOAD_DIR=app/data/uploads
```


## Running the app:

1. Run the app using the following command:
```bash
uvicorn main:app --reload
```
    The app runs at http://127.0.0.1:8000 by default.

2. Open the API docs at:

    http://127.0.0.1:8000/docs : Swagger UI

    http://127.0.0.1:8000/redoc : ReDoc UI


## Features:

### Intelligent Task Routing
A dedicated Tool Selector Node classifies the user query into:
- summarize
- qna
- compare
- extract
- insight
The correct node is dynamically triggered via conditional edges in LangGraph.

### RAG (Retrieval-Augmented Generation)
- Document loading
- Chunking
- Embedding creation
- Vector database storage (session-scoped)
- Context-aware retrieval

### Comparison Engine
- Groups retrieved chunks by document
- Produces structured tabular comparison
- Includes source citations (document name + page)

### Report Generation
- Markdown reports generated for extraction tasks
- Downloadable via API endpoint

### Clean Session Lifecycle
- Unique session ID per request
- Temporary upload directory
- Automatic cleanup in finally block


## API Summary:

| Endpoint | Method | Description | Inputs | Outputs |
|-----------|--------|------------|--------|---------|
| `/ai/ai-research` | `POST` | Main research endpoint. Uploads documents, ingests them, runs LangGraph workflow, and returns AI-generated results. | `query` (string), `files` (List[UploadFile]) | `answer` (string), optional `report_url` (string) |
| `/ai/reports/download/{report_filename}` | `GET` | Downloads generated markdown report file. | `report_filename` (path param) | Markdown file download |


## Graph Nodes

| Node Name      | Function Handler        | Usage |
|---------------|------------------------|--------|
| `tool_selector` | `tool_selector_node` | Classifies the user query into one of the predefined tasks (`qna`, `compare`, `summarize`, `extract`, `insight`). |
| `retrieve` | `retrieve_node` | Performs semantic similarity search using FAISS vector store and groups retrieved chunks by document. |
| `qna` | `qna_node` | Answers specific factual questions based on retrieved document context. |
| `compare` | `compare_node` | Compares information across multiple documents and returns structured comparative output. |
| `insight` | `insight_node` | Generates analytical insights, recommendations, or higher-level interpretations from documents. |
| `summarize` | `summarize_node` | Produces a concise or structured summary of the retrieved documents. |
| `extract` | `extract_node` | Extracts specific structured data points or generates section-based reports from documents. |


## Graph Edges:

| From | To | Type | Remarks |
|------|----|------|---------|
| `START` | `tool_selector` | Direct | Entry point of the workflow. |
| `tool_selector` | `retrieve` | Direct | After task classification, documents are retrieved from vector DB. |
| `retrieve` | `qna` | Conditional | Routed when `state["task"] == "qna"`. |
| `retrieve` | `compare` | Conditional | Routed when `state["task"] == "compare"`. |
| `retrieve` | `insight` | Conditional | Routed when `state["task"] == "insight"`. |
| `retrieve` | `summarize` | Conditional | Routed when `state["task"] == "summarize"`. |
| `retrieve` | `extract` | Conditional | Routed when `state["task"] == "extract"`. |
| `qna` | `END` | Direct | Workflow terminates after answer generation. |
| `compare` | `END` | Direct | Workflow terminates after comparison output. |
| `insight` | `END` | Direct | Workflow terminates after insight generation. |
| `summarize` | `END` | Direct | Workflow terminates after summary generation. |
| `extract` | `END` | Direct | Workflow terminates after extraction/report generation. |