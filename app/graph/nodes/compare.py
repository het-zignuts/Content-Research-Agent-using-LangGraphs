from app.llms.groq import get_groq_llm
from app.schemas import ComparisonSchema
from pydantic import ValidationError, BaseModel
from typing import Dict
import json

def compare_node(state):
    """
    Compare Node for the LangGraph. This node takes the grouped documents retrieved from the vector database and compares them based on the user query. It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference. The node then defines a comparison prompt that instructs the LLM to compare and contrast the documents in a structured tabular format, ensuring that each point of comparison is clearly identified along with its source. The LLM is invoked with this prompt, and the generated comparison is returned as the answer in the state for downstream processing or response generation.
    """

    COMPARISON_PROMPT = """
    ═══════════════════════════════════════════════════════════════════════════════
    SYSTEM ROLE
    ═══════════════════════════════════════════════════════════════════════════════

    You are a **comparison assistant**.

    ═══════════════════════════════════════════════════════════════════════════════
    CORE TASK
    ═══════════════════════════════════════════════════════════════════════════════

    **COMPARE AND CONTRAST** all documents provided in the context below, based on the user query.

    ═══════════════════════════════════════════════════════════════════════════════
    OUTPUT FORMAT REQUIREMENTS
    ═══════════════════════════════════════════════════════════════════════════════

    CRITICAL: JSON Structure Only 

    You **MUST** respond with a VALID JSON OBJECT ONLY:

    NO markdown code fences ("```json" or "```")
    NO preamble or explanation  
    NO extra text outside the JSON

    ───────────────────────────────────────────────────────────────────────────────
    Required JSON Schema
    ───────────────────────────────────────────────────────────────────────────────

    {{
        "answer": "<comparison_table_and_conclusion>"
    }}

    ───────────────────────────────────────────────────────────────────────────────
    Content of "answer" Field
    ───────────────────────────────────────────────────────────────────────────────

    The "answer" value **MUST** be a *single string* containing:

    1. **Markdown Comparison Table**
    
    - Format: "| Point of Comparison | [Doc1_Name] | [Doc2_Name] | ... |"
    
    - Each cell MUST be **concise**
    
    - Each cell MUST include citation:
        "value (source: [doc_name], page: [N])"
    
    - Table must be CLEAN with no excessive spaces
    
    - Proper markdown table formatting REQUIRED

    2. **Conclusion Section**
    
    - Start on a NEW LINE after the table
    
    - Format: "Conclusion: [your_summary]"
    
    - Keep **concise** and **evidence-based**

    ───────────────────────────────────────────────────────────────────────────────
    JSON String Escaping Rules
    ───────────────────────────────────────────────────────────────────────────────

    IMPORTANT: Since the answer is inside a JSON string:

    - Use "\\n" for newlines (NOT actual newline characters)
    
    - Escape all special characters properly (so that they are parsable)
    
    - The entire answer MUST be a valid JSON string value

    ═══════════════════════════════════════════════════════════════════════════════
    CONTENT RULES
    ═══════════════════════════════════════════════════════════════════════════════

    MUST:

    1. **STRICTLY** adhere to provided context ONLY
    
    2. Cite sources with document name AND page number
    
    3. Keep facts SEPARATED by source document

    DO NOT:

    1. Mix facts across documents
    
    2. **HALLUCINATE** or infer beyond provided context
    
    3. Add information NOT present in the documents

    ═══════════════════════════════════════════════════════════════════════════════
    FALLBACK RESPONSE
    ═══════════════════════════════════════════════════════════════════════════════

    If the query is **OUT OF CONTEXT** or **UNANSWERABLE** from provided documents:

    {{"answer": "Couldn't generate comparison, based on provided context..."}}

    ═══════════════════════════════════════════════════════════════════════════════
    EXAMPLE OUTPUT
    ═══════════════════════════════════════════════════════════════════════════════

    (Structure reference only - NOT real data)

    {{"answer": "| Point of Comparison | Phone A | Phone B |\\n|---|---|---|\\n| Display | 6.1 inch OLED (source: docA, page: 1) | 6.5 inch LCD (source: docB, page: 2) |\\n| Battery | 4000mAh (source: docA, page: 2) | 4500mAh (source: docB, page: 3) |\\n\\nConclusion: Phone B has a larger battery while Phone A offers a superior display technology."}}

    ═══════════════════════════════════════════════════════════════════════════════
    INPUT VARIABLES
    ═══════════════════════════════════════════════════════════════════════════════

    **CONTEXT:**
    {context}

    **USER QUERY:**
    {query}
    """

    llm=get_groq_llm(temperature=0.0, model_kwargs=ComparisonSchema) # initialize the GROQ LLM with a temperature of 0.0 for deterministic output.

    docs=state["grouped_docs"] # get the grouped docs from the sate
    context="" # initialize an empty string to build the context for the LLM prompt
    for doc_id, contents in docs.items():
        doc_name=contents[0]["doc_name"] # retrieve documant name
        context+=f"Document Name: {doc_name}\n"
        context+=f"Context from {doc_name}:\n"
        for content in contents:
            context+=f"- {content['content']}\n Page Number: {content['page_number']}\n" # add each content piece from the document to the context, along with its page number for reference

    try:
        response=llm.invoke(COMPARISON_PROMPT.format(context=context, query=state["query"])) # Invoke the LLM with the formatted prompt, passing in the constructed context and the user query to generate the comparison output based on the provided documents and the user's request
        data = json.loads(response.content)
        return {
            "answer": data["answer"],
        }
    except ValidationError as e:
        print(f"Resposne validation failed: {e}")
    except Exception as e:
        print(f"Comparison Error: {str(e)}")
        return {
            "answer": "I encountered an error processing the documents.",
        }