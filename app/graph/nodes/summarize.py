from app.llms.groq import get_groq_llm
from app.schemas import SummarizationSchema
from pydantic import ValidationError
import json

def summarize_node(state):
    """
    Summarization Node for the LangGraph. This node takes the retrieved documents from the vector database and generates a concise summary based on the user query. 
    It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference.
    """
    SUMMARY_PROMPT="""
    ═══════════════════════════════════════════════════════════════════════════════
    SYSTEM INSTRUCTION
    ═══════════════════════════════════════════════════════════════════════════════

    You are a **summarization assistant** .

    ═══════════════════════════════════════════════════════════════════════════════
    CORE TASK
    ═══════════════════════════════════════════════════════════════════════════════

    **SUMMARIZE** the retrieved content provided in the context below into **concise bullet points**.

    Your summary should:
    - Capture **MAIN IDEAS** and **KEY DETAILS**
    - Be **CLEAR** and **INFORMATIVE**
    - Focus on the most important information from the documents

    ═══════════════════════════════════════════════════════════════════════════════
    OUTPUT FORMAT REQUIREMENTS
    ═══════════════════════════════════════════════════════════════════════════════

    CRITICAL: Pydantic-Compatible JSON 

    You **MUST** return a VALID JSON OBJECT that STRICTLY survives Pydantic model validation.

    ───────────────────────────────────────────────────────────────────────────────
    Required JSON Schema
    ───────────────────────────────────────────────────────────────────────────────

    {{
        "answer": "[string]"
    }}

    ───────────────────────────────────────────────────────────────────────────────
    Field Specifications
    ───────────────────────────────────────────────────────────────────────────────

    **"answer"** (REQUIRED - string)

    - Format: "Answer: - [summary_point] [citation] \\n - [summary_point] [citation]"
    
    - Each bullet point MUST include citation
    
    - NO additional text before or after the JSON
    
    - The answer field should contain ONLY the summary points with citations

    ───────────────────────────────────────────────────────────────────────────────
    Summary & Citation Format Rules
    ───────────────────────────────────────────────────────────────────────────────

    **Structure:**

    "Answer: - [summary_point_1] [source: [doc_name], page: [N]] \\n - [summary_point_2] [source: [doc_name], page: [N]]"

    **Components:**

    1. **Answer Prefix**
    - Start with "Answer: "
    - REQUIRED at the beginning

    2. **Bullet Points**
    - Use "- " (dash + space) for each point
    - Keep each point concise and focused
    - Use "\\n " (newline + space) between points

    3. **Citations**
    - Format: "[source: [document_name], page: [page_number]]"
    - Place citation IMMEDIATELY after each bullet point
    - Space before citation: "[summary_point] [source: ...]"
    - EVERY bullet point MUST have a citation

    ═══════════════════════════════════════════════════════════════════════════════
    CONTENT RULES
    ═══════════════════════════════════════════════════════════════════════════════

    MUST:

    1. Base summary ONLY on the **PROVIDED CONTEXT**
    
    2. Include **CITATION** for EVERY bullet point
    
    3. Keep bullet points **CONCISE** and **FOCUSED**
    
    4. Capture **MAIN IDEAS** and **KEY DETAILS**
    
    5. Use proper citation format: "[source: [doc_name], page: [N]]"

    DO NOT:

    1. **HALLUCINATE** or add information not in documents
    
    2. Omit citations from any bullet point
    
    3. Add extra text outside the specified format
    
    4. Summarize content **OUT OF CONTEXT** of provided documents
    
    5. Include preamble or explanatory text in answer field

    ═══════════════════════════════════════════════════════════════════════════════
    FALLBACK RESPONSE
    ═══════════════════════════════════════════════════════════════════════════════

    If the query is **OUT OF CONTEXT** or you **CANNOT** generate summary from the provided documents:

    {{
        "answer": "Couldn't generate summary from the provided docs..."
    }}

    IMPORTANT: Use this EXACT fallback text when:
    - Query is unrelated to provided documents
    - Context is insufficient for summarization
    - Documents don't contain relevant information

    ═══════════════════════════════════════════════════════════════════════════════
    EXAMPLE OUTPUTS
    ═══════════════════════════════════════════════════════════════════════════════

    **Example 1:** Valid summary with multiple points

    Query: "Explain the points to remember in short."

    Response:
    {{
        "answer": "Answer: - Sun is essential for life on Earth. [source: 238237_sdq_sun.pdf, page: 2] \\n - It provides heat and light energy to the planet. [source: qqiuqdu_2381_sun.pdf, page: 3] \\n - Photosynthesis in plants depends on solar radiation. [source: 238237_sdq_sun.pdf, page: 5] \\n - The sun is approximately 93 million miles away from Earth. [source: qqiuqdu_2381_sun.pdf, page: 1]"
    }}

    **Example 2:** Summary from single document

    Query: "Summarize the key findings."

    Response:
    {{
        "answer": "Answer: - The study found a 25 per cent increase in productivity. [source: Research_Report_2024.pdf, page: 12] \\n - Employee satisfaction scores improved by 15 points. [source: Research_Report_2024.pdf, page: 18] \\n - Remote work adoption reached 60% of the workforce. [source: Research_Report_2024.pdf, page: 22]"
    }}

    **Example 3:** Summary from multiple documents

    Query: "What are the main policy changes?"

    Response:
    {{
        "answer": "Answer: - New remote work policy allows up to 3 days per week. [source: HR_Policy_2024.pdf, page: 4] \\n - Annual leave increased from 15 to 20 days. [source: Benefits_Guide.pdf, page: 7] \\n - Health insurance now covers mental health services. [source: Benefits_Guide.pdf, page: 12] \\n - Performance reviews shifted to quarterly basis. [source: HR_Policy_2024.pdf, page: 9]"
    }}

    **Example 4:** Out of context query

    Query: "What is the weather forecast for next week?"

    Response:
    {{
        "answer": "Couldn't generate summary from the provided docs..."
    }}

    **Example 5:** Insufficient context

    Query: "Summarize the investment strategy."

    Response:
    {{
        "answer": "Couldn't generate summary from the provided docs..."
    }}

    ═══════════════════════════════════════════════════════════════════════════════
    INPUT VARIABLES
    ═══════════════════════════════════════════════════════════════════════════════

    **CONTEXT:**
    {context}

    **USER QUERY:**
    {query}
    """
    llm=get_groq_llm(temperature=0.0, model_kwargs=SummarizationSchema)
    # prepare context
    context="\n".join(f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Page Content: {d.page_content}\n" for d in state["documents"])
    #invoke llm
    try:
        response=llm.invoke(SUMMARY_PROMPT.format(context=context, query=state["query"]))
        data = json.loads(response.content)
        return {
            "answer": data["answer"],
        }
    except ValidationError as e:
        print(f"Resposne validation failed: {e}")
    except Exception as e:
        print(f"Summarization Error: {str(e)}")
        return {"answer": "I encountered an error processing the documents."}   