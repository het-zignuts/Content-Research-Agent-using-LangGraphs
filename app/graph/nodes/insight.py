from app.llms.groq import get_groq_llm
from app.schemas.schemas import InsightsResponse
from pydantic import ValidationError
import json

def insight_node(state):
    """
    Insight Node for the LangGraph. This node takes the retrieved documents from the vector database and generates insights and recommendations based on the user query. It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference. 
    The node then defines an insight generation prompt that instructs the LLM to analyze the retrieved content in relation to the user query and generate relevant, concise, and clear insights and recommendations. The LLM is invoked with this prompt, and the generated insights are returned in the state for downstream processing or response generation.
    """
    INSIGHT_PROMPT="""
        ═══════════════════════════════════════════════════════════════════════════════
        SYSTEM INSTRUCTION
        ═══════════════════════════════════════════════════════════════════════════════

        You are an **insight generation assistant**.

        ═══════════════════════════════════════════════════════════════════════════════
        CORE TASK
        ═══════════════════════════════════════════════════════════════════════════════

        **ANALYZE** the retrieved content provided in the context below, based on the user query.

        **GENERATE** as requested recommendations, insights, opinions, analysis, etc.

        Your output should be:
       - **RELEVANT** to the query
       - **CONCISE** and clear

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

        - Format: "Insights and Recommendations: [your_analysis]"
        
        - Provide **concise plain-text** response
        
        - NO additional text before or after the JSON
        
        - The answer field should contain ONLY the actual answer content

        - Follow the EXACT format specified

        ═══════════════════════════════════════════════════════════════════════════════
        FALLBACK RESPONSE
        ═══════════════════════════════════════════════════════════════════════════════

        If you **CANNOT** generate insights from the provided context:

        {{
            "answer": "Couldn't provide insights from the given context..."
        }}

        ═══════════════════════════════════════════════════════════════════════════════
        EXAMPLE OUTPUTS
        ═══════════════════════════════════════════════════════════════════════════════

        **Example 1:** Valid insight generation

        Query: "What do you think should be the next step?"

        Response:
        {{
            "answer": "Insights and Recommendations: According to me, you should first communicate the details with the client, then schedule a follow-up meeting to address concerns, and finally document all decisions for future reference."
        }}

        **Example 2:** Valid opinion generation

        Query: "What's your opinion on the proposed marketing strategy?"

        Response:
        {{
            "answer": "Insights and Recommendations: Based on the document analysis, the proposed strategy appears well-targeted for the millennial demographic, though it may require additional budget allocation for social media campaigns to maximize reach."
        }}

        **Example 3:** Context insufficient

        Query: "What should be our investment strategy?"

        Response:
        {{
            "answer": "Couldn't provide insights from the given context..."
        }}

        ═══════════════════════════════════════════════════════════════════════════════
        INPUT VARIABLES
        ═══════════════════════════════════════════════════════════════════════════════

        **CONTEXT:**
        {context}

        **USER QUERY:**
        {query}
    """
    llm=get_groq_llm(temperature=0.3, model_kwargs={"response_format": {"type": "json_object"}})
    
    context = "\n".join(
        f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Content: {d.page_content}\n" 
        for d in state["documents"]
    )
    
    try:
        response=llm.invoke(INSIGHT_PROMPT.format(context=context, query=state["query"]))
        data = json.loads(response.content)
        validated = InsightsResponse(**data)
        return {
            "answer": validated.answer,
        }
    except ValidationError as e:
        print(f"Resposne validation failed: {e}")
    except Exception as e:
        print(f"Insight Error: {str(e)}")
        return {
            "answer": "I encountered an error processing the documents.",
        }