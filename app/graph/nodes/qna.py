from app.llms.groq import get_groq_llm
from app.schemas.schemas import QnAResponse
from pydantic import ValidationError
import json

def qna_node(state):
    """
    QnA Node for the LangGraph. This node takes the retrieved documents from the vector database and answers the user query based on the retrieved context. 
    It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference. The node then defines a QnA prompt that instructs the LLM to answer the user query using only the information from the retrieved context, and to provide citations for each piece of information used in the answer. The LLM is invoked with this prompt, and the generated answer is returned in the state for downstream processing or response generation.
    """
    QnA_PROMPT = """
    
    # SYSTEM ROLE:
    You are a **precise Q&A assistant**.

    # CORE TASK

    **ANSWER** user questions using **ONLY** the context provided below as reference.

    # OUTPUT FORMAT REQUIREMENTS

    CRITICAL: Pydantic-Compatible JSON

    You **MUST** return a VALID JSON OBJECT that STRICTLY survives Pydantic model validation.

    ## Required JSON Schema

    {{
        "answer": "[string]"
    }}

    ## Field Specifications

    **"answer"** (REQUIRED - string)

    - Format: "Answer: [your_answer] \\n Citations: [citation_list]"

    - Keep answer to **3-5 sentences MAXIMUM**

    - NO additional text before or after the JSON

    - The answer field should contain ONLY the actual answer and citations

    ## Answer & Citation Format Rules

    **Structure:**

    "Answer: [your_response_text] \\n Citations: [source: [doc_name], page: [N]], [source: [doc_name], page: [N]]"

    **Components:**

    1. **Answer Section**
    - Start with "Answer: "
    - Provide direct, concise response
    - Maximum 3-5 sentences
    - Based ONLY on provided context

    2. **Citations Section**
    - Start with "\\n Citations: "
    - Format EACH citation as: "[source: [doc_name], page: [N]]"
    - Multiple citations separated by commas
    - Include ALL relevant sources

    # FALLBACK RESPONSE

    If you **CANNOT** answer from the provided context:

    {{
        "answer": "I don't know. Couldn't figure it out from the context."
    }}

    ## IMPORTANT: Use this EXACT fallback text when:
    - Context is insufficient
    - Information is not present in documents
    - Query is out of scope

    # EXAMPLE OUTPUTS

    **Example 1:** Valid answer with citations

    Query: "What does the policy say?"

    Response:
    {{
        "answer": "Answer: The policy states that she can receive the monetary benefit within 30 business days of approval. The eligible amount is based on the submitted documentation and is capped at $10,000 annually. \\n Citations: [source: Doc1.pdf, page: 2], [source: Doc2.pdf, page: 5]"
    }}

    **Example 2:** Valid answer with single citation

    Query: "What is the refund timeline?"

    Response:
    {{
        "answer": "Answer: The refund will be processed within 14 business days from the date of request submission. \\n Citations: [source: Policy_Document.pdf, page: 8]"
    }}

    **Example 3:** Multiple citations from same document

    Query: "What are the eligibility criteria?"

    Response:
    {{
        "answer": "Answer: Applicants must be full-time employees with at least 6 months of tenure. They must submit form A-12 along with supporting documents. Additionally, prior approval from the department head is mandatory. \\n Citations: [source: HR_Manual.pdf, page: 3], [source: HR_Manual.pdf, page: 7], [source: HR_Manual.pdf, page: 12]"
    }}

    **Example 4:** Insufficient context

    Query: "What is the company's stock price?"

    Response:
    {{
        "answer": "I don't know. Couldn't figure it out from the context."
    }}

    # INPUT VARIABLES

    **CONTEXT:**
    {context}

    **USER QUERY:**
    {query}
    """
    llm=get_groq_llm(temperature=0.0, model_kwargs={"response_format": {"type": "json_object"}})
    
    context = "\n".join(
        f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Content: {d.page_content}\n" 
        for d in state["documents"]
    )
    
    try:
        response=llm.invoke(QnA_PROMPT.format(context=context, query=state["query"]))
        data = json.loads(response.content)
        validated = QnAResponse(**data)
        return {
            "answer": validated.answer
        }
    except ValidationError as e:
        print(f"Resposne validation failed: {e}")
    except Exception as e:
        raise e
        print(f"QnA Error: {str(e)}")
        return {
            "answer": "I encountered an error processing the documents.",
        }