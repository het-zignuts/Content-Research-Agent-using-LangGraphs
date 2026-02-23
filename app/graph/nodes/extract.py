from app.llms.groq import get_groq_llm
from app.schemas.schemas import ExtractionResponse # Assuming this is a Pydantic model
import json

def extract_node(state):
    EXTRACTION_PROMPT = """
        You are an information extraction assistant. You extract the details asked by the user in the query, from the uploaded docs and answer them with the extracted data.
        You generate markdown report when asked.
        
        CONTEXT:
        {context}

        USER QUERY:
        {query}

        INSTRUCTIONS:
        1. Provide a concise plain-text answer in the 'answer' field.
        2. If the user requested a report, generate a markdown report in the 'report' field.
        3. If no report is requested, leave 'report' as null or an empty string.
        4. the generated response should be such that it should STRICTLY survive pydantic model validation.
        5. Strictly write 'Couldn't extract answer from the provided docs...' in case you are unable to generate 'answer', do not hallucinate or give any out of context answer.
    """

    llm=get_groq_llm(temperature=0.0).with_structured_output(ExtractionResponse)
    
    context = "\n".join(
        f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Content: {d.page_content}\n" 
        for d in state["documents"]
    )
    
    try:
        response=llm.invoke(EXTRACTION_PROMPT.format(context=context, query=state["query"]))
        return {
            "answer": response.answer,
            "report_md": response.report if response.report else None
        }
    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return {
            "answer": "I encountered an error processing the documents.",
            "report_md": None
        }
