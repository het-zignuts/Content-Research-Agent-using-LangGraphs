from app.llms.groq import get_groq_llm
from app.schemas.schemas import SummarizationResponse 

def summarize_node(state):
    """
    Summarization Node for the LangGraph. This node takes the retrieved documents from the vector database and generates a concise summary based on the user query. 
    It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference.
    """
    SUMMARY_PROMPT="""
        SYSTEM INSTRUCTION:
        1. You are a summarization assistant. 
        2. Summarize the retrieved content provided as context below, into concise bullet points that capture the main ideas and key details. 
        3. Focus on providing a clear and informative summary of the retrieved documents (context) given below.
        4. Also, for each bullet point in the summary, mention the source of the information in the retrieved context as source/document name and page number in the format [source: document_name, page: page_number].
        5. If case the query is out of context of the provided documents or if you are not able to generate summary points from the provided documents, strictly write "Couldn't generate summary from the provided docs..." as the answer. 
           Do not hallucinate or give any out of context answer.
        6. The generated answer should STRICTLY survive pydantic model validation.

        ANSWER FORMAT:
        Your answer in the answer field of response should be in the following format, if the summary points can be generated:
        - summary point 1 [source: document_name, page: page_number]
        - summary point 2 [source: document_name, page: page_number]
        ...
        
        No other text other than the summary points along with the error fallback message mentioned in point 5 should be there in the answer field of the response. Do not write any other text apart from that in the answer field.
        
        CONTEXT:
        {context}

        USER QUERY:
        {query}
    """
    llm=get_groq_llm(temperature=0.0).with_structured_output(SummarizationResponse)
    # prepare context
    context="\n".join(f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Page Content: {d.page_content}\n" for d in state["documents"])
    #invoke llm
    try:
        summary=llm.invoke(
            SUMMARY_PROMPT.format(context=context, query=state["query"])
        )
        return {"answer": summary.answer} # add the summary to the state.
    except Exception as e:
        print(f"Summarization Error: {str(e)}")
        return {"answer": "I encountered an error processing the documents."}       
    # summary=llm.invoke(
    #     SUMMARY_PROMPT.format(context=context, query=state["query"])
    # )
    # return {"answer": summary.answer} # add the summary to the state.