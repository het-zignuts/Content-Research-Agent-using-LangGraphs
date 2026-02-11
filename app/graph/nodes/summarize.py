from app.llms.groq import get_groq_llm

def summarize_node(state):
    """
    Summarization Node for the LangGraph. This node takes the retrieved documents from the vector database and generates a concise summary based on the user query. 
    It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference.
    """
    SUMMARY_PROMPT="""
        SYSTEM INSTRUCTION:
            You are a summarization assistant. 
            Summarize the retrieved content provided as context below, into concise bullet points that capture the main ideas and key details. 
            Focus on providing a clear and informative summary of the retrieved documents (context) given below.
            Also, for each bullet point in the summary, mention the source of the information in the retrieved context as source/document name and page number in the format [source: document_name, page: page_number].

        ANSWER FORMAT:
            Your answer should be in the following format:
            - summary point 1 [source: document_name, page: page_number]
            - summary point 2 [source: document_name, page: page_number]
            ...
        
        CONTEXT:
        {context}

        USER QUERY:
        {query}
    """
    llm=get_groq_llm(temperature=0.0) # initialize llm
    # prepare context
    context="\n".join(f"Document: {d.metadata['source']}, Page: {d.metadata['page']}\n Page Content: {d.page_content}\n" for d in state["documents"])
    #invoke llm
    summary=llm.invoke(
        SUMMARY_PROMPT.format(context=context, query=state["query"])
    )
    return {"answer": summary} # add the summary to the state.