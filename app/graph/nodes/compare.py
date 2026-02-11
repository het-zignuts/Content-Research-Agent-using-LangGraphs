from app.llms.groq import get_groq_llm

def compare_node(state):
    """
    Compare Node for the LangGraph. This node takes the grouped documents retrieved from the vector database and compares them based on the user query. It constructs a context for the LLM prompt by aggregating the content from the retrieved documents, including their names and page numbers for reference. The node then defines a comparison prompt that instructs the LLM to compare and contrast the documents in a structured tabular format, ensuring that each point of comparison is clearly identified along with its source. The LLM is invoked with this prompt, and the generated comparison is returned as the answer in the state for downstream processing or response generation.
    """

    COMPARISION_PROMPT="""
        SYSTEM INSTRUCTION:
            You are a compare assistant.
            Compare and contrast all the documents given in the context below, based on the user query.
            The comparision should be returned in a tabular format with the following columns:
            - (Mention the Point of Comparison here): The specific aspect or criteria being compared across the documents.
            - Document 1: The information or perspective from Document 1 related to the point of comparison.
            - Document 2: The information or perspective from Document 2 related to the point of comparison.
            - Document ...
            Your comparison should be comprehensive and cover all relevant aspects of the documents based on the user query.
            Also, for each point in the comparison, mention the source of the information in the retrieved context as source/document name and page number in the format [source: document_name, page: page_number].
            e.g. Cell-n should be like 'comparision details (source: document_name, page: page_number)'.
            Do not mix facts.

        CONTEXT:
        {context}

        USER QUERY:
        {query}

    """
    docs=state["grouped_docs"] # get the grouped docs from the sate
    context="" # initialize an empty string to build the context for the LLM prompt
    for doc_id, contents in docs.items():
        doc_name=contents[0]["doc_name"] # retrieve documant name
        context+=f"Document Name: {doc_name}\n"
        context+=f"Context from {doc_name}:\n"
        for content in contents:
            context+=f"- {content['content']}\n Page Number: {content['page_number']}\n" # add each content piece from the document to the context, along with its page number for reference

    llm=get_groq_llm(temperature=0.0) # initialize the GROQ LLM with a temperature of 0.0 for deterministic output.
    comparison=llm.invoke(
        COMPARISION_PROMPT.format(context=context, query=state["query"]) # Invoke the LLM with the formatted prompt, passing in the constructed context and the user query to generate the comparison output based on the provided documents and the user's request
    )
    return {"answer": comparison} # return the generated comparison as the answer in the state, which can be used by downstream nodes in the graph to provide a response to the user or for further processing.