from app.llms.groq import get_groq_llm

def compare_node(state):
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
    docs=state["grouped_docs"]
    context=""
    for doc_id, contents in docs.items():
        doc_name=contents[0]["doc_name"]
        context+=f"Document Name: {doc_name}\n"
        context+=f"Context from {doc_name}:\n"
        for content in contents:
            context+=f"- {content['content']}\n Page Number: {content['page_number']}\n"

    llm=get_groq_llm(temperature=0.0)
    comparison=llm.invoke(
        COMPARISION_PROMPT.format(context=context, query=state["query"])
    )
    return {"answer": comparison}