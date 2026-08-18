from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

def rag_chatbot() :

    loaders = DirectoryLoader(
        path = "./papers",
        glob = "**/*.pdf",
        loader_cls = UnstructuredFileLoader,
        show_progress = True,
        use_multithreading = True 
    )

    docs = loaders.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    text_splitter = SemanticChunker(
        embeddings = embeddings,
        breakpoint_threshold_amount = 0.85,
    )

    splits = text_splitter.split_documents(docs)
 
    vectorstore = FAISS.from_documents(
        documents = splits,
        embedding = embeddings,
        distance_strategy = DistanceStrategy.COSINE
    )

    retriever = vectorstore.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs = {"k": 5, "score_threshold": 0.2}
    )

    template = (
        "You are a strict, citation-focused assistant for a private knowledge base.\n"
        "RULES:\n"
        "1) Use ONLY the provided context to answer.\n"
        "2) If the answer is not clearly contained in the context, say: "
        "\"I don't know based on the provided documents.\"\n"
        "3) Do NOT use outside knowledge, guessing, or web information.\n"
        "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(template)


    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    while True: 
        user_input = input("Question: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chatbot. Goodbye!")
            break

        answer = rag_chain.invoke(user_input) 

        print(answer)


if __name__ == "__main__":
    rag_chatbot() 
