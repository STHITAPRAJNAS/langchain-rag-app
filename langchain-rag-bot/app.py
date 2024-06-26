import streamlit as st
from pathlib import Path
import asyncio
import os
from typing import List

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.retrievers import RetrieverOutputLike
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document

# Init Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR = BASE_DIR / 'db'
DB_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def init_vector_store():
    """
    Initializes and returns ChromaDB vector store from existing database
    """
    embedding_function = OllamaEmbeddings(model='mxbai-embed-large')

    if os.path.exists(str(DB_DIR)) and os.listdir(str(DB_DIR)):
        st.info("Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=embedding_function,
            collection_name="pdf_v_db"
        )
    else:
        st.info("Creating new vector store...")
        vector_store = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=embedding_function,
            collection_name="pdf_v_db"
        )

    return vector_store

def get_existing_doc_ids(vector_store: Chroma) -> set:
    return set(vector_store.get()["ids"])

async def process_pdf(file_path: Path) -> List[Document]:
    loader = PyPDFLoader(str(file_path))
    document = await asyncio.to_thread(loader.load)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return await asyncio.to_thread(text_splitter.split_documents, document)

async def process_new_files(vector_store: Chroma):
    """
    Process new files and add them to the vector store
    """
    existing_docs = get_existing_doc_ids(vector_store)
    files = [f for f in DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']
    new_files = [f for f in files if f.name not in existing_docs]

    if new_files:
        for file in new_files:
            document_chunks = await process_pdf(file)
            vector_store.add_documents(document_chunks)

        vector_store.persist()
        st.success(f"Added {len(new_files)} new file(s) to the vector store.")
    else:
        st.info("No new files to add.")

def persist_file(upload):
    save_path = Path(DATA_DIR, upload.name)
    with open(save_path, mode="wb") as w:
        w.write(upload.getvalue())
    if save_path.exists():
        st.success(f"File {upload.name} is successfully saved")

def init_ui():
    st.set_page_config(page_title="Langchain RAG Bot", layout="wide")
    st.title("Langchain RAG Bot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I'm here to help. Ask me anything!")
        ]

    vector_store = init_vector_store()

    with st.sidebar:
        st.header("Document Capture")
        st.write("Please select docs to use as context")
        st.markdown("**Please fill the below form:**")
        with st.form(key="Form", clear_on_submit=True):
            uploads = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
            submit = st.form_submit_button(label="Upload")

        if submit and uploads:
            for upload in uploads:
                persist_file(upload)
            asyncio.run(process_new_files(vector_store))

    init_chat_interface(vector_store)

def get_related_context(vector_store: Chroma) -> RetrieverOutputLike:
    llm = Ollama(model="llama3")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    chain_element = create_history_aware_retriever(llm, retriever, prompt)
    return chain_element

def get_context_aware_prompt(context_chain: RetrieverOutputLike) -> Runnable:
    llm = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer the user's questions. Use the provided context to answer the question as accurately as possible. If the answer is not in the context, say so and provide a general response based on your knowledge:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(context_chain, docs_chain)
    return rag_chain

def get_response(user_query: str, vector_store: Chroma) -> str:
    try:
        context_chain = get_related_context(vector_store)
        rag_chain = get_context_aware_prompt(context_chain)
        res = rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })
        return res["answer"]
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request. Please try again later."

def init_chat_interface(vector_store: Chroma):
    user_query = st.chat_input("Ask a question...")
    if user_query is not None and user_query != "":
        response = get_response(user_query, vector_store)

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

if __name__ == "__main__":
    init_ui()