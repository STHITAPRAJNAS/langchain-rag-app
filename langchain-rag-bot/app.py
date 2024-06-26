import streamlit as st
from pathlib import Path
import asyncio
import os

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

# Init Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR = BASE_DIR / 'db'
DB_DIR.mkdir(parents=True, exist_ok=True)

def persist_file(upload):
    """
    persist_file Will persist the provided file to the file system

    Args:
        upload (UploadedFile): The file uploaded via the UI
    """
    save_path = Path(DATA_DIR, upload.name)
    with open(save_path, mode="wb") as w:
        w.write(upload.getvalue())

    if save_path.exists():
        st.success(f"File {upload.name} is successfully saved")

def init_ui():
    st.set_page_config(page_title="Langchain RAG Bot", layout="wide")
    st.title("Langchain RAG Bot")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I'm here to help. Ask me anything!")
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

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
            st.session_state.vector_store = asyncio.run(init_vector_store())

    # Initialize the chat interface regardless of vector store status
    init_chat_interface()

    # If vector store is not initialized, show a message
    if st.session_state.vector_store is None:
        st.info("Please upload documents to enable context-aware responses.")


async def init_vector_store():
    """
    Initializes and returns ChromaDB vector store from document chunks
    """
    embedding_function = OllamaEmbeddings(model='mxbai-embed-large')

    # Check if the vector store already exists
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

    # Get existing document IDs from the vector store
    existing_docs = set(vector_store.get()["ids"])

    # Process new files
    files = [f for f in DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']
    new_files = [f for f in files if f.name not in existing_docs]

    if new_files:
        for file in new_files:
            loader = PyPDFLoader(str(file))
            document = await asyncio.to_thread(loader.load_and_split)
            text_splitter = RecursiveCharacterTextSplitter()
            document_chunks = await asyncio.to_thread(text_splitter.split_documents, document)
            await vector_store.aadd_documents(document_chunks)

        vector_store.persist()
        st.success(f"Added {len(new_files)} new file(s) to the vector store.")
    else:
        st.info("No new files to add. Using existing vector store.")

    return vector_store

def get_related_context(vector_store: Chroma) -> RetrieverOutputLike:
    """
    Will retrieve the relevant context based on the user's query 
    using Approximate Nearest Neighbor search (ANN)

    Args:
        vector_store (Chroma): The initialized vector store with context

    Returns:
        RetrieverOutputLike: The chain component to be used with the LLM
    """

    # Specify the model to use
    llm = Ollama(model="llama3")

    # Here we are using the vector store as the source
    retriever = vector_store.as_retriever()

    # Create a prompt that will be used to query the vector store for related content
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])

    # Create the chain element which will fetch the relevant content from ChromaDB
    chain_element = create_history_aware_retriever(llm, retriever, prompt)
    return chain_element


def get_context_aware_prompt(context_chain: RetrieverOutputLike) -> Runnable:
    """
    Combined the chain element to fetch content with one that then creates the 
    prompt used to interact with the LLM

    Args:
        context_chain (RetrieverOutputLike): The retriever chain that can 
            fetch related content from ChromaDB

    Returns:
        Runnable: The full runnable chain that can be executed
    """

    # Specify the model to use
    llm = Ollama(model="llama3")

    # A standard prompt template which combined chat history with user query
    # NOTE: You must pass the context into the system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer the users questions. Use provided context to answer the question as accurately as possible:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # This method creates a chain for passing documents to a LLM
    docs_chain = create_stuff_documents_chain(llm, prompt)

    # Now we merge the context chain & docs chain to form the full prompt
    rag_chain = create_retrieval_chain(context_chain, docs_chain)
    return rag_chain

from langchain_community.llms.ollama import Ollama

def get_response(user_query: str) -> str:
    try:
        if st.session_state.vector_store is None:
            llm = Ollama(model="llama3")
            response = llm(user_query)
            return response

        context_chain = get_related_context(st.session_state.vector_store)
        rag_chain = get_context_aware_prompt(context_chain)

        res = rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })
        return res["answer"]
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request. Please try again later."


def init_chat_interface():
    """
    Initializes a chat interface which will leverage our rag chain & a local LLM
    to answer questions about the context provided
    """

    user_query = st.chat_input("Ask a question....")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        
        # Add the current chat to the chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Print the chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        

if __name__ == "__main__":
    init_ui()
