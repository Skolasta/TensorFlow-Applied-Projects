import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# We will create a Streamlit-based RAG (Retrieval-Augmented Generation) application by combining the codes from previous ingest.py and chat.py files.
# This application creates a database by loading and chunking PDF files, and then uses this database to answer questions.

llm = OllamaLLM(model="llama3")
"""We will create a Streamlit-based RAG (Retrieval-Augmented Generation) application by combining the codes from previous ingest.py and chat.py files.
This application creates a database by loading and chunking PDF files, and then uses this database to answer questions.
"""

# Streamlit application title and description
st.title("RAG Assistant")
st.write(
    "This app processes uploaded PDF files to create a searchable database and then answers questions based on that content."
)

"""Define a function to load the database and optimize it with Streamlit's cache mechanism
so the database is loaded only once and accessed quickly in subsequent operations."""


@st.cache_resource
def load_db():
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )


# Create a section in the sidebar for users to upload a PDF file and save the uploaded file temporarily.
with st.sidebar:
    st.title("Document Management")
    uploaded_file = st.file_uploader("Upload a new PDF", type="pdf")

    if uploaded_file and st.button("Process into Database"):
        with st.status("Processing...", expanded=True) as status:
            # Save the temporary file
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Run ingestion logic (from ingest.py)
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = text_splitter.split_documents(docs)

            # Add to existing DB
            db = load_db()
            db.add_documents(chunks)

            # Cleanup
            os.remove(temp_path)
            st.cache_resource.clear()  # Clear cache to recognize the new file
            status.update(label="✅ Successfully added!", state="complete")

# Create a list for message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Question: ")

# Send a prompt to the LLM to determine whether the incoming question requires technical document information (PDF content) or is a general chat/greeting.
general_rag = f"""
Question: {prompt}
Task: Does this question require technical document knowledge (PDF content) or is it a general chat/greeting?
Answer format: Just write 'RAG' or 'CHAT'.
"""
response_rag = llm.invoke(general_rag)

if response_rag.strip().upper() == "RAG":
    # Display and save user message
    st.chat_message("user").markdown(prompt)

    # RAG Operations (Retrieval)
    db = load_db()
    retriever = db.as_retriever()
    docs = retriever.invoke(prompt)

    """
    The prompt below creates a context containing the user's question and document chunks. 
    Llama will answer in English by default, but the prompt species it to answer in English normally. 
    The model will try to answer the question using the information in the document chunks, and if the answer is not in the content, it will say 'This information is not found in the PDF file'.
    """

    # Generation
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    full_prompt = f"""Question: {prompt}\n\nContext:\n{context_text}\n\n
    Answer the question based on the 'Context' content. If the answer is not present in the content, say 'This information is not found in the PDF file'. Please mention the page numbers when possible.
    """

    with st.chat_message("assistant"):
        # Output the answer

        response = llm.invoke(full_prompt)
        st.markdown(response)

        # Show Metadata
        with st.expander("Sources"):
            for doc in docs:
                st.write(f"Page: {doc.metadata.get('page', 'N/A')}")

    # Add to history
    st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # CHAT Operations (Q&A)
    st.chat_message("user").markdown(prompt)

    response = llm.invoke(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
