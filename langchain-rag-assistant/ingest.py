from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# This code snippet performs the process of loading a PDF file, splitting the text into chunks, vectorizing them, and saving them to the Chroma database.

# Create a PDF loader and load the PDF file
pdf_loader = PyPDFLoader(
    "attention_all_you_need.pdf"
)  # Specify the name and path of the PDF file
documents = pdf_loader.load()

# Split the text into chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

## Show the number of chunks and the first 200 characters of each
print(f"Total number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}: {chunk.page_content[:200]}...")  # Show the first 200 characters

# Vectorize the chunks using OllamaEmbeddings and save them to the Chroma database
embeddings = OllamaEmbeddings(model="nomic-embed-text")

persist_directory = "chroma_db"
db = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory=persist_directory
)
print("Database created and chunks added successfully.")
