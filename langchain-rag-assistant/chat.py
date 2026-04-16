from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# This python script takes a question from the user, finds relevant document chunks using the Chroma database created in ingest.py, and generates an answer using these chunks.
# The ingest.py script must be run to create the database before running the chat.

# Load the Chroma database and define the embedding function using OllamaEmbeddings
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
)

# Create a retriever from the database and get relevant document chunks for the user's question
retriever = db.as_retriever()

input_text = input("Question: ")
docs = retriever.invoke(input_text)
metada_list = [doc.metadata for doc in docs]
print(f"\nFound {len(docs)} document chunks:")
for i, metadata in enumerate(metada_list):
    print(
        f"Chunk {i + 1} - Page: {metadata.get('page', 'Unknown')}, Other Info: {metadata}"
    )

# Create a context by joining the document chunks, define the appropriate chat model using this context, create a smart prompt, and generate the answer by running the model
context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

## Define the appropriate chat model (You can use any model you want, here "llama3" is used as an example)
llm = OllamaLLM(model="llama3")

"""Create a smart prompt using the generated context. The prompt should include the user's question and the document chunks. 
Instruct the model that it should answer the question using the information in the document chunks."""

prompt = f"""
Question: {input_text}

Answer the question using the following document chunks. 
If the answer is not in the document, say 'I do not have this information'.
Use the information in the document chunks when answering and specify page numbers if possible.

Document Content:
{context_text}
"""

# Run to generate the answer and print it to the screen
response = llm.invoke(prompt)
print(f"\nAnswer:\n{response}")
