import os
from dotenv import load_dotenv
from uuid import uuid4
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
import requests
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from playwright_loader import load_urls


load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=500
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )
def load_urls(urls):
    docs = []
    for url in urls:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        docs.append(Document(page_content=text, metadata={"source": url}))

    return docs

def process_urls(urls):
    yield "Initializing Components"
    initialize_components()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading data...✅"
    data = load_urls(urls)

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)

    yield "Adding to vector DB...✅"
    uuids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done ✅"

prompt = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""
Answer using the provided context.

Extract relevant information clearly.
If exact numbers are not available, explain based on context.
DO NOT say "I don't know".

Context:
{summaries}

Question:
{question}

Answer:
"""
)

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector DB not initialized")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    result = chain.invoke({"query": query})

    return result["result"], "Sources not tracked"