#🏡 Real Estate AI Assistant

This project is an AI-powered tool that analyzes real estate news articles and answers user queries using Retrieval-Augmented Generation (RAG).

## 🚀 Features
- Input multiple article URLs
- Extract and process content
- Store embeddings using ChromaDB
- Ask questions and get contextual answers

## 🧠 Tech Stack
- Python
- LangChain
- Streamlit
- ChromaDB
- HuggingFace Embeddings
- Groq (LLaMA 3)

## ⚙️ How it Works
1. Load data from URLs
2. Split into chunks
3. Convert to embeddings
4. Store in vector DB
5. Retrieve relevant chunks
6. Generate answer using LLM

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run main.py
