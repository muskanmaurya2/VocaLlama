# main.py

# ---
# PURPOSE:
# To create a FastAPI server that allows users to upload a PDF file, 
# processes it using a RAG pipeline with LangChain and LM Studio, and 
# then answers user queries based on the content of the UPLOADED PDF.
# ---

# --- Core FastAPI & Server Imports ---
import os
import uvicorn
import logging
import shutil
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional

# --- All your LangChain imports ---
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global variable to hold the RAG chain ---
qa_chain: Optional[RetrievalQA] = None

# --- Pydantic model for the request body ---
class QueryRequest(BaseModel):
    question: str

def setup_rag_pipeline(pdf_path: str) -> RetrievalQA:
    """
    Sets up the RAG pipeline for a given PDF file.
    """
    logger.info(f"Starting RAG pipeline setup for: {pdf_path}")

    # Step 1: Set LM Studio local endpoint
    os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:4096/v1"
    os.environ["OPENAI_API_KEY"] = "lm-studio" 

    logger.info(f"Loading PDF from {pdf_path}...")
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    logger.info(f"PDF loaded. Pages: {len(docs)}")

    # Step 2: Chunking & Embedding
    logger.info("Loading HuggingFace embedding model...")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info("HuggingFaceEmbeddings model loaded.")

    logger.info("Chunking documents...")
    text_splitter = SemanticChunker(embedder)
    documents = text_splitter.split_documents(docs)
    logger.info(f"Chunking complete. Documents created: {len(documents)}")

    # Step 3: Vector DB
    logger.info("Creating FAISS vector store...")
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    logger.info("Vector store created.")

    # Step 4: Initialize the LLM (Using your specified model name)
    llm = ChatOpenAI(
        model="llama-2-7b-langchain-chat", # <-- YOUR MODEL
        temperature=0.7,
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        request_timeout=300,
        verbose=True
    )
    logger.info("ChatOpenAI LLM initialized.")

    # Step 5: Define prompts and create the chain
    prompt_template = """
    You are a helpful assistant. Use the provided context to answer the question.
    If the answer isn't in the context, state that clearly.
    Provide a well-structured answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
    
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)
    
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        verbose=True
    )

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )
    
    logger.info("RAG pipeline setup complete.")
    return qa

# --- Lifespan event handler (no-op on startup now) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server startup: Ready to accept file uploads.")
    yield
    logger.info("Server shutting down.")

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---
@app.get("/")
async def get_index():
    """Serves the main HTML page"""
    return FileResponse("index.html")

# --- 1. CRITICAL FIX: Removed 'async' ---
# This tells FastAPI to run this blocking function in a thread pool
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """
    Handles PDF upload, saves it temporarily, and builds the RAG chain.
    """
    global qa_chain
    
    if file.content_type != "application/pdf":
        raise HTTPException(400, detail="Invalid file type. Please upload a PDF.")

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File '{file.filename}' saved to '{file_path}'")
        
        # This blocking call is now safe
        qa_chain = setup_rag_pipeline(file_path)
        
        return JSONResponse(content={
            "message": f"File '{file.filename}' processed. Ready to answer questions."
        })

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(500, detail=f"Failed to process file: {e}")
    finally:
        if os.path.exists(file_path):
            pass # Keep the file for now

@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Handles the user's question against the currently loaded PDF.
    """
    if not qa_chain:
        logger.warning("Query received, but no PDF has been processed.")
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload a PDF file before asking a question."}
        )
    
    logger.info(f"Handling query: {request.question}")
    
    try:
        # --- 2. CRITICAL FIX: Use .ainvoke() for async ---
        result = await qa_chain.ainvoke({"query": request.question})
        return {"answer": result["result"]}
    
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(500, detail=f"An error occurred: {e}")

# --- Run the server ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)