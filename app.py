import gradio as gr
import os
import time
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. SET UP THE ENVIRONMENT ---
print("--- [1/6] Setting up environment ---")
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:4096/v1"
os.environ["OPENAI_API_KEY"] = "lm-studio"  # Any string works

# --- 2. LOAD AND PROCESS THE PDF (RUNS ONCE AT STARTUP) ---
print("--- [2/6] Loading PDFPlumberLoader... ---")
start_time = time.time()
loader = PDFPlumberLoader("Basic_Home_Remedies.pdf")
docs = loader.load()
print(f"--- [3/6] PDF loaded ({len(docs)} pages). Starting semantic chunking... ---")

# Chunking
text_splitter = SemanticChunker(HuggingFaceEmbeddings())
documents = text_splitter.split_documents(docs)
print(f"--- [4.1/6] Text split into {len(documents)} chunks. ---")

# Vector DB
print("--- [4.2/6] Creating FAISS vector store (this may take a moment)... ---")
embedder = HuggingFaceEmbeddings()
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 2})

end_time = time.time()
print(f"--- [4/6] PDF processing complete. ({end_time - start_time:.2f} seconds) ---")


# --- 5. INITIALIZE THE LLM AND QA CHAIN (RUNS ONCE AT STARTUP) ---
print("--- [5/6] Initializing LLM and RetrievalQA chain... ---")
# Initialize the LLM
llm = ChatOpenAI(
    model="llama-2-7b-langchain-chat",  # Matches your LM Studio
    temperature=0.7,
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    request_timeout=300,  # 5-minute timeout
    verbose=False
)

# Define the prompt
prompt_template = """
You are a domain expert assistant.
Use the provided context to answer the question clearly and accurately.
If the answer cannot be found in the context, say "The information is not available in the provided context."
Provide a well-structured answer in 3–4 sentences and keep it factual.

Context:
{context}

Question:
{question}

Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

# Set up the chains
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=False)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None,
)

# This is our persistent, reusable QA object
QA_CHAIN = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=retriever,
    return_source_documents=True,
    verbose=False,
)
print("--- [6/6] System is ready. Launching Gradio interface... ---")

# --- 6. DEFINE THE GRADIO QUERY FUNCTION ---
def get_answer(user_question):
    """
    This function is called every time the user clicks "Submit".
    It uses the QA_CHAIN that was already created.
    """
    if not user_question:
        return "Please ask a question."
        
    print(f"\n[Query Received]: {user_question}")
    try:
        # Use the globally created QA_CHAIN to get a result
        result = QA_CHAIN(user_question)
        answer = result["result"]
        print(f"[Answer Generated]: {answer}")
        return answer
    except Exception as e:
        print(f"!!! [ERROR]: {e}")
        return f"An error occurred: {e}. (Is your LM Studio server running with the correct model?)"


# --- 7. CREATE THE GRADIO INTERFACE ---
with gr.Blocks(theme="soft") as demo:
    gr.Markdown(
        """
        # 🌿 RAG Chatbot: Basic Home Remedies PDF
        Ask a question about the `Basic_Home_Remedies.pdf` document.
        The app will find relevant parts of the PDF and use LM Studio to generate an answer.
        """
    )
    with gr.Row():
        question_box = gr.Textbox(
            label="Your Question", 
            placeholder="e.g., What are some healthy lifestyle tips?",
            lines=2
        )
    with gr.Row():
        submit_btn = gr.Button("Submit Query")
    with gr.Row():
        answer_box = gr.Textbox(
            label="Answer from PDF", 
            interactive=False, 
            lines=8
        )

    # Wire up the button
    submit_btn.click(
        fn=get_answer,
        inputs=question_box,
        outputs=answer_box
    )

# Launch the app
print("\nGradio app is launching. Open the local URL in your browser.")
demo.launch(share=True)