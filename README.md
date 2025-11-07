# 🦙 VocaLlama

🎙️ **Talk to Your PDFs with a Local-First Voice AI**



## 📘 Project Overview

**VocaLlama** is a full-stack, locally-hosted RAG (Retrieval-Augmented Generation) application. It empowers you to **upload any PDF document** and have a real-time, **voice-based conversation** with it.

It's built to be 100% private, running the **`llama-2-7b-langchain-chat`** model (or any compatible model) locally via **LM Studio**. The system uses a **FastAPI** backend to handle document processing with **LangChain** and a simple **HTML/JavaScript** frontend to provide a clean, interactive chat and voice experience.

## 🧠 Key Features

| Feature | Description |
| :--- | :--- |
| **📁 Dynamic PDF Upload** | **Upload any PDF** document directly through the web interface for on-the-fly processing. |
| **🗣️ Voice-Enabled RAG** | Uses browser-native Speech-to-Text (STT) and Text-to-Speech (TTS) for a hands-free conversational experience. |
| **🧠 Llama 2 Powered** | Processes queries using the `llama-2-7b-langchain-chat` model hosted locally via LM Studio. |
| **⚙️ FastAPI Backend** | A robust `main.py` server manages file uploads, the LangChain RAG pipeline, and query handling. |
| **🌐 Modern Web UI** | A clean `index.html` frontend using modern CSS and JavaScript to manage file uploads, state, and voice interaction. |
| **🔒 100% Offline & Private**| No cloud APIs, no data collection. Your documents and conversations never leave your machine. |

## 🧩 How It Works

1.  **🚀 Get Started:** The user launches the `main.py` script and the `LM Studio` server (with Llama 2 loaded).
2.  **📤 Upload Doc:** The user opens `index.html` in their browser and **uploads any PDF**.
3.  **🧠 Process Doc:** The FastAPI backend (`/upload`) receives the PDF and:
    * Loads the document using `PDFPlumberLoader`.
    * Splits the text into smart chunks using `SemanticChunker`.
    * Generates embeddings using `HuggingFaceEmbeddings`.
    * Creates a `FAISS` vector store in memory for the `RetrievalQA` chain.
4.  **🎙️ Voice Query:** The user speaks a question. The browser's `SpeechRecognition` API converts **voice → text**.
5.  **🔍 Retrieve Info:** The text query is sent to the `/query` endpoint. LangChain's `RetrievalQA` chain:
    * Finds the most relevant chunks from the FAISS vector store.
    * Stuffs these chunks into a prompt for the LLM.
6.  **💬 AI Generates:** The prompt is sent to the local **Llama 2** model via LM Studio, which generates a factual answer based *only* on the PDF content.
7.  **🔊 Spoken Response:** The answer text is sent back to the browser, displayed, and spoken aloud using the `SpeechSynthesis` API.

## 🚀 Installation & How to Run

Follow these steps to get VocaLlama running on your local machine.

### **Prerequisites**

* [Python 3.9+](https://www.python.org/downloads/)
* [LM Studio](https://lmstudio.ai/)

### **Step 1: Set Up LM Studio (The LLM Server)**

This will be your "engine." It needs to be running in the background.

1.  **Download & Install LM Studio:** Get it from [lmstudio.ai](https://lmstudio.ai/).
2.  **Download Your Model:**
    * Open LM Studio.
    * In the **"Search"** tab (🔍), type `llama-2-7b-langchain-chat`.
    * Find a GGUF version (like one from TheBloke) and click **Download**.
3.  **Start the Local Server:**
    * Go to the **"Local Server"** tab (Looks like `<->`).
    * At the top, select the `llama-2-7b-langchain-chat` model you just downloaded.
    * **Important:** In the "Server Settings" on the right, change the **"Port"** to `4096` to match your `main.py` code.
    * Click **"Start Server"**.

Leave this server running in the background.

### **Step 2: Set Up VocaLlama (The Web App)**

This is the code from your repository.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/muskanmaurya2/VocaLlama.git](https://github.com/muskanmaurya2/VocaLlama.git)
    cd VocaLlama
    ```

2.  **Create a Virtual Environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    * This will install FastAPI, LangChain, FAISS, and everything from your `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### **Step 3: Run the Application!**

You need two things running: your server (from Step 1) and your app (this step).

1.  **Start the FastAPI Server:**
    * **(Make sure you've added the `if __name__ == "__main__":` block to `main.py`!)**
    * In your terminal (where you activated the `venv`), run:
    ```bash
    python main.py
    ```
    * You should see a message like `Uvicorn running on http://127.0.0.1:8000`.

2.  **Open the Frontend:**
    * Simply open the `index.html` file in your web browser.
    * You can just double-click the file, or right-click and "Open with..." your browser.

You are now ready to upload a PDF and start chatting!

## 💻 Tech Stack

| Layer | Technology Used |
| :--- | :--- |
| **🧠 LLM** | `llama-2-7b-langchain-chat` (or any model in LM Studio) |
| **⚙️ Backend** | FastAPI, Uvicorn |
| **🌐 Frontend** | HTML5, CSS3, JavaScript |
| **🗣️ Voice (Browser)** | Web Speech API (`SpeechRecognition`, `SpeechSynthesis`) |
| **🐍 Orchestration** | LangChain |
| **📄 PDF Loading** | `langchain_community.document_loaders.PDFPlumberLoader` |
| **✂️ Text Splitting** | `langchain_experimental.text_splitter.SemanticChunker` |
| **🧩 Embeddings** | `langchain_huggingface.HuggingFaceEmbeddings` |
| **💾 Vector Store** | `langchain_community.vectorstores.FAISS` (using `faiss-cpu`) |
| **📦 Python Env** | `requirements.txt` |

## 🚀 Applications

* **📚 Study Assistant:** Upload a textbook chapter and ask it questions.
* **📄 Research Analysis:** Quickly query dense academic papers or technical manuals.
* **⚖️ Document Review:** Analyze legal contracts, financial reports, or policy documents.
* **♿ Accessibility Tool:** Provides a voice-first way for users to interact with written content.

---
🧑‍💻 **Developed By- Muskan Maurya**
