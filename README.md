# 📚 AI PDF Document Q&A — Setup & Run Guide

This project is an **AI-powered PDF Question Answering System** built using:

- **Groq**
- **LangChain**
- **Streamlit**
- **FAISS Vector Database**
- **HuggingFace Embeddings**

The application allows users to upload PDF documents, create embeddings, and ask questions based on document content using **RAG (Retrieval-Augmented Generation)**.

---

## 🚀 Features

- ✅ PDF Document Loading  
- ✅ Semantic Search using FAISS  
- ✅ AI Question Answering  
- ✅ Groq LLM Integration  
- ✅ Streamlit Web Interface  
- ✅ HuggingFace Embeddings  
- ✅ Fast RAG Pipeline  

---

## 📁 Project Structure

```bash
Groq end-to-end/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
│
├── your_data/
│   ├── sample1.pdf
│   ├── sample2.pdf
│
└── venv/

⚙️ Prerequisites
Install the following before running the project:
Python 3.10 or above
Git
Internet connection


1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name


2️⃣ Create Virtual Environment
Windows
python -m venv venv
Linux / Mac
python3 -m venv venv


3️⃣ Activate Virtual Environment
Windows PowerShell
.\venv\Scripts\activate
Windows CMD
venv\Scripts\activate
Linux / Mac
source venv/bin/activate

After activation, the terminal should show:

(venv)


4️⃣ Install Dependencies
Install all required packages:
pip install -r requirements.txt


5️⃣ Setup API Keys
Create a .env file in the project root directory and add your API keys:
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
🔑 Get API Keys
Groq API Key
Create an account and generate a key from:
Groq Console
Google API Key
Generate a Gemini API key from:
Google AI Studio



6️⃣ Add PDF Files
Place your PDF files inside:
your_data/
Example
your_data/
├── document1.pdf
├── document2.pdf


7️⃣ Run the Application
Start the Streamlit app:
streamlit run app.py


8️⃣ Open in Browser
After running, Streamlit will provide URLs like:
Local URL: http://localhost:8501
Open it in your browser.



🖥️ How to Use
Step 1
Click:
📄 Create Vector DB
This will:
Load PDFs
Split documents
Create embeddings
Store vectors in FAISS


Step 2
Enter your question in:
💬 Ask a Question From Your Documents
Example
What is the population growth mentioned in the report?


Step 3
View:
AI-generated answer
Retrieved document chunks
Response time


📦 Requirements Example
requirements.txt
streamlit
langchain
langchain-community
langchain-core
langchain-groq
langchain-huggingface
langchain-text-splitters
faiss-cpu
sentence-transformers
pypdf
python-dotenv
huggingface-hub


📄 .gitignore
.env
venv/


🛠️ Common Errors & Fixes

1. ModuleNotFoundError
Run: pip install -r requirements.txt

2. Streamlit Not Found
Install Streamlit:
pip install streamlit

3. Groq Model Decommissioned
Replace old model:
model_name="Llama3-8b-8192"
with:
model_name="llama-3.1-8b-instant"

4. PDF Folder Not Found
Ensure:
./your_data
exists and contains PDFs.


🧠 Technologies Used
Technology	Purpose
Python	Backend
Streamlit	Frontend UI
LangChain	RAG Pipeline
FAISS	Vector Database
HuggingFace	Embeddings
Groq	LLM Inference


👨‍💻 Author Developed by Siddhesh.
⭐ If You Like This Project  Star the repository on GitHub and contribute improvements




