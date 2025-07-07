# InternGenie  

InternGenie is an intelligent assistant that helps students prepare for internships and job interviews by providing insights from real interview experiences, structured Q&A, and tailored advice—powered by RAG (Retrieval-Augmented Generation) using LangChain, LLMs, and vector search.

## 1. Authors  
This project was collaboratively developed by:  
- **Bhavesh Chamaria** – [GitHub Profile](https://github.com/bhavesh0609)  
- **Kasukurthi Sujeeth** – [GitHub Profile](https://github.com/Princesujeeth7)  


---

## 2. What It Does

InternGenie intelligently analyzes resumes, JSON interview logs, and documents (PDF/DOCX) from your dataset folder to answer questions like:

- "What were the interview questions for Google SDE roles?"
- "Share the candidate experience for data analyst roles at EY."
- "Any tips for cracking intern interviews at Microsoft?"

It returns structured results including:

- Company  
- Role  
- Interview Questions  
- Experience Summary  
- Tips  

---

## 3. Features

-  Smart Retrieval with Chroma DB and HuggingFace embeddings
-  PDF/DOCX/JSON ingestion using LangChain + Unstructured
-  RAG pipeline using Ollama LLaMA3 model
-  Structured, formatted answers for career help
-  Streamlit UI for interactive querying

---

## 4. Tech Stack

| Layer       | Tooling                             |
|------------|--------------------------------------|
| LLM        | Ollama (LLaMA3)                      |
| Embedding  | HuggingFace (`intfloat/e5-small-v2`) |
| RAG        | LangChain                            |
| Retrieval  | ChromaDB                             |
| Frontend   | Streamlit                            |
| File I/O   | `pymupdf`, `unstructured`, `docx`    |

---

## 5. 🗂️Project Structure

```bash
.
├── app.py              # Main Streamlit app
├── a.ipynb             # Experimental/Supporting notebook
├── req.txt             # Required dependencies
└── data/               # Folder for input documents (PDFs, DOCX, JSON)
```

