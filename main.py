from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import os
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
UPLOAD_DIR = "uploaded_pdfs"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma3:1b"
# Batasi maksimal 5 MB (dalam bytes)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Initialize
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm = Ollama(model=OLLAMA_MODEL)

# Custom Prompt
PROMPT_TEMPLATE = """
Anda adalah DocBot dari Encrypt Lab. Tugas Anda membantu pengguna memahami konten PDF.

Gunakan informasi berikut:
{context}

Aturan:
1. Jika tidak ada informasi relevan buat permintaan maaf
2. JANGAN mengarang jawaban!
3. Gunakan bahasa profesional.

Percakapan Sebelumnya:
{history}

Pertanyaan: {question}
Jawaban: 
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT_TEMPLATE),
    ("human", "{input}"),
])
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# Load existing ChromaDB
vector_db = (
    Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    if os.path.exists(CHROMA_DIR)
    else None
)


def process_pdf(file_path: str, filename: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)

    # Tambahkan metadata ke setiap chunk
    for text in texts:
        text.metadata["source"] = filename  # Simpan nama file asli

    return texts


@app.get("/ask")
async def ask_question(request: str):
    async def generate_stream():
        try:
            retriever = vector_db.as_retriever()
            history = []
            history.append({"role": "user", "content": HumanMessage(content=request)})

            # Pastikan `context` diberikan
            relevant_docs = retriever.invoke(request)
            context_documents_str = "\n\n".join(doc.page_content for doc in relevant_docs)

            qa_prompt_local = qa_prompt.partial(history=history, context=context_documents_str, question=request)
            llm_chain = {"input": RunnablePassthrough()} | qa_prompt_local | llm

            response = llm_chain.invoke(request)  # Mendapatkan teks penuh
            for word in response.split():  # Kirim kata per kata
                yield word + " "
                await asyncio.sleep(0.05)
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_stream(), media_type="text/plain")


@app.delete("/delete-pdf/{filename}")
async def delete_pdf(filename: str):
    try:
        # Hapus dari ChromaDB berdasarkan metadata
        global vector_db
        if vector_db:
            collection = vector_db._collection
            ids_to_delete = [
                doc.id for doc in collection.get(where={"source": filename})["ids"]
            ]
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)

        # Hapus file fisik
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        return {"message": f"PDF '{filename}' dihapus!"}

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Simpan PDF
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Proses PDF dengan metadata
        texts = process_pdf(file_path, file.filename)

        global vector_db
        if vector_db is None:
            vector_db = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
                collection_metadata={"hnsw:space": "cosine"},  # Optimasi similarity
            )
        else:
            vector_db.add_documents(texts)

        return {"message": f"PDF '{file.filename}' diproses!"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
