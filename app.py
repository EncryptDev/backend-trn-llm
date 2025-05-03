
import asyncio
import logging
import warnings
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from TTS.api import TTS
from fastapi.responses import Response
from fastapi.responses import FileResponse
from TTS.utils.synthesizer import Synthesizer
import pyttsx3
import tempfile


# Suppress warnings
def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings("ignore")

os.environ["USERPROFILE"] = r"C:\Users\BAYUFP"
os.environ["HOME"] = r"C:\Users\BAYUFP"
os.environ["TTS_HOME"] = r"D:\Project\Python\backend-trn-llm\tts_models"

# Inisialisasi FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)   

@app.on_event("startup")
def load_model():
    global synthesizer
    synthesizer = Synthesizer(
        tts_checkpoint="checkpoint.pth",
        tts_config_path="config.json",
        use_cuda=False
    )

# Direktori penyimpanan dokumen
UPLOAD_DIR = "./context/"


# Load model embedding
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Inisialisasi atau load ChromaDB
if os.path.exists("chroma_db"):
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
else:
    vectorstore = None

# LLM dari Ollama
# llm = Ollama(model="llama3.2-vision")
llm = Ollama(model="llama3.2")

# Prompt untuk chatbot
system_prompt = """
    Kamu adalah seorang Trainer PT Surabaya Autocomp Indonesia bernama BEJO.
    Peran Anda adalah memberikan informasi perusahaan yang jelas, ringkas, dan akurat hanya berdasarkan informasi dari dokumen yang diberikan dan percakapan sebelumnya dengan pengguna.
    Anda harus selalu menanggapi sebagai trainer dan menghindari penyangkalan atas keahlian Anda.
    Jika jawabannya tidak diketahui, cukup sebutkan saja dan jangan berspekulasi.
    Kutip bagian penjelasan, tindakan, atau ketentuan yang relevan dalam tanggapan Anda.
    Catatan: Pengembang telah menyediakan dokumen legal, bukan pengguna.

    Percakapan Sebelumnya:
    {history}

    Konteks dokumen:
    {context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


class TTSRequest(BaseModel):
    text:str
    rate:int = 150
    volume: float = 1.0

@app.post('/ttsx3')
async def text_to_speech(request: TTSRequest):
    try:
        engine = pyttsx3.init()

        engine.setProperty('rate', request.rate)
        engine.setProperty('volume', request.volume)

        voices = engine.getProperty("voices")
        indonesian_voice = None

        for voice in voices:
            if "andika" in voice.name.lower():  # cari "Andika" secara spesifik
                indonesian_voice = voice.id
                break

        if indonesian_voice:
            engine.setProperty("voice", indonesian_voice)
        else:
            raise HTTPException(status_code=500, detail="Voice Bahasa Indonesia (Andika) tidak ditemukan.")

        # Simpan TTS ke file sementara
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            temp_filename = tmp_file.name

        engine.save_to_file(request.text, temp_filename)
        engine.runAndWait()

        with open(temp_filename, "rb") as audio_file:
            audio_content = audio_file.read()

        os.unlink(temp_filename)

        return Response(
            content=audio_content,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=tts_output.mp3"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")


@app.get("/generate-tts")
async def generate_tts(text: str, speaker: str = "wibowo"):
    output_path = "output.wav"
    
    # Generate audio
    wav = synthesizer.tts(text=text, speaker_name=speaker)
    synthesizer.save_wav(wav, path=output_path)
    
    return FileResponse(output_path)

# API untuk upload file
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat PDF!")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Simpan file ke direktori
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Proses PDF yang baru diunggah
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    for text in texts:
        text.metadata["source"] = file.filename

    # Inisialisasi atau update ChromaDB
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            texts, embeddings, persist_directory="chroma_db"
        )
    else:
        vectorstore.add_documents(texts)

    return JSONResponse(content={"message": "File berhasil diunggah dan diproses!"})


class DeleteRequest(BaseModel):
    filename: str


# API untuk delete
@app.delete("/delete")
async def delete_pdf(request: DeleteRequest):
    file_name = request.filename
    try:
        global vectorstore
        if vectorstore:
            # Hapus dari ChromaDB berdasarkan metadata
            collection = vectorstore._collection

            # Dapatkan semua dokumen dengan metadata source yang sesuai
            docs = collection.get(where={"source": file_name})

            if docs and docs["ids"]:
                # Hapus dokumen berdasarkan IDs
                collection.delete(ids=docs["ids"])

            # Hapus file fisik
            file_path = os.path.join(UPLOAD_DIR, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

            return {"message": f"PDF '{file_name}' berhasil dihapus!"}
        else:
            raise HTTPException(
                status_code=404, detail="Vectorstore belum diinisialisasi"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghapus: {str(e)}")


@app.get("/chat_stream/")
async def chat_stream(query: str):
    async def generate_response():
        try:
            retriever = vectorstore.as_retriever()
            history = []
            history.append({"role": "user", "content": HumanMessage(content=query)})

            # Pastikan `context` diberikan
            relevant_docs = retriever.invoke(query)
            context_documents_str = "\n\n".join(
                doc.page_content for doc in relevant_docs
            )

            qa_prompt_local = qa_prompt.partial(
                history=history, context=context_documents_str
            )
            llm_chain = {"input": RunnablePassthrough()} | qa_prompt_local | llm

            response = llm_chain.invoke(query)  # Mendapatkan teks penuh
            for word in response.split():  # Kirim kata per kata
                yield word + " "
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            logging.warning("Streaming request dibatalkan oleh client.")
            return

    return StreamingResponse(generate_response(), media_type="text/event-stream")


@app.get("/pdfs", response_model=list[str])
async def list_pdfs():
    try:
        files = os.listdir(UPLOAD_DIR)
        pdf_files = [file for file in files if file.lower().endswith(".pdf")]
        return pdf_files
    except FileNotFoundError:
        return {"error": "PDF folder not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 500


# Menjalankan server FastAPI
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
