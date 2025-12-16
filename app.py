import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq  # GANTI: Pakai Groq (Gratis)
from langchain_huggingface import HuggingFaceEmbeddings # GANTI: Pakai HF (Gratis)
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Virtual STAR Interview Coach (Free Version)", page_icon="🚀")

st.title("🚀 Virtual Interview Coach (Llama-3 Version)")
st.caption("Powered by Groq (Fast Inference) & HuggingFace Embeddings")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    # Link untuk user mengambil API Key gratis
    st.markdown("[Dapatkan API Key Gratis di sini](https://console.groq.com/keys)")
    api_key = st.text_input("Masukkan Groq API Key (gsk_...)", type="password")
    
    st.divider()
    st.subheader("📚 Knowledge Base")
    uploaded_file = st.file_uploader("Upload Panduan Wawancara (PDF)", type="pdf")
    
    process_button = st.button("Proses Dokumen")

# --- FUNGSI PROSES DOKUMEN ---
def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # 1. Load PDF
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)

    # 3. Embedding (GRATIS via HuggingFace - Jalan di CPU Laptop)
    # Model ini kecil & cepat (all-MiniLM-L6-v2)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Simpan ke Vector DB
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    os.remove(tmp_file_path)
    return vectorstore.as_retriever()

# --- STATE MANAGEMENT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if process_button and uploaded_file and api_key:
    with st.spinner("Sedang memproses dokumen (Download model embedding mungkin butuh waktu di awal)..."):
        try:
            st.session_state.retriever = process_document(uploaded_file)
            st.success("Dokumen siap! Silakan mulai simulasi.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- TAMPILAN CHAT ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Contoh: Ceritakan pengalaman Anda memimpin tim...")

if user_input:
    if not api_key:
        st.warning("Masukkan Groq API Key dulu di sidebar.")
        st.stop()
    
    if "retriever" not in st.session_state:
        st.warning("Upload dokumen panduan dulu ya.")
        st.stop()

    # Tampilkan input user
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # --- RAG CHAIN (Menggunakan Llama-3 via Groq) ---
    template = """
    Anda adalah Senior HR Recruiter. Gunakan panduan berikut untuk menilai jawaban kandidat.
    
    KONTEKS PANDUAN:
    {context}

    JAWABAN KANDIDAT:
    {question}

    TUGAS:
    1. Cek apakah ada Situation, Task, Action, Result (STAR).
    2. Beri Nilai (0-100).
    3. Beri kritik pedas tapi membangun jika Action/Result tidak spesifik.
    4. Beri contoh perbaikan.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Menggunakan Model Llama3-8b atau 70b (Gratis di Groq)
    llm = ChatGroq(
        temperature=0.3, 
        model_name="llama3-8b-8192", 
        groq_api_key=api_key
    )

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    with st.chat_message("assistant"):
        with st.spinner("Llama-3 sedang menganalisis..."):
            try:
                response = rag_chain.invoke(user_input)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error koneksi ke Groq: {e}")