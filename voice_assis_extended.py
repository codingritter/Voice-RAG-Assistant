import os
import gc
import json
import uuid
import shutil
import logging
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple

import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

from elevenlabs.client import ElevenLabs

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("voice_rag_assistant")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler("voice_rag_assistant.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()

CHAT_HISTORY_FILE = "chat_history.json"


def save_chat_history(history: List[dict]) -> None:
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Could not save chat history: {e}")


def load_chat_history() -> List[dict]:
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load chat history: {e}")
    return []


def _kb_pointer_file(base_directory: str) -> str:
    return os.path.join(base_directory, "_active.txt")


def _get_active_kb_dir(base_directory: str) -> Optional[str]:
    pointer = _kb_pointer_file(base_directory)
    if os.path.exists(pointer):
        try:
            with open(pointer, "r", encoding="utf-8") as f:
                name = f.read().strip()
        except Exception as e:
            logger.warning(f"Could not read knowledge base pointer file: {e}")
            return None
        path = os.path.join(base_directory, name)
        if os.path.isdir(path):
            return path
    return None


def _set_active_kb_dir(base_directory: str, version_dir: str) -> None:
    os.makedirs(base_directory, exist_ok=True)
    with open(_kb_pointer_file(base_directory), "w", encoding="utf-8") as f:
        f.write(os.path.basename(version_dir))


def cleanup_orphaned_kb_dirs(base_directory: str = "knowledge_base") -> None:
    """Löscht alte, nicht mehr aktive Wissensdatenbank-Versionen."""
    if not os.path.isdir(base_directory):
        return
    active = _get_active_kb_dir(base_directory)
    for name in os.listdir(base_directory):
        path = os.path.join(base_directory, name)
        if not os.path.isdir(path):
            continue
        if active and os.path.abspath(path) == os.path.abspath(active):
            continue
        try:
            shutil.rmtree(path)
            logger.info(f"Removed orphaned knowledge base folder: {path}")
        except Exception as e:
            logger.warning(f"Could not remove orphaned folder {path}: {e}")


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )

    def load_documents(self, directory: str) -> List[Document]:
        """Load documents from different file types"""
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(
                directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
            ),
        }

        documents: List[Document] = []
        for file_type, loader in loaders.items():
            try:
                loaded = loader.load()
                documents.extend(loaded)
                logger.info(f"Loaded {len(loaded)} {file_type} document(s)")
            except Exception as e:
                logger.error(f"Error loading {file_type} documents: {e}")

        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s)")
        return chunks

    def create_vector_store(
        self, documents: List[Document], base_directory: str = "knowledge_base"
    ) -> Chroma:
        """Erstellt eine neue, eindeutig benannte Version der Wissensdatenbank."""
        version_dir = os.path.join(base_directory, f"kb_{uuid.uuid4().hex[:12]}")
        os.makedirs(version_dir, exist_ok=True)
        logger.info(f"Creating new vector store in {version_dir}")

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=version_dir,
        )
        if hasattr(vector_store, "persist"):
            vector_store.persist()

        _set_active_kb_dir(base_directory, version_dir)
        return vector_store

    def load_existing_vector_store(
        self, base_directory: str = "knowledge_base"
    ) -> Optional[Chroma]:
        """Lädt die zuletzt erstellte Wissensdatenbank-Version, ohne
        Dokumente erneut hochladen zu müssen. Gibt None zurück, falls noch
        keine Version existiert."""
        active_dir = _get_active_kb_dir(base_directory)
        if not active_dir:
            return None
        logger.info(f"Loading existing vector store from {active_dir}")
        return Chroma(persist_directory=active_dir, embedding_function=self.embeddings)


class VoiceGenerator:
    def __init__(self, api_key: str):
        self.client = ElevenLabs(api_key=api_key)
        self._voice_id_cache: dict = {}
        self._voices_loaded = False

    def _load_voices(self) -> None:
        """Lädt einmalig die im Account verfügbaren Stimmen."""
        if self._voices_loaded:
            return
        try:
            response = self.client.voices.search()
            for voice in response.voices:
                self._voice_id_cache[voice.name] = voice.voice_id
        except Exception as e:
            logger.error(f"Could not load voices from ElevenLabs: {e}")
        self._voices_loaded = True

    def refresh_voices(self) -> None:
        """Erzwingt ein erneutes Laden der Stimmenliste."""
        self._voice_id_cache = {}
        self._voices_loaded = False
        self._load_voices()

    @property
    def available_voices(self) -> List[str]:
        """Namen der tatsächlich im Account verfügbaren Stimmen."""
        self._load_voices()
        return list(self._voice_id_cache.keys())

    @property
    def default_voice(self) -> Optional[str]:
        """Bevorzugt 'Rachel', falls vorhanden, sonst die erste verfügbare Stimme."""
        voices = self.available_voices
        if "Rachel" in voices:
            return "Rachel"
        return voices[0] if voices else None

    def _resolve_voice_id(self, voice_name: str) -> str:
        """Namen wie 'Rachel' auf die tatsächliche voice_id abbilden"""
        self._load_voices()

        if voice_name not in self._voice_id_cache:
            raise ValueError(f"Voice '{voice_name}' wurde in diesem Account nicht gefunden")

        return self._voice_id_cache[voice_name]

    def generate_voice_response(
        self, text: str, voice_name: Optional[str] = None
    ) -> Optional[str]:
        """Generate voice response. Returns the path to an mp3 file, or None on failure."""
        try:
            selected_voice = voice_name or self.default_voice
            if not selected_voice:
                raise ValueError("Keine Stimme im ElevenLabs-Account verfügbar")
            voice_id = self._resolve_voice_id(selected_voice)

            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )

            audio_bytes = b"".join(audio_generator)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                return temp_audio.name

        except Exception as e:
            logger.error(f"Error generating voice response: {e}")
            return None


class VoiceAssistantRAG:
    def __init__(self, elevenlabs_api_key: str, gemini_api_key: str):
        self.whisper_model = whisper.load_model("base")
        self.llm = ChatOpenAI(
            model="gemini-3.6-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_api_key,
            temperature=0,
        )
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )
        self.vector_store: Optional[Chroma] = None
        self.qa_chain: Optional[ConversationalRetrievalChain] = None
        self.sample_rate = 44100
        self.voice_generator = VoiceGenerator(elevenlabs_api_key)

    def setup_vector_store(self, vector_store: Chroma) -> None:
        """Initialize the vector store and QA chain"""
        self.vector_store = vector_store

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=memory,
            return_source_documents=True,
            verbose=True,
        )

    def record_audio(self, duration: int = 5):
        """Record audio from microphone"""
        recording = sd.rec(
            int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1
        )
        sd.wait()
        return recording

    def transcribe_audio(self, audio_array) -> str:
        """Transcribe microphone audio using Whisper."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
        try:
            sf.write(temp_path, audio_array, self.sample_rate)
            result = self.whisper_model.transcribe(temp_path)
            return result["text"]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def transcribe_audio_file(self, file_bytes: bytes, suffix: str = ".wav") -> str:
        """Transkribiert eine bereits vorhandene Audiodatei."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
            temp_path = temp_audio.name
        try:
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            result = self.whisper_model.transcribe(temp_path)
            return result["text"]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def generate_response(self, query: str) -> Tuple[str, List[Document]]:
        """Generate response using RAG system. Returns (answer, source_documents)."""
        if self.qa_chain is None:
            return "Error: Vector store not initialized", []

        response = self.qa_chain.invoke({"question": query})
        answer = response.get("answer", "")
        sources = response.get("source_documents", [])
        return answer, sources

    def text_to_speech(self, text: str, voice_name: Optional[str] = None) -> Optional[str]:
        """Convert text to speech"""
        return self.voice_generator.generate_voice_response(text, voice_name)


def setup_knowledge_base():
    """Standalone Streamlit-Seite zum Hochladen und Verarbeiten von Dokumenten."""
    st.title("Knowledge Base Setup")

    base_directory = "knowledge_base"

    with st.expander("Advanced chunking settings"):
        chunk_size = st.slider("Chunk size (characters)", 200, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, 200, step=50)

    doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    uploaded_files = st.file_uploader(
        "Upload your documents", accept_multiple_files=True, type=["pdf", "txt", "md"]
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            temp_dir = tempfile.mkdtemp()

            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            try:
                documents = doc_processor.load_documents(temp_dir)
                processed_docs = doc_processor.process_documents(documents)

                vector_store = doc_processor.create_vector_store(
                    processed_docs, base_directory
                )

                st.success(f"Processed {len(processed_docs)} document chunks!")
                return vector_store

            except Exception as e:
                logger.error(f"Error processing documents: {e}")
                st.error(f"Error processing documents: {str(e)}")
                return None

            finally:
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)

    st.divider()
    active_dir = _get_active_kb_dir(base_directory)
    if active_dir:
        st.caption(f"An existing knowledge base is stored (version: {os.path.basename(active_dir)}).")

        col_load, col_clear = st.columns(2)

        with col_load:
            if st.button("Load existing knowledge base"):
                vector_store = doc_processor.load_existing_vector_store(base_directory)
                if vector_store:
                    st.success("Loaded existing knowledge base.")
                    return vector_store
                st.error("No existing knowledge base found.")

        with col_clear:
            if st.button("Clear Knowledge Base", type="secondary"):
                st.session_state.pop("vector_store", None)
                st.session_state.pop("assistant", None)

                try:
                    from chromadb.api.client import SharedSystemClient

                    SharedSystemClient.clear_system_cache()
                except Exception as e:
                    logger.warning(f"Could not clear Chroma system cache: {e}")

                gc.collect()

                pointer = _kb_pointer_file(base_directory)
                if os.path.exists(pointer):
                    os.remove(pointer)

                try:
                    shutil.rmtree(active_dir)
                    logger.info(f"Removed knowledge base folder: {active_dir}")
                except Exception as e:
                    logger.warning(
                        f"Old knowledge base folder still locked, will be "
                        f"cleaned up on next app restart: {e}"
                    )

                st.success("Knowledge base cleared. Please upload documents again.")
                st.rerun()

    return None


def main():
    st.set_page_config(page_title="Voice RAG Assistant", layout="wide")

    if "kb_cleanup_done" not in st.session_state:
        cleanup_orphaned_kb_dirs()
        st.session_state.kb_cleanup_done = True

    elevenlabs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not all([elevenlabs_api_key, gemini_api_key]):
        st.error(
            "Please set ELEVEN_LABS_API_KEY and GEMINI_API_KEY in your environment variables"
        )
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Setup Knowledge Base", "Voice Assistant"])

    if page == "Setup Knowledge Base":
        vector_store = setup_knowledge_base()
        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.pop("assistant", None)
        return

    if "vector_store" not in st.session_state:
        with st.spinner("Loading existing knowledge base..."):
            doc_processor = DocumentProcessor()
            vector_store = doc_processor.load_existing_vector_store("knowledge_base")

        if vector_store:
            st.session_state.vector_store = vector_store
            logger.info("Automatically loaded existing knowledge base")
        else:
            st.error("Please setup knowledge base first!")
            return

    st.title("Voice Assistant RAG System")

    if "assistant" not in st.session_state:
        assistant = VoiceAssistantRAG(elevenlabs_api_key, gemini_api_key)
        assistant.setup_vector_store(st.session_state.vector_store)
        st.session_state.assistant = assistant
    assistant = st.session_state.assistant

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    try:
        available_voices = assistant.voice_generator.available_voices
        if not available_voices:
            st.error("No voices available")
            return

        if st.sidebar.button("Refresh voices"):
            assistant.voice_generator.refresh_voices()
            st.rerun()

        selected_voice = st.sidebar.selectbox(
            "Select Voice",
            available_voices,
            index=(
                available_voices.index("Rachel")
                if "Rachel" in available_voices
                else 0
            ),
        )
        duration = st.sidebar.slider("Recording Duration (seconds)", 1, 10, 5)

        input_method = st.sidebar.radio(
            "Input method", ["Microphone", "Upload audio file"]
        )

        show_sources = st.sidebar.checkbox("Show sources used for answers", value=True)

        query: Optional[str] = None

        if input_method == "Microphone":
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Start Recording"):
                    with st.spinner(f"Recording for {duration} seconds..."):
                        audio_data = assistant.record_audio(duration)
                        st.session_state.audio_data = audio_data
                        st.success("Recording completed!")

            with col2:
                if st.button("Process Recording"):
                    if "audio_data" not in st.session_state:
                        st.error("Please record audio first!")
                    else:
                        with st.spinner("Transcribing..."):
                            query = assistant.transcribe_audio(st.session_state.audio_data)
                            st.write("You said:", query)
        else:
            uploaded_audio = st.file_uploader(
                "Upload an audio file", type=["wav", "mp3", "m4a"]
            )
            if uploaded_audio and st.button("Transcribe uploaded audio"):
                suffix = os.path.splitext(uploaded_audio.name)[1] or ".wav"
                with st.spinner("Transcribing..."):
                    query = assistant.transcribe_audio_file(
                        uploaded_audio.getbuffer(), suffix=suffix
                    )
                    st.write("You said:", query)

        if query:
            response = None
            sources: List[Document] = []
            with st.spinner("Generating response..."):
                try:
                    response, sources = assistant.generate_response(query)
                    st.write("Response:", response)

                    if show_sources and sources:
                        with st.expander(f"Sources used ({len(sources)})"):
                            for i, doc in enumerate(sources, start=1):
                                raw_source = doc.metadata.get("source")
                                source_name = (
                                    os.path.basename(raw_source)
                                    if raw_source
                                    else "unknown source"
                                )
                                page = doc.metadata.get("page")
                                label = f"{source_name}" + (
                                    f" (page {page + 1})" if page is not None else ""
                                )
                                st.markdown(f"**{i}. {label}**")

                                preview = doc.page_content.strip()[:300]
                                if len(doc.page_content) > 300:
                                    preview += "..."
                                if not preview:
                                    preview = "_(empty chunk)_"
                                st.caption(preview)

                    st.session_state.last_response = response

                    entry = {
                        "query": query,
                        "answer": response,
                        "timestamp": datetime.now().isoformat(),
                    }
                    st.session_state.chat_history.append(entry)
                    save_chat_history(st.session_state.chat_history)

                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    st.error(f"Error generating response: {str(e)}")

            if response:
                with st.spinner("Converting to speech..."):
                    audio_file = assistant.voice_generator.generate_voice_response(
                        response, selected_voice
                    )
                    if audio_file:
                        st.audio(audio_file)
                        with open(audio_file, "rb") as f:
                            st.download_button(
                                "Download response audio",
                                data=f.read(),
                                file_name="response.mp3",
                                mime="audio/mp3",
                            )
                        os.unlink(audio_file)
                    else:
                        st.error("Failed to generate voice response")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error(f"Unexpected error: {str(e)}")

    if st.session_state.chat_history:
        st.subheader("Chat History")

        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.rerun()

        for entry in reversed(st.session_state.chat_history):
            st.write("Q:", entry["query"])
            st.write("A:", entry["answer"])
            st.caption(entry.get("timestamp", ""))
            st.write("---")


if __name__ == "__main__":
    main()