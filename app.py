import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import docx2txt
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore


DATA_DIR = Path(__file__).parent / "data"
EMBEDDING_MODEL = "models/text-embedding-004"
CHAT_MODEL = "models/gemini-2.5-flash"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVER_K = 10

load_dotenv()

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn là trợ lý học thuật sử dụng đúng ngữ cảnh được cung cấp. "
            "Trả lời bằng tiếng Việt, súc tích và luôn dẫn chứng bằng mã nguồn dạng [TênTàiLiệu#Đoạn]. "
            "Nếu không thấy câu trả lời trong tài liệu thì nói rõ.",
        ),
        (
            "human",
            "Câu hỏi: {question}\n\n" "Các đoạn tài liệu (mỗi đoạn có nhãn tham chiếu ở đầu):\n{context}",
        ),
    ]
)


def sanitize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def ensure_api_key() -> str:
    api_key = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Chưa tìm thấy GOOGLE_API_KEY. Thiết lập biến môi trường trước khi chạy ứng dụng.")
    return api_key


def read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    return docx2txt.process(str(path)) or ""


def load_documents(data_dir: Path) -> List[Document]:
    documents: List[Document] = []
    if not data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

    for file_path in sorted(data_dir.glob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        text = ""
        try:
            if suffix == ".pdf":
                text = read_pdf(file_path)
            elif suffix in {".docx", ".doc"}:
                text = read_docx(file_path)
            elif suffix in {".txt", ".md"}:
                text = file_path.read_text(encoding="utf-8")
            else:
                continue
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.warning(f"Không thể đọc {file_path.name}: {exc}")
            continue

        clean_text = sanitize_text(text)
        if clean_text:
            documents.append(Document(page_content=clean_text, metadata={"source": file_path.name}))
    return documents


def split_documents(documents: Sequence[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(list(documents))
    per_source_counter: Dict[str, int] = Counter()
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        per_source_counter[source] += 1
        chunk.metadata["chunk_id"] = per_source_counter[source]
    return chunks


def build_vector_store(chunks: Sequence[Document]) -> InMemoryVectorStore:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=ensure_api_key(),
    )
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(list(chunks))
    return vector_store


@st.cache_resource(show_spinner=False)
def load_knowledge_base() -> Tuple[InMemoryVectorStore, List[Document]]:
    raw_docs = load_documents(DATA_DIR)
    if not raw_docs:
        raise ValueError("Không có tài liệu nào trong thư mục data để lập chỉ mục.")
    chunks = split_documents(raw_docs)
    store = build_vector_store(chunks)
    return store, chunks


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.1,
        google_api_key=ensure_api_key(),
    )


def format_docs(docs: Sequence[Document]) -> str:
    formatted = []
    for doc in docs:
        ref = format_reference(doc)
        formatted.append(f"[{ref}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def format_reference(doc: Document) -> str:
    chunk_id = doc.metadata.get("chunk_id")
    source = doc.metadata.get("source", "unknown")
    return f"{source}#{chunk_id}" if chunk_id else source


def answer_question(
    vector_store: InMemoryVectorStore, question: str, llm: ChatGoogleGenerativeAI
) -> Tuple[str, List[str]]:
    if not question.strip():
        return "Hãy nhập câu hỏi cụ thể nhé!", []

    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return (
            "Mình chưa tìm thấy thông tin phù hợp trong bộ tài liệu hiện có. "
            "Vui lòng thử diễn đạt lại câu hỏi hoặc kiểm tra thư mục data.",
            [],
        )

    context = format_docs(docs)
    chain_input = {"question": question, "context": context}
    messages = QA_PROMPT.format_messages(**chain_input)
    response = llm.invoke(messages)
    answer_text = getattr(response, "content", str(response))

    refs = []
    for doc in docs:
        ref = format_reference(doc)
        if ref not in refs:
            refs.append(ref)

    answer_with_refs = f"{answer_text}\n\nNguồn: {', '.join(refs)}"
    return answer_with_refs, refs


def render_sidebar(chunks: Sequence[Document]) -> None:
    st.sidebar.header("Nguồn tài liệu được trích dẫn")
    counts = Counter(doc.metadata.get("source", "unknown") for doc in chunks)
    for name, count in counts.items():
        st.sidebar.markdown(f"- {name}")
    st.sidebar.caption(f"Trích dẫn {RETRIEVER_K} đoạn tài liệu gần nhất cho mỗi câu hỏi.")


def main():
    st.set_page_config(page_title="VNRchat", layout="wide")
    st.title("Hỏi đáp về công cuộc Đổi mới từ 1986 đến nay")
    st.caption("Hỏi bất kỳ điều gì về kho tài liệu; câu trả lời sẽ kèm tham chiếu Gemini.")

    try:
        vector_store, chunks = load_knowledge_base()
        llm = get_llm()
    except Exception as exc:
        st.error(f"Lỗi khi khởi tạo hệ thống: {exc}")
        return

    render_sidebar(chunks)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Bạn muốn biết điều gì?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang hỏi Gemini và tra cứu tài liệu..."):
                response, _ = answer_question(vector_store, prompt, llm)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
