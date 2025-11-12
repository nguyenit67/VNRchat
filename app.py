import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import docx2txt
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = Path(__file__).parent / "data"


@dataclass
class DocumentChunk:
    content: str
    source_name: str
    chunk_id: int

    @property
    def reference(self) -> str:
        return f"{self.source_name}#{self.chunk_id + 1}"


def sanitize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if not text:
        return []
    tokens = text.split()
    if not tokens:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        chunk = " ".join(chunk_tokens).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_docx(path: Path) -> str:
    return docx2txt.process(str(path)) or ""


def load_documents(data_dir: Path) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for file_path in sorted(data_dir.glob("*")):
        if not file_path.is_file():
            continue
        text = ""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            try:
                text = read_pdf(file_path)
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.warning(f"Không thể đọc {file_path.name}: {exc}")
                continue
        elif suffix in {".docx", ".doc"}:
            try:
                text = read_docx(file_path)
            except Exception as exc:  # pragma: no cover
                st.warning(f"Không thể đọc {file_path.name}: {exc}")
                continue
        elif suffix in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8")
        else:
            continue

        clean_text = sanitize_text(text)
        for idx, content in enumerate(chunk_text(clean_text)):
            chunks.append(DocumentChunk(content=content, source_name=file_path.name, chunk_id=idx))
    return chunks


class DocumentIndex:
    def __init__(self, data_dir: Path):
        if not data_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")
        self.chunks = load_documents(data_dir)
        if not self.chunks:
            raise ValueError("Không có nội dung nào trong thư mục data để lập chỉ mục.")
        texts = [chunk.content for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        if not query.strip():
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        paired = list(enumerate(scores))
        paired.sort(key=lambda item: item[1], reverse=True)
        results = []
        for idx, score in paired[:top_k]:
            if score <= 0:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


def summarize_snippet(text: str, limit: int = 400) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        candidate = f"{summary} {sentence}".strip() if summary else sentence.strip()
        if len(candidate) > limit and summary:
            break
        summary = candidate
        if len(summary) >= limit:
            break
    if not summary:
        summary = text[:limit]
    return summary.strip()


@st.cache_resource(show_spinner=False)
def load_index() -> DocumentIndex:
    return DocumentIndex(DATA_DIR)


def answer_question(index: DocumentIndex, question: str) -> Tuple[str, List[str]]:
    hits = index.search(question)
    if not hits:
        return ("Mình chưa tìm thấy thông tin phù hợp trong tài liệu. "
                "Hãy thử diễn đạt câu hỏi theo cách khác nhé!"), []

    body_parts = []
    refs = []
    for chunk, _ in hits:
        snippet = summarize_snippet(chunk.content)
        ref_tag = chunk.reference
        body_parts.append(f"{snippet} [{ref_tag}]")
        refs.append(ref_tag)

    reference_line = "Nguồn: " + ", ".join(dict.fromkeys(refs))
    return "\n\n".join(body_parts) + "\n\n" + reference_line, refs


def render_sidebar(index: DocumentIndex):
    st.sidebar.header("Tài liệu đã lập chỉ mục")
    doc_counts = {}
    for chunk in index.chunks:
        doc_counts[chunk.source_name] = doc_counts.get(chunk.source_name, 0) + 1
    for name, count in doc_counts.items():
        st.sidebar.markdown(f"- {name} ({count} đoạn)")
    st.sidebar.caption("Mỗi đoạn ~800 từ với chồng lấn để giữ ngữ cảnh.")


def main():
    st.set_page_config(page_title="VNRchat", layout="wide")
    st.title("Hỏi đáp tài liệu Đảng sử")
    st.caption("Đặt câu hỏi và nhận câu trả lời kèm tài liệu tham chiếu.")

    try:
        index = load_index()
    except Exception as exc:
        st.error(f"Lỗi khi tải dữ liệu: {exc}")
        return

    render_sidebar(index)

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
            with st.spinner("Đang tìm trong tài liệu..."):
                response, refs = answer_question(index, prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
