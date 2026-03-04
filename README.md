# VNRchat 🇻🇳

Ứng dụng hỏi đáp thông minh về **công cuộc Đổi Mới của Việt Nam từ 1986 đến nay**, sử dụng [Streamlit](https://streamlit.io/) và Google Gemini AI kết hợp kỹ thuật RAG (Retrieval-Augmented Generation).

---

## Tính năng

- 💬 Giao diện chat tương tác xây dựng bằng Streamlit
- 📄 Đọc và lập chỉ mục tài liệu từ thư mục `data/` (hỗ trợ `.pdf`, `.docx`, `.doc`, `.txt`, `.md`)
- 🔍 Tìm kiếm ngữ nghĩa với Google Generative AI Embeddings (`text-embedding-004`)
- 🤖 Trả lời bằng Gemini (`gemini-2.5-flash`) kèm tham chiếu trực tiếp đến đoạn tài liệu nguồn
- 🗂️ Thanh bên hiển thị danh sách tài liệu đang được sử dụng

---

## Cấu trúc dự án

```
VNRchat/
├── app.py            # Mã nguồn chính
├── data/             # Thư mục chứa tài liệu (PDF, DOCX, TXT, MD)
├── pyproject.toml    # Cấu hình dự án & phụ thuộc
├── uv.lock           # Lock file cho uv
└── .devcontainer/    # Cấu hình GitHub Codespaces / VS Code Dev Container
```

---

## Yêu cầu

- Python ≥ 3.9
- [uv](https://github.com/astral-sh/uv) (khuyến nghị) hoặc `pip`
- Google API Key có quyền truy cập Gemini & Generative AI Embeddings

---

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/nguyenit67/VNRchat.git
cd VNRchat
```

### 2. Cài đặt phụ thuộc

Dùng **uv** (khuyến nghị):

```bash
uv sync
```

Hoặc dùng **pip**:

```bash
pip install -e .
```

### 3. Thiết lập API Key

Tạo file `.env` ở thư mục gốc:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Hoặc thiết lập biến môi trường trực tiếp:

```bash
export GOOGLE_API_KEY=your_google_api_key_here
```

> **Lưu ý:** Khi triển khai lên Streamlit Cloud, đặt `GOOGLE_API_KEY` trong **Secrets** của ứng dụng.

### 4. Thêm tài liệu

Đặt các file PDF, DOCX, TXT hoặc MD vào thư mục `data/`. Ứng dụng sẽ tự động đọc và lập chỉ mục tất cả tài liệu khi khởi động.

---

## Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ khởi động tại [http://localhost:8501](http://localhost:8501).

---

## Sử dụng GitHub Codespaces

Repository này hỗ trợ [GitHub Codespaces](https://github.com/features/codespaces). Nhấn **Code → Open with Codespaces** để khởi động môi trường phát triển sẵn sàng, ứng dụng Streamlit sẽ tự động chạy trong Codespace.

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Giao diện | [Streamlit](https://streamlit.io/) |
| LLM | [Google Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) |
| Embeddings | Google `text-embedding-004` |
| RAG Framework | [LangChain](https://www.langchain.com/) |
| Vector Store | `InMemoryVectorStore` (LangChain Community) |
| Đọc PDF | [PyPDF2](https://pypdf2.readthedocs.io/) |
| Đọc DOCX | [docx2txt](https://github.com/ankushshah89/python-docx2txt) |

---

## Giấy phép

Dự án này được phân phối theo giấy phép [MIT](LICENSE).
