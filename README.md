# KisanBandhu: Multilingual AI Farming Advisory Chatbot

KisanBandhu is a **multilingual, offline-ready AI Advisory Chatbot** designed to help farmers get immediate, scientific crop guidelines from PDF manuals. It runs 100% locally on standard laptop CPUs with **zero external paid API costs** (no OpenAI/Gemini keys required).

---

## 🌟 Key Highlights (For Recruiters)

* **Cross-Lingual RAG (Retrieval-Augmented Generation)**: Farmers can query in regional languages (**Hindi and Marathi**). The system dynamically translates the query to search an English PDF database, generates a precise answer using a local LLM, and translates the response back to the user's selected language.
* **Hybrid Semantic & Keyword Search**: Integrates **FAISS (Dense Vector Search)** for conceptual understanding and **BM25 (Sparse Keyword Search)** to capture exact numbers, pesticide names, and crop values.
* **Dynamic Context Routing**: Automatically extracts target crop/plant nouns from queries to boost target content (e.g., Mango guidelines) and penalize unrelated crops (e.g., Wheat guidelines), avoiding cross-context confusion.
* **100% Local CPU Execution**: Powered by `google/flan-t5-base` and `sentence-transformers/all-MiniLM-L6-v2`. Optimized using PyTorch thread pooling to deliver detailed answers in **under 4 seconds** on average CPU hardware.

---

## 🛠️ Tech Stack

* **Backend**: Python, Flask, PyTorch
* **AI & NLP**: Hugging Face `transformers`, `sentence-transformers`, `faiss-cpu`, `deep-translator`
* **Frontend**: Responsive Vanilla HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API)

---

## 📂 Project Architecture

```
├── app.py                # Flask Web Server & Gen-AI routing
├── hybrid_search.py      # Custom BM25 + FAISS Hybrid Searcher
├── rag_pipeline.py       # Document chunking & FAISS index builder
├── kisanbandhu_ui.html   # Main Web Interface (served at /)
├── data/                 # Folder containing scientific agricultural PDFs
└── faiss_index/          # Local vector database storage
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Shreya10110/kissanbhandu_chatbot.git
cd kissanbhandu_chatbot
```

### 2. Install Dependencies
```bash
pip install flask flask-cors sentence-transformers faiss-cpu transformers torch deep-translator pypdf
```

### 3. Build Vector Index
Place your agricultural PDFs inside the `data/` folder and build the local vector database:
```bash
python rag_pipeline.py
```

### 4. Start the Application
Run the Flask server:
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser to start chatting!