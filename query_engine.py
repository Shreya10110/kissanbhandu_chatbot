from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import torch


def load_index(index_dir="faiss_index"):
    """Load FAISS index and corresponding text chunks."""
    idx = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "index.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return idx, chunks


def embed_query(query, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode([query], convert_to_numpy=True)[0]


def load_llm():
    """Load flan-t5-base — lightweight, CPU-friendly, encoder-decoder model."""
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    class FlanT5Generator:
        def __init__(self, tokenizer, model):
            self.tokenizer = tokenizer
            self.model = model

        def __call__(self, prompt):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512  # flan-t5-base's practical input limit
            )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,          # beam search gives more coherent answers than sampling for this model size
                temperature=0.7,
                repetition_penalty=1.3,  # discourages the repeated-sentence problem
                early_stopping=True
            )
            # Encoder-decoder models only output the NEW tokens, no prompt echo
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return [{"generated_text": generated_text}]

    return FlanT5Generator(tokenizer, model)


def search_query(query, k=3):
    """Reduced k from 5 to 3 — flan-t5-base has a small context window,
    so fewer, more relevant chunks work better than many chunks."""
    idx, chunks = load_index()
    q_vec = embed_query(query)
    D, I = idx.search(q_vec.reshape(1, -1), k)
    results = [chunks[i] for i in I[0]]
    return results


def generate_answer(question, docs):
    # Trim each chunk more tightly since flan-t5-base's input limit is only ~512 tokens.
    # 3 chunks x 400 chars is a safer budget than 5 chunks x 800 chars.
    context = "\n".join([doc[:400] for doc in docs])

    # flan-t5 is instruction-tuned — short, direct instructions work far better
    # than long rule lists (which is what was likely confusing the small model before).
    prompt = f"""Answer the farmer's question using only the context below. Be clear, practical, and specific about timing and recommendations.

Context: {context}

Question: {question}

Answer:"""

    # DEBUG: print what's actually being retrieved and sent to the model
    print("\n--- RETRIEVED CONTEXT ---")
    print(context)
    print("--- END CONTEXT ---\n")

    llm = load_llm()
    result = llm(prompt)
    return result[0]["generated_text"]


if __name__ == "__main__":

    question = input("Ask farmer question (or 'quit' to exit): ")

    if question.lower() == "quit":
        exit()

    results = search_query(question)

    answer = generate_answer(question, results)

    print("\nFinal Answer:\n")
    print(answer)