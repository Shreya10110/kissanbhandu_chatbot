from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from hybrid_search import HybridSearcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

app = Flask(__name__)
CORS(app)

print("Loading Hybrid Searcher and local LLM generator...")
searcher = HybridSearcher()

llm_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
model = AutoModelForSeq2SeqLM.from_pretrained(llm_name, low_cpu_mem_usage=True)

@app.route("/")
def home():
    return send_file("kisanbandhu_ui.html")

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("q", "").strip()
    lang = data.get("lang", "en").strip().lower()

    if not question:
        return jsonify({"answer": "Please type a question.", "source": "KisanBandhu"})

    # 1. Translate query to English if it's not English
    query_in_english = question
    if lang in ["hi", "mr"]:
        try:
            from deep_translator import GoogleTranslator
            query_in_english = GoogleTranslator(source=lang, target='en').translate(question)
            print(f"Translated query from {lang} to en: {question} -> {query_in_english}")
        except Exception as e:
            print(f"Translation failed: {e}")

    # 2. Search the hybrid index using the English query
    results = searcher.search(query_in_english, k=3)
    if not results:
        no_info_msg = "I couldn't find relevant information for that question."
        if lang == "hi":
            no_info_msg = "मुझे उस प्रश्न के लिए प्रासंगिक जानकारी नहीं मिल सकी।"
        elif lang == "mr":
            no_info_msg = "मला त्या प्रश्नासाठी संबंधित माहिती आढळली नाही."
        return jsonify({"answer": no_info_msg, "source": "KisanBandhu AI Advisory"})

    # 3. Format context for prompt
    cleaned_results = []
    for r in results:
        r = r.strip()
        r = re.sub(r'\n+', ' ', r)
        cleaned_results.append(r)
    context = " ".join(cleaned_results)[:1200]

    # 4. Formulate Prompt
    prompt = f"Based on the following advisory, write a detailed, complete answer to the farmer's question: {query_in_english}\nAdvisory: {context}\nAnswer:"

    # 5. Generate response using local flan-t5-base on CPU
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        min_length=55,
        num_beams=4,
        repetition_penalty=1.25,
        early_stopping=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Clean any incomplete trailing sentences
    matches = list(re.finditer(r'[.!?](?:\s|$)', generated_text))
    if matches:
        generated_text = generated_text[:matches[-1].end()].strip()

    # 6. Translate response back to the user's chosen language
    final_answer = generated_text
    source_tag = "KisanBandhu AI Advisory"
    if lang in ["hi", "mr"]:
        try:
            from deep_translator import GoogleTranslator
            final_answer = GoogleTranslator(source='en', target=lang).translate(generated_text)
            if lang == "hi":
                source_tag = "किसानबंधु एआई सलाहकार"
            elif lang == "mr":
                source_tag = "किसानबंधू एआय सल्लागार"
        except Exception as e:
            print(f"Translation back to {lang} failed: {e}")

    return jsonify({"answer": final_answer, "source": source_tag})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)