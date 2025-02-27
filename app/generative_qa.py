import os
import json
import numpy as np
import streamlit as st
import yaml
from datasets import Dataset
from transformers import pipeline
import torch

from utils.embedding import get_embeddings

# ==============================
# 1. KONFIGURASI STREAMLIT
# ==============================
#st.set_page_config(layout="wide")

@st.cache_resource
def load_config():
    with open('cfg/config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config = load_config()

TRAIN_FILE = config["Config"]["DATASET_NAME"]            # ex: "data/train.json"
FAISS_INDEX_PATH = "faiss_index/my_index.faiss"
CONTEXT_EMBEDDING_COL = "context_embedding"
PIPELINE_NAME = config["Config"]["PIPELINE_NAME"]        # ex: "question-answering"
FINETUNED_MODEL_NAME = config["Config"]["FINETUNED_MODEL_NAME"]  # ex: "distilbert-finetuned-squadv2"
TOP_K = config.get("Config", {}).get("TOP_K", 5)         # default 5

# ==============================
# 2. MUAT PIPELINE QA
# ==============================
@st.cache_resource
def load_qa_pipeline():
    return pipeline(PIPELINE_NAME, model=FINETUNED_MODEL_NAME)

qa_pipeline = load_qa_pipeline()

# ==============================
# 3. FUNGSI MEMUAT DATASET TRAIN + FAISS INDEX
# ==============================
@st.cache_resource
def load_train_dataset_with_faiss():
    """
    Memuat dataset train yang sama seperti di faiss_index.py,
    lalu load FAISS index 'my_index.faiss'.
    """
    with open(TRAIN_FILE, "r", encoding='utf-8') as f:
        squad_dict = json.load(f)

    contexts, questions, answers, ids = [], [], [], []
    for article in squad_dict["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id_ = qa["id"]
                if "answers" not in qa or not qa["answers"]:
                    answer = {"text": [""], "answer_start": [0]}
                else:
                    answer = {
                        "text": [qa["answers"][0]["text"]],
                        "answer_start": [qa["answers"][0]["answer_start"]]
                    }
                contexts.append(context)
                questions.append(question)
                ids.append(id_)
                answers.append(answer)

    dataset = Dataset.from_dict({
        "id": ids,
        "context": contexts,
        "question": questions,
        "answers": answers
    })

    # Load FAISS index. Pastikan kolom "context_embedding" sudah ada.
    # Namun, kolom ini belum ada di dataset saat ini. 
    # Trik: Gunakan load_faiss_index() agar dataset menautkan index walau kolom aslinya belum tampak.
    dataset.load_faiss_index(CONTEXT_EMBEDDING_COL, FAISS_INDEX_PATH)
    return dataset

# ==============================
# 4. FUNGSI UTAMA: generate_answer
# ==============================
@st.cache_data
def generate_answer(input_question):
    """
    - Menghitung embedding pertanyaan
    - Mencari TOP_K context paling relevan
    - Memilih jawaban terbaik (skor tertinggi)
    """
    dataset = load_train_dataset_with_faiss()

    # 4.1. Hitung embedding pertanyaan
    question_emb = get_embeddings([input_question])  # shape [1, emb_dim]
    question_emb = question_emb.cpu().detach().numpy()

    # 4.2. Dapatkan k context terdekat berdasarkan context_embedding
    scores, samples = dataset.get_nearest_examples(
        CONTEXT_EMBEDDING_COL,
        question_emb,
        k=TOP_K
    )

    # 4.3. Pipeline Q&A: pakai pertanyaan input dan context dari dataset
    answer_scores = []
    for idx, score in enumerate(scores):
        context_text = samples["context"][idx]
        ans = qa_pipeline(question=input_question, context=context_text)
        # Jika pipeline mengembalikan list, ambil elemen pertama
        if isinstance(ans, list) and len(ans) > 0:
            ans = ans[0]
        answer_scores.append(ans["score"])

    best_idx = int(np.argmax(answer_scores))
    best_context = samples["context"][best_idx]

    final_answer = qa_pipeline(question=input_question, context=best_context)
    if isinstance(final_answer, list) and len(final_answer) > 0:
        final_answer = final_answer[0]

    return final_answer["answer"], final_answer["score"]

# ==============================
# 5. Fungsi Render untuk Streamlit
# ==============================
def render():
    st.subheader("Tanyakan kondisi kesehatanmu pada saya")
    st.write("Silakan masukkan pertanyaan di bawah ini:")

    with st.form("form_generative_qa", clear_on_submit=False):
        question = st.text_area("Pertanyaan:")
        submit = st.form_submit_button("Submit", type="primary")

        if submit:
            st.success("Selesai!")
            answer, score = generate_answer(question)
            st.subheader("Jawaban:")
            st.info(f"🤖 {answer}")
            st.subheader("Skor:")
            st.info(score)
