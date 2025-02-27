import os
import json
import torch
import faiss
from datasets import Dataset
import yaml

# Fungsi get_embeddings harus ada di utils/embedding.py
# Pastikan sudah menyesuaikan model embedding (misalnya Sentence-BERT)
from utils.embedding import get_embeddings

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# === 1. BACA KONFIGURASI DARI FILE YAML ===
with open('cfg/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

TRAIN_FILE = config["Config"]["DATASET_NAME"]  # ex: "data/train.json"
FAISS_INDEX_DIR = "faiss_index"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "my_index.faiss")

# Kita tentukan nama kolom embedding yang akan dibuat
CONTEXT_EMBEDDING_COL = "context_embedding"

# === 2. FUNGSI MEMUAT DATASET SQUAD DARI FILE LOKAL (TRAIN) ===
def load_squad_data(file_path):
    """
    Membaca file JSON berformat SQuAD dan mengonversinya ke Dataset Hugging Face.
    Menghasilkan kolom: [id, context, question, answers].
    """
    with open(file_path, "r", encoding="utf-8") as f:
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

    return Dataset.from_dict({
        "id": ids,
        "context": contexts,
        "question": questions,
        "answers": answers
    })

if __name__ == "__main__":
    # === 3. MUAT DATASET TRAIN ===
    train_dataset = load_squad_data(TRAIN_FILE)
    
    # (Opsional) Filter contoh yang benar-benar punya context (biasanya selalu ada)
    # train_dataset = train_dataset.filter(lambda x: len(x["context"].strip()) > 0)

    # === 4. BUAT KOLOM EMBEDDING DARI context ===
    # Kita memetakan embedding untuk setiap context, lalu simpan di kolom CONTEXT_EMBEDDING_COL
    def make_context_embedding(example):
        emb = get_embeddings([example["context"]])  # Hasil shape [1, emb_dim]
        return {CONTEXT_EMBEDDING_COL: emb.detach().cpu().numpy()[0]}

    embeddings_dataset = train_dataset.map(
        make_context_embedding,
        batched=False
    )

    # === 5. BANGUN FAISS INDEX DARI KOLOM context_embedding ===
    embeddings_dataset.add_faiss_index(column=CONTEXT_EMBEDDING_COL)

    # Pastikan direktori untuk menyimpan index ada
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    # === 6. SIMPAN FAISS INDEX KE FILE ===
    embeddings_dataset.save_faiss_index(CONTEXT_EMBEDDING_COL, FAISS_INDEX_PATH)
    print(f"✅ FAISS index berhasil disimpan di '{FAISS_INDEX_PATH}'")
