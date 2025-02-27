import torch
from sentence_transformers import SentenceTransformer
import yaml

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Muat konfigurasi dari file YAML
with open('cfg/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Pastikan Anda mendefinisikan entri MODEL_NAME khusus untuk Sentence-BERT
# di config.yaml, misal:
# Config:
#   SBERT_MODEL_NAME: "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = config["Config"].get("SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Inisialisasi model Sentence-BERT
model = SentenceTransformer(MODEL_NAME, device=device)

def get_embeddings(text_list):
    """
    Menghasilkan embedding dari list string (atau string tunggal).
    :param text_list: list of strings atau string tunggal
    :return: tensor shape (batch_size, embedding_dim)
    """
    if isinstance(text_list, str):
        text_list = [text_list]

    # Encode teks menggunakan Sentence-BERT
    embeddings = model.encode(text_list, convert_to_tensor=True, show_progress_bar=False)
    # embeddings akan berupa torch.Tensor dengan shape [batch_size, emb_dim]
    return embeddings
