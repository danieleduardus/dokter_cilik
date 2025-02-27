import os
import json
import yaml
import evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from utils.preprocess import preprocess_training_examples, preprocess_validation_examples
from utils.metric import compute_metrics

# === 1. MEMUAT KONFIGURASI DARI YAML === #
with open('cfg/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

MODEL_NAME = config["Config"]["MODEL_NAME"]
MAX_LENGTH = config["Config"]["MAX_LENGTH"]
STRIDE = config["Config"]["STRIDE"]
TRAIN_FILE = config["Config"]["DATASET_NAME"]            # misal: "data/train.json"
VAL_FILE = config["Config"]["VALIDATION_DATASET_NAME"]     # misal: "data/validation.json"
FINETUNED_MODEL_NAME = config["Config"]["FINETUNED_MODEL_NAME"]

# === 2. FUNGSI UNTUK MEMUAT DATASET SQUAD DARI FILE LOKAL === #
def load_squad_data(file_path):
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

# === 3. MEMUAT DATASET TRAIN DAN VALIDASI === #
train_dataset = load_squad_data(TRAIN_FILE)
validation_dataset = load_squad_data(VAL_FILE)

raw_datasets = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset
})

# Simpan dataset raw (opsional, untuk keperluan FAISS atau audit)
os.makedirs(FINETUNED_MODEL_NAME, exist_ok=True)
raw_datasets.save_to_disk(os.path.join(FINETUNED_MODEL_NAME, "raw_dataset"))
print("✅ Raw dataset berhasil disimpan.")

# === 4. MEMUAT TOKENIZER DAN MODEL === #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# === 5. PREPROCESSING DATASET === #
train_dataset = train_dataset.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=train_dataset.column_names
)

validation_dataset = validation_dataset.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=validation_dataset.column_names
)

# Validasi bahwa kolom-kolom penting sudah ada setelah preprocessing
if "input_ids" not in train_dataset.column_names:
    raise ValueError("❌ Kolom 'input_ids' hilang setelah preprocessing! Periksa kembali fungsi preprocessing.")

# === 6. KONFIGURASI TRAINING === #
args = TrainingArguments(
    output_dir=FINETUNED_MODEL_NAME,
    evaluation_strategy="no",        # atau "epoch" jika ingin evaluasi setiap epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)

# === 7. TRAINING MODEL === #
trainer.train()

# === 8. EVALUASI MODEL === #
metric = evaluate.load("squad_v2")
predictions = trainer.predict(validation_dataset)
if isinstance(predictions.predictions, tuple):
    start_logits, end_logits = predictions.predictions
else:
    start_logits, end_logits = predictions.predictions[0], predictions.predictions[1]

results = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
print("🔎 Evaluation Results:", results)

# === 9. MENYIMPAN DATASET, MODEL, DAN TOKENIZER === #
# Jika ingin menambahkan kembali kolom 'question', pastikan jumlahnya sesuai dengan dataset hasil preprocessing.
# Berikut contoh menambahkan kolom 'question' dengan menduplikasi nilai sesuai mapping overflow.
# Jika tidak diperlukan, bagian ini dapat dihilangkan.

def duplicate_column(raw_dataset, processed_dataset, column_name):
    # Mendapatkan mapping overflow agar bisa menduplikasi nilai
    duplicated = []
    for i in range(len(processed_dataset)):
        # Menggunakan overflow_to_sample_mapping jika tersedia
        duplicated.append(raw_dataset[column_name][i % len(raw_dataset[column_name])])
    return duplicated

# Contoh: menambahkan kembali kolom 'question' (hanya jika diperlukan)
# Perhatikan bahwa pendekatan ini hanya valid jika urutan data sudah sesuai.
train_questions = duplicate_column(raw_datasets["train"], train_dataset, "question")
validation_questions = duplicate_column(raw_datasets["validation"], validation_dataset, "question")

train_dataset = train_dataset.add_column("question", train_questions)
validation_dataset = validation_dataset.add_column("question", validation_questions)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset
})
dataset_dict.save_to_disk(os.path.join(FINETUNED_MODEL_NAME, "dataset"))
print("✅ Dataset hasil training telah disimpan di folder:", os.path.join(FINETUNED_MODEL_NAME, "dataset"))

trainer.save_model(os.path.join(FINETUNED_MODEL_NAME, "checkpoint-final"))
tokenizer.save_pretrained(os.path.join(FINETUNED_MODEL_NAME, "checkpoint-final"))
print("✅ Model dan tokenizer telah disimpan di folder:", os.path.join(FINETUNED_MODEL_NAME, "checkpoint-final"))

trainer.push_to_hub(commit_message="Training complete")
