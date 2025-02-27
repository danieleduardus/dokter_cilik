import torch
from transformers import AutoTokenizer
import yaml

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Memuat konfigurasi dari file YAML
with open('cfg/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

MODEL_NAME = config["Config"]["MODEL_NAME"]
MAX_LENGTH = config["Config"]["MAX_LENGTH"]
STRIDE = config["Config"]["STRIDE"]

# Inisialisasi tokenizer dari model yang telah ditentukan
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_training_examples(examples):
    # Ambil daftar pertanyaan dan hilangkan spasi berlebih
    questions = [q.strip() for q in examples["question"]]
    
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    # Simpan offset_mapping dan mapping overflow, lalu hapus dari inputs
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    
    # Iterasi melalui setiap chunk yang dihasilkan oleh tokenizer
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        sequence_ids = inputs.sequence_ids(i)
        
        # Tentukan indeks awal dan akhir konteks (bagian dengan sequence_id == 1)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        answer = answers[sample_idx]
        if len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            
            # Jika jawaban di luar batas konteks token, set posisi sebagai 0
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    
    return inputs

def preprocess_validation_examples(examples):
    # Bersihkan pertanyaan dari spasi berlebih
    questions = [q.strip() for q in examples["question"]]
    
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    
    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
    
    inputs["example_id"] = example_ids
    return inputs
