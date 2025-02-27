# Dokter Cilik Virtual

Dokter Cilik Virtual adalah aplikasi chatbot interaktif yang membantu Anda berkonsultasi mengenai penyakit-penyakit yang umum terjadi di masyarakat Indonesia. Aplikasi ini memberikan informasi mengenai gejala, solusi alami untuk mengatasi penyakit, serta tips kesehatan yang mudah dipahami.

## Fitur Utama

- **Konsultasi Kesehatan:** Dapatkan informasi terkait penyakit umum di Indonesia.
- **Solusi Alami:** Berikan solusi alami dan tips kesehatan untuk mengatasi gejala.
- **Chatbot Interaktif:** Menggunakan model Question Answering (QA) yang dikombinasikan dengan FAISS untuk pencarian konteks secara semantik.
- **Deploy Mudah:** Aplikasi berbasis Streamlit yang mudah untuk di-deploy dan digunakan.

## Teknologi yang Digunakan

- **Python 3.8+**
- **HuggingFace Transformers & Datasets**
- **FAISS (Facebook AI Similarity Search)**
- **Streamlit**
- **Sentence-BERT** (untuk embedding konteks yang lebih relevan)
- **PyTorch**

## Prasyarat

Pastikan Anda telah menginstal:
- Python (versi 3.8 atau lebih baru)
- Git

## Instalasi dan Deployment

Ikuti langkah-langkah berikut untuk menjalankan aplikasi:

1. **Clone Repository**
   ```bash
   git clone https://github.com/username/dokter-cilik-virtual.git
   cd dokter-cilik-virtual
   ```
   
2. **Install Dependensi**

   Pastikan Anda sudah membuat virtual environment (opsional) dan kemudian instal semua dependensi yang tertera di requirements.txt:

   ```bash
   pip install -r requirements.txt
   ```
   
3. **Training Model**

   Lakukan training model dengan menjalankan script trainer.py. Script ini akan melatih model Question Answering berdasarkan dataset lokal.

   ```bash
   python trainer.py
   ```
4. **Membangun FAISS Index**

   Setelah model selesai di-training, jalankan script faiss_index.py untuk membangun FAISS index dari dataset train.

   ```bash
   python faiss_index.py
   ```
5. **Menjalankan Aplikasi**

   Jalankan aplikasi menggunakan Streamlit:

   ```bash
   streamlit run app.py
   ```

## Struktur Proyek
```plaintext

dokter-cilik-virtual/
├── app/
│   ├── app.py
│   ├── extractive_qa.py
│   ├── generative_qa.py
│   └── sidebar.py
├── cfg/
│   └── config.yaml
├── data/
│   ├── train.json
│   └── validation.json
├── utils/
│   ├── embedding.py
│   ├── metric.py
│   └── preprocess.py
├── faiss_index.py
├── trainer.py
├── requirements.txt
└── README.md
```

##Konfigurasi
File cfg/config.yaml berisi konfigurasi penting seperti:

*Model yang digunakan untuk training dan embedding
*Parameter tokenisasi (MAX_LENGTH, STRIDE, dsb.)
*Nama dataset (misalnya data/train.json dan data/validation.json)
*Nama kolom embedding (misalnya question_embedding atau context_embedding)
*Parameter lain seperti TOP_K untuk pencarian FAISS
*Pastikan file ini telah disesuaikan sebelum melakukan training dan indexing.