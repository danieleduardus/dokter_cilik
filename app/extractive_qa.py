import streamlit as st
from transformers import pipeline
import yaml

# Muat konfigurasi dari file YAML
@st.cache_resource
def load_config():
    with open('cfg/config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config = load_config()

PIPELINE_NAME = config["Config"]["PIPELINE_NAME"]
FINETUNED_MODEL_NAME = config["Config"]["FINETUNED_MODEL_NAME"]

# Inisialisasi pipeline Q&A dengan caching sebagai resource (agar tidak di-hash)
@st.cache_resource
def load_qa_pipeline():
    return pipeline(PIPELINE_NAME, model=FINETUNED_MODEL_NAME)

qa_pipeline = load_qa_pipeline()

# Fungsi untuk mengekstrak jawaban, hanya melakukan transformasi data sehingga bisa di-cache dengan st.cache_data
@st.cache_data
def extract_answer(question, context):
    result = qa_pipeline(question=question, context=context)
    # Jika pipeline mengembalikan list, ambil elemen pertama
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    return result

def render():
    col1, col2 = st.columns(2)
    with col1:
        with st.form("form1", clear_on_submit=False):
            context = st.text_area("Enter your context here")
            question = st.text_input("Enter your question here")
            submit = st.form_submit_button("Submit", type="primary")
            
            if submit:
                with col2:
                    st.success("Done!")
                    result = extract_answer(question, context)
                    answer = result.get("answer", "No answer found")
                    score = result.get("score", 0)
                    
                    st.subheader("Answer:")
                    st.info(f"🤖 {answer}")
                    
                    st.subheader("Score:")
                    st.info(score)
