import streamlit as st
# Set konfigurasi halaman streamlit dengan layout wide
st.set_page_config(layout="wide")

import app.sidebar as sidebar
import app.extractive_qa as extractive_qa
import app.generative_qa as generative_qa

# Tampilkan sidebar dan ambil pilihan halaman
page = sidebar.show()

# Sembunyikan footer bawaan Streamlit
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Judul dan deskripsi halaman utama
st.title("Dokter Cilik Virtual")
st.caption("Sistem Chatbot menggunakan BERT dan FAISS")

# Footer custom
footer = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
a:link, a:visited{
  color: blue;
  background-color: transparent;
  text-decoration: underline;
}
a:hover, a:active {
  color: red;
  background-color: transparent;
  text-decoration: underline;
}
.footer {
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background-color: white;
  color: black;
  text-align: center;
}
</style>
<div class="footer">
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

# Panggil halaman sesuai pilihan pada sidebar
if page == "Extractive Q&A":
    extractive_qa.render()
elif page == "Mulai Chat":
    generative_qa.render()
