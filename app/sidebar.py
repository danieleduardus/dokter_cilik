import streamlit as st
from streamlit_option_menu import option_menu
import base64

def show():
    # 1. Ubah sidebar menjadi kontainer relative, setinggi layar (100vh), tanpa padding
    st.markdown(
        """
        <style>
        /* Hilangkan margin/padding default pada kontainer sidebar */
        [data-testid="stSidebar"] > div:first-child {
            position: relative;
            height: 100vh;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden; /* atau auto jika Anda butuh scrollbar */
        }
        /* Kelas untuk menempatkan elemen di bawah secara absolut */
        .bottom-image {
            position: absolute;
            margin-top:300px !important;
            left: 0;
            width: 100%;
            margin: 0;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 2. Baca file gambar, konversi ke base64 (agar path tidak bermasalah)
    with open("images/logo.png", "rb") as f:
        data = f.read()
    encoded_img = base64.b64encode(data).decode("utf-8")

    with st.sidebar:
        st.markdown("## Menu")

        selected = option_menu(
            menu_title=None,
            options=["Mulai Chat"],
            icons=["card-text"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#FFFFFF"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                    "color": "#000000"
                },
                "nav-link-selected": {
                    "background-color": "#0dcaf0",
                    "color": "white"
                }
            }
        )

        # 3. Sisipkan gambar di bawah
        st.markdown(
            f"""
            <div class="bottom-image">
                <img src="data:image/png;base64,{encoded_img}" style="max-width:100%; margin-bottom:0;" alt="logo" />
            </div>
            """,
            unsafe_allow_html=True
        )

    return selected
