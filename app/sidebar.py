import streamlit as st
from streamlit_option_menu import option_menu

def show():
    with st.sidebar:
        st.markdown("""
                    # Menu
                    """, unsafe_allow_html = False)
        selected = option_menu(
            menu_title = None, #required
            
            options = ["Mulai Chat"], #required
            icons = ["card-text"], #optional
            
            menu_icon="cast", #optional
            default_index = 0, #optional
        )
        return selected