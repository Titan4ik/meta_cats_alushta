import streamlit as st
from app.pages import train_app, train_list_app, prophet_app
from app.pages.utils import Page, render_sidebar_pages
from state import session_state

def main():
    st.sidebar.markdown('<br>', unsafe_allow_html=True)
    pages = [
        Page('# Обучить нейросеть ', train_app),
        Page('# Обученные модели ', train_list_app),
    ]
    render_sidebar_pages(pages, session_state=session_state)
    st.sidebar.markdown('<br>', unsafe_allow_html=True)

    st.sidebar.markdown('<hr>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()

