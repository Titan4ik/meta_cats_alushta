import streamlit as st

def app():
    st.markdown(
        """
    Выберите обученную модель.  
    """
    )

    option = st.selectbox(
        'Выберите обученную модель',
        ('Алушта 2021.29.05 20:08', 'Симферополь'))

    predict_date = st.date_input("Выбирите дату прогноза" )

    st.button("Получить прогноз")