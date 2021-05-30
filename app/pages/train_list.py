import streamlit as st
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import datetime as dt
import os

def app():
    st.markdown(
        """
    Данное приложение предназначено для предсказания температуры, давления, скорости и направления ветра на основе данных с метео станций.  
    """
    )

    st.markdown(
        """
        ## Алушта 2021.29.05 20:08 ##  
        ### Прогноз температуры ###
    """
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.array([dt.datetime(2013, 9, i).strftime("%Y-%m-%d") for i in range(1,6)], 
            dtype='datetime64')
    x = x.astype(dt.datetime)
    ax.plot(x, [1,2,3,4,10])
    ax.plot(x, [2,2,2,4,10])
    ax.legend(['Температура','Предсказание'])
    xfmt = mdates.DateFormatter('%m.%d')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel('Дата')
    ax.set_ylabel('Температура (градус Цельсия)')
    plt.savefig('data/plots/alushta/t.png')

    image = Image.open('data/plots/alushta/t.png')
    st.image(image, caption='Test score: 4.61 RMSE', width=750)

    st.markdown(
        """
        ### Прогноз давления ###
    """
    )

    image = Image.open('data/plots/alushta/p.jpg')
    st.image(image, caption='Test score: 2.58 RMSE')

    st.markdown(
        """
        ## Симферополь 2021.29.05 20:12 ##  
        ### Прогноз температуры ###
    """
    )

    for root, dirs, files in os.walk("./data/models/meta/"):  
        for filename in files:
            st.markdown(filename)
