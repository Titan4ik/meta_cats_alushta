import streamlit as st
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import datetime as dt
import os
import json
models_path = 'C:\\alushta\\meta_cats_alushta\\data\\models\\'
def app():
    st.markdown(
        """
    Данное приложение предназначено для предсказания температуры, давления, скорости и направления ветра на основе данных с метео станций.  
    """
    )
    arr =  [None]
    for root, dirs, files in os.walk("./data/datasets/meta/"):
        for filename in files:
            arr.append(filename.replace('.meta.json', ''))
    
    model = st.selectbox('Обученные модели', arr)

    if model:
        st.markdown(model)

        st.markdown(
            f"""
            ## {model} ##  
        """
        )
        with open(f'./data/datasets/meta/{model}.meta.json', 'r') as fp:
            data = json.load(fp)
        
        for col in data['columns']:
            st.markdown(
                f"""
                ### Прогноз {col} ###
            """
            )
            p = os.path.join(models_path, model, f'{col}.png')
            image = Image.open(p)
            st.image(image, caption='', width=750)

            p = os.path.join(models_path, model, f'{col}_diff.png')
            image = Image.open(p)
            st.image(image, caption='', width=750)

            


