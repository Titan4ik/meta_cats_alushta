import streamlit as st
import numpy as np
import pandas as pd
from state import session_state
from pathlib import Path
from datetime import datetime
from io import BytesIO
import time
import json
from api_trainer import run
import threading

def app():
    st.markdown(
        """
    ## Пример загружаеммого dataset'a:
    """
    )

    df = pd.DataFrame(
        columns=(["timestamp",  "T",  "Po",  "U",  "DD",  "Ff"])).set_index('timestamp', drop=False)
    df.loc[0] = ["30.05.2021 00:00", "17.9",  "749.8",  "94",  "Ветер, дующий с северо-востока",  "1"]
    df.loc[1] = ["29.05.2021 21:00", "19.1",  "748.9",  "86",  "Ветер, дующий с востоко-северо-востока",  "2"]
    
    st.dataframe(df)

    st.markdown(
        """
    ## Ввод данных:
    """
    )

    station_name = st.text_input("Название станции")
    dataset = st.file_uploader("Загрузите dataset", type=['csv'])
    train_button = st.button("Обучить")
    
    if train_button:
        df = _save_dataset(station_name, dataset)
        if df:
            st.info('Сохранено')
            t1 = threading.Thread(target=run, args=(station_name,))
            t1.start()
            print(f'start train {station_name}')
            time.sleep(1)
            session_state.subpage = None
            st.experimental_rerun()
            


def _save_dataset(station_name, uploaded_file):
    if not station_name:
        st.error('Необходимо заполнить название станции')
        return None

    if not uploaded_file:
        st.error('Необходимо загрузить dataset')
        return None
    
    path = Path(f'data/datasets/csv/{station_name}.csv')
    path.parent.mkdir(exist_ok=True)
    
    with open(path,'wb') as out:
        data = uploaded_file.read()
        out.write(BytesIO(data).read())

    
    return True
