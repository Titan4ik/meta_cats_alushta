import streamlit as st
import numpy as np
import pandas as pd
from state import session_state
from pathlib import Path
from datetime import datetime
from io import BytesIO
import time
import json

def app():
    st.markdown(
        """
    ## Пример загружаеммого dataset'a:
    """
    )

    df = pd.DataFrame(
        np.random.randn(3, 7),
        columns=('col %d' % i for i in range(7)))
    
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
        if df is not None:
            st.info('Сохранено')
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

    data = uploaded_file.read()
    try:
        df = pd.read_csv(BytesIO(data), sep="\t", encoding='utf-8')
    except Exception as e:
        st.error('Dataset должен соответсвовать формату')
        return None
        
    column = {'published', 'domain'}
    if not column.issubset(df.columns):
        st.error('Данные dataset\'a не соответсувуют формату')
        return None

    df.to_csv(path, sep='\t', encoding='utf-8')

    path_meta = Path(f'data/datasets/meta/{station_name}.meta.json')
    path_meta.parent.mkdir(exist_ok=True)
    with open(path_meta, 'w') as f:
        meta = {'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'row_counts': df.shape[0]}
        f.write(json.dumps(meta))

    return df
