n_window = 30
models_path = 'C:\\alushta\\meta_cats_alushta\\data\\models\\'
csvs = 'C:\\alushta\\meta_cats_alushta\\data\\datasets\\csv\\'
images = 'C:\\alushta\\meta_cats_alushta\\data\\plots\\'
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
import matplotlib.patches as mpatches
import math
from sklearn.metrics import mean_squared_error
import datetime as dt
from pickle import dump,load
import os
import json
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
def construct_df(path):
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        raise Exception('column timestamp not found in csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    st_date, end1_dt, end2_dt =(
        df['timestamp'].values[0],
        df['timestamp'].values[int(len(df)*0.8)],
        df['timestamp'].values[-1],
    )

    df.index = df['timestamp']
    data = df.sort_index(ascending=True, axis=0)
    # columns = df.columns
    data.drop('timestamp', axis=1, inplace=True)
    return data, df, st_date, end1_dt, end2_dt

    
# new_data = data

def get_in_data(train, valid):
    x_train,y_train,x_test,y_test = [],[],[],[]
    for i in range(n_window,train.shape[0]):
        x_train.append(train[i-n_window:i, 0])
        y_train.append(train[i, 0])

    for z in range(n_window,valid.shape[0]):
        x_test.append(valid[z-n_window:z, 0])
        y_test.append(valid[z, 0])



    x_train, y_train,x_test,y_test = np.array(x_train), np.array(y_train),np.array(x_test),np.array(y_test)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    return x_train,y_train,x_test,y_test


def get_model(x_train,y_train,x_test,y_test):
    model = Sequential()
    model.add(LSTM(units=150,input_shape=(x_train.shape[1],1),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer="adam")

    history = model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=1)
    return model, history

def start(dataset_name, new_data_, df, col,st_date, end1_dt, end2_dt,day_count=10):
    new_data = pd.DataFrame(index=range(0,len(df)),columns=[col])
    for j, val in enumerate(new_data_[col]):
        new_data[col][j] = val
    dataset = new_data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train= scaled_data[:int(df.shape[0]*0.8)]
    valid = scaled_data[int(df.shape[0]*0.8):]


    x_train,y_train,x_test,y_test = get_in_data(train, valid)
    model, _ = get_model(x_train,y_train,x_test,y_test)
    model.save(f'{models_path}/{dataset_name}/model.{col}')
    dump(scaler, open(os.path.join(models_path, dataset_name, f'scaler_{col}.pkl'), 'wb'))


    count_pred = day_count * 8
    prediction = model.predict(x_test)

    y_pred = scaler.inverse_transform(prediction)

    

    y_pred = np.reshape(y_pred, [1, -1])

    draw(
        dataset_name,
        col, end1_dt, np.timedelta64(3, 'h'),
        new_data[int(df.shape[0]*0.8)-count_pred:int(df.shape[0]*0.8)][col],
        new_data[int(df.shape[0]*0.8):int(df.shape[0]*0.8)+count_pred][col],
        y_pred[0][:count_pred]
    )

    res = []
    values = []
    for x1, x2 in zip(y_pred[0],new_data[int(df.shape[0]*0.8):][col].values):
        if np.isnan(x1) or np.isnan(x2):
            continue
        values.append(abs(x1-x2))
    for  k in [3,10,30,90,120,180,240, 300,360]:
        in_days = k * 8
        testScore = sum(values[:in_days]) / in_days
        res.append((k, testScore))
        print(f'Diff in {col} for {k} days: {testScore}')
    draw_info(dataset_name, col, res)
    return res
    
def draw_info(dataset_name, name, values):
    fig, ax = plt.subplots()

 

    lab= f"Размер погрешности предикта для {name} с течением времени"
    plt.title(label=lab)
    ax.set_xlabel('кол-во дней')
    ax.set_ylabel('DIFF')

    plt.plot([x[0] for x in values], [x[1] for x in values], color='forestgreen',label="Разница между предиктом и факт. значением")


    plt.legend()
    plt.grid('both')
    # plt.show()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(models_path, dataset_name, f'{name}_diff.png'), dpi=100)
def draw(dataset_name, name, start_dt_test,  step, train, test, predict):
    fig, ax = plt.subplots()
    print(f'-----{name}-----')
    y_train = [
        start_dt_test - step * i
        for i in range(len(train))
    ][::-1]

    y_test = [
        start_dt_test + step * i
        for i in range(len(test))
    ]

    ax.plot(y_train, [x for x in train], color='seagreen',label="train")
    ax.plot(y_test, [x for x in test], color='forestgreen', label="test")
    ax.plot(y_test, [x for x in predict], color='coral', label="predict")
    ax.set_xlabel('Дата')
    ax.set_ylabel(name)
    ax.xaxis_date()

    fig.autofmt_xdate()
    lab= f"Фактические показатели и предикт для {name}"
    plt.title(label=lab)

    ax.legend()
    plt.grid('both')
    # plt.show()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(models_path, dataset_name, f'{name}.png'), dpi=100)

# from tensorflow import keras
# model = keras.models.load_model('1.model')
def run(dataset_name):
    file_name = dataset_name + '.csv'
    new_data_, df,st_date, end1_dt, end2_dt = construct_df(csvs + file_name)
    cols = []
    for i, col in enumerate(df.columns[1:]):
        try:
            start(dataset_name, new_data_, df, col,st_date, end1_dt, end2_dt)
        except Exception as e:
            print(f"ERROR! {e}")
        else:
            cols.append(col)
    path_meta = Path(f'data/datasets/meta/{dataset_name}.meta.json')
    path_meta.parent.mkdir(exist_ok=True)
    with open(path_meta, 'w') as f:
        meta = {
            'time': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'columns': cols,
        }
        f.write(json.dumps(meta))

# dataset_name = 'stovropol'
# file_name = '1.csv'

# run(dataset_name,file_name)

# scaler = load(open(f'{scalers_path}/{dataset_name}/scaler_{col}.pkl', 'rb'))