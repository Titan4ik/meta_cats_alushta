import pandas as pd
import math
df = pd.read_csv('in.csv')
df['timestamp'] = pd.to_datetime(df.timestamp, format='%d.%m.%Y %H:%M')
df.sort_values('timestamp',ascending=True)
df.index = df['timestamp']
df = df.reindex(index=df.index[::-1])
df.drop('timestamp', axis=1, inplace=True)

df.to_csv('tmp.csv')
num = [
    [['запад', 'север'], 315 * math.pi / 180],
    [['запад', 'юг'], 255 * math.pi / 180],
    [['восток', 'север'], 45 * math.pi / 180],
    [['восток', 'юг'], 135 * math.pi / 180],
    [['север'], 0],
    [['юг'], 180  * math.pi / 180],
    [['запад'], 270 * math.pi / 180],
    [['восток'], 90 * math.pi / 180],
    [['безветрие'], -1],
]
import csv
import datetime as dt
prev_row = None
with open('tmp.csv', 'r', encoding='utf-8') as fin:
    with open('out.csv', 'w', encoding='utf-8') as fout:
        read = csv.reader(fin)
        wr = csv.writer(fout)
        for i, row in enumerate(read):

            if i == 0:
                wr.writerow(row)
            else:
                for i in range(1,8):
                    if not row[i] and prev_row is not None:
                        row[i] = prev_row[i]
                    elif not row[i]:
                        row[i] = 0
                    if i == 6:
                        try:
                            s = row[i].lower()
                        except:
                            ...
                        else:
                            for arg, val in num:
                                t = True
                                for a in arg:
                                    if a not in s:
                                        t = False
                                        break
                                if t:
                                    row[i] = val
                                    break
                    try:
                        row[i] = float(row[i].replace(',', '.'))
                    except:
                        ...
                wr.writerow(row)
            prev_row = row