from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики


import random
from Generator import Generator
import tensorflow
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator  # для генерации выборки временных рядов
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate, \
    Input, Dense, Dropout, BatchNormalization, \
    Flatten, Conv1D, Conv2D, \
    LSTM  # Стандартные слои


import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


class Worker:
    def __init__(self, fname):
        '''
        Инициализация
        '''
        self.df = pd.read_csv(fname, decimal=',')

    def info(self):
        '''
        Вывод информации о датафрейма
        '''
        print(self.df.head())
        print(f'\nРазмер: {self.df.shape}')

    def get_y_collektors(self):
        '''
        Получение y_data для столбца "Коллекторы"
        '''
        # Преобразование в OHE
        enc = OneHotEncoder()
        y_data_coll = enc.fit_transform(
            self.df['Коллекторы'].values.reshape(-1, 1)
        ).toarray().astype(np.int16)
        y_data_rest = self.df[['KPEF', 'KNEF']].values.astype(np.float32)
        # y_data = np.concatenate((y_data_coll, y_data_rest), axis=1)
        print(f'Размер: {y_data_coll.shape, y_data_rest.shape}', self.df.columns)
        return y_data_coll, y_data_rest

    def get_x_data(self, columns):
        '''
        Получение x_data
        - columns - список столбцов вида ['GGKP_korr', 'GK_korr', 'DTP_korr']
        '''
        get_x_data = self.df[columns].values.astype(np.float32)
        print(f'Размер: {get_x_data.shape}')
        return get_x_data


train_worker = Worker('train.csv')
test_worker = Worker('test.csv')
train_worker.info()
columns = ['GGKP_korr', 'GK_korr', 'PE_korr', 'DS_korr', 'DTP_korr', 'Wi_korr', 'BK_korr', 'BMK_korr']
x_train = train_worker.get_x_data(columns)
y_train_coll, y_train_rest = train_worker.get_y_collektors()

x_test = test_worker.get_x_data(columns)
y_test_coll, y_test_rest = test_worker.get_y_collektors()

step = 0.001
# Список ошибок
range_ggkp_loss = 0.1
range_gk_loss = 1.5
range_pe_loss = 1
range_ds_loss = 50
range_dtp_loss = 20

lenght = 30
batch_size = 100
errors = [range_ggkp_loss, range_gk_loss, range_pe_loss,
          range_ds_loss, range_dtp_loss, 0, 0, 0]


error_column_inx = [0, 1, 2, 4, 5, 6, 7]
print(error_column_inx)
print(x_train.shape)
# проверка работы генератора - добавление ошибки и нормализация
Gen = Generator(x_train,
                y_train_coll,
                y_train_rest,
                lenght,
                batch_size=batch_size,
                x_columns=error_column_inx,
                y_columns=[0], only_colls=False)
Gen.add_error(error_column_inx, errors)
norm_fit, norm_y_fit = Gen.normalize()
print('norm fit',len(norm_fit))
Gen_test = Generator(x_test,
                     y_test_coll,
                     y_test_rest,
                     lenght,
                     batch_size=batch_size,
                     x_columns=error_column_inx,
                     y_columns=[0], only_colls=False)
Gen_test.add_error(error_column_inx, errors)
Gen_test.normalize_test(norm_fit, norm_y_fit)


data = []
answer = []

# Проверка выходных данных
for i in Gen:
    data.append(i[0])
    # answer.append(np.concatenate(i[1:],axis=1))
    answer.append(i[1])

# Приведение к массивам numpy
data = np.array(data)
answer = np.array(answer)
print(data.shape, answer.shape)

# Тестирование, в случае ошибок приведения формы
prev_shape = 0
for i in range(data.shape[0]):
    current_shape = len(data[i])
    if prev_shape != current_shape:
        print('shape', prev_shape, current_shape, i)
        prev_shape = current_shape
for i in range(len(answer)):
    current_shape = len(answer[i])
    if prev_shape != current_shape:
        print('shape _ answer', prev_shape, current_shape, i)
        prev_shape = current_shape