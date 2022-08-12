from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики


import random

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


errors = [range_ggkp_loss, range_gk_loss, range_pe_loss,
          range_ds_loss, range_dtp_loss]


error_column_inx = [i for i in range(len(x_train[0]) - 3)]




def adding_error(x: np.array, columns: list, errors: list) -> np.array:
    '''
    Добавляет ошибку к исходным данным и возвращает их
    :param x: входной массив
    :param columns: индексы колонок к которым применяются ошибки
    :param errors: ошибки колонок
    :return: массив данных с ошибками из списка
    '''
    for row in range(len(x)):
        for column in columns:
            x[row][column] = round(x[row][column] + \
                                   random.uniform(-errors[column],
                                                  errors[column]), 3)
    return x

# Добавление ошибки в исходные данные
x_train = adding_error(x_train,
                       error_column_inx,
                       errors)

x_test = adding_error(x_test,
                      error_column_inx,
                      errors)

# Нормировка данных
xScaler = StandardScaler()
xScaler.fit(x_train)
x_train = xScaler.transform(x_train)
x_test = xScaler.transform(x_test)

yScaler = StandardScaler()
yScaler.fit(y_train_rest)
y_train_rest = yScaler.transform(y_train_rest)
y_test_rest = yScaler.transform(y_test_rest)

# print(type(x_train), type(y_train))
# print(x_train.shape)
# print(x_train[:5])
# print(y_train_rest[:3])
# print(y_train_coll[:3])
x_len = 50
batch_size = 40


# Функция разделения набора данных на выборки для обучения нейросети
def split_sequence(x_data,
                   y_data_coll,
                   y_data_rest,
                   seq_len,
                   predict_lag=0):
    '''
    :param x_data: набор входных данных
    :param y_data_coll: набор выходных данных
    :param y_data_rest: набор выходных данных
    :param seq_len: длина серии (подпоследовательности) входных данных для анализа
    :param predict_lag: количество шагов в будущее для предсказания
    :return:
    '''
    # Определение максимального индекса начала подпоследовательности
    xlen = len(x_data) - seq_len - (predict_lag - 1)
    # Формирование подпоследовательностей входных данных
    x = [x_data[i:i + seq_len] for i in range(xlen)]
    # Формирование меток выходных данных,
    # отстоящих на predict_lag шагов после конца подпоследовательности
    y_coll = [y_data_coll[i + seq_len + predict_lag - 1]
              for i in range(xlen)]
    y_rest = [y_data_rest[i + seq_len + predict_lag - 1]
              for i in range(xlen)]
    # Возврат результатов в виде массивов numpy
    return np.array(x), np.array(y_coll), np.array(y_rest)

print(x_train.shape, y_train_rest.shape, y_train_coll.shape)

x_train, y_train_coll, y_train_rest = split_sequence(x_train,
                                                     y_train_coll,
                                                     y_train_rest,
                                                     seq_len=x_len,
                                                     predict_lag=0)
x_test, y_test_coll, y_test_rest = split_sequence(x_test,
                                                  y_test_coll,
                                                  y_test_rest,
                                                  seq_len=x_len,
                                                  predict_lag=0)
# x_train = x_train.reshape(-1, 1,x_train.shape[1])
# x_test = x_test.reshape(-1, 1, x_test.shape[1])
print(x_train.shape, y_train_rest.shape, y_train_coll.shape)

input_model = Input(shape=(x_len, x_train.shape[2]))
dense = Dense(1024, activation='relu')(input_model)
lstm = LSTM(1024, return_sequences=False)(input_model)
lstm = Dropout(0.3)(lstm)
lstm = BatchNormalization()(lstm)
# lstm = LSTM(256, return_sequences=True)(lstm)
flatten = Flatten()(lstm)
# output_rest = Dense(y_train_rest.shape[1], activation='linear')(flatten)
output_coll = Dense(3, activation='softmax')(lstm)

model_name = 'test_coll_lstm_n50'
folder = 'data/'
model_folder = folder + 'models/'
graph_folder = folder + 'graphs/'

model = Model(input_model, output_coll, name=model_name)
model.summary()
model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=Adam(learning_rate=1e-5))
epochs = 20
history = model.fit(x_train, y_train_coll,
                        epochs=epochs,
                        verbose=1,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test_coll))
plt.subplot(2, 2, 1)
plt.title(label='ошибка')
plt.plot(history.history['val_loss'][6:], label='Test')
plt.plot(history.history['loss'][6:], label='Train')
model.save(model_folder + model_name)
plt.legend()
plt.subplot(1, 2, 2)
plt.title(label='точность')
plt.plot(history.history['val_accuracy'], label='Test')
plt.plot(history.history['accuracy'],label='Train')
plt.savefig(graph_folder + model_name + '.jpg')
plt.show()
