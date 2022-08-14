from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики
from sklearn.model_selection import train_test_split

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

print('before concat:', x_train.shape, x_test.shape, y_train_coll.shape, y_test_coll.shape,
      y_train_rest.shape, y_test_rest.shape)
# x_data = np.concatenate([x_train, x_test], axis=0)
# y_data_colls = np.concatenate([y_train_coll, y_test_coll], axis=0)
# y_data_rest = np.concatenate([y_train_rest, y_test_rest], axis=0)
# x_train, x_test, y_train_coll, y_test_coll = train_test_split(x_data, y_data_colls, test_size=0.3, shuffle=True)
# x_train, \
# x_test, \
# y_train_rest, \
# y_test_rest, \
# y_train_rest, \
# y_test_rest = train_test_split(x_data,
#                                [y_data_colls,
#                                 y_data_rest],
#                                test_size=0.3,
#                                shuffle=True)
print('after concat:', x_train.shape, x_test.shape, y_train_coll.shape, y_test_coll.shape,
      y_train_rest.shape, y_test_rest.shape)

step = 0.001
# Список ошибок
range_ggkp_loss = 0.1
range_gk_loss = 1.5
range_pe_loss = 1
range_ds_loss = 50
range_dtp_loss = 20

lenght = 10
batch_size = 50
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
print('norm fit', len(norm_fit))
Gen_test = Generator(x_test,
                     y_test_coll,
                     y_test_rest,
                     lenght,
                     batch_size=batch_size,
                     x_columns=error_column_inx,
                     y_columns=[0], only_colls=False)
Gen_test.add_error(error_column_inx, errors)
Gen_test.normalize_test(norm_fit, norm_y_fit)

input_model = Input(shape=(lenght, len(error_column_inx)))
lstm = Dense(1024, activation='relu')(input_model)
# lstm = LSTM(256, return_sequences=True)(input_model)
lstm = Dropout(0.1)(lstm)
lstm = BatchNormalization()(lstm)
# lstm = LSTM(256, return_sequences=True)(lstm)
flatten = Flatten()(lstm)
output_rest = Dense(2, activation='linear')(flatten)
output_coll = Dense(3, activation='softmax')(flatten)

model_name = 'test_coll_lstm_n50'
folder = 'data/'
model_folder = folder + 'models/'
graph_folder = folder + 'graphs/'
# test colls
# model = Model(input_model, output_coll, name=model_name)
# model.summary()
# model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=Adam(learning_rate=1e-5))
# epochs = 100
# history = model.fit(Gen,
#                         epochs=epochs,
#                         verbose=1,
#                         batch_size=batch_size,
#                         validation_data=Gen_test)
# test rest
model = Model(input_model, output_rest, name=model_name)
model.summary()
model.compile(loss="mse", metrics=['accuracy'], optimizer=Adam(learning_rate=1e-5))
epochs = 20
history = model.fit(Gen,
                    epochs=epochs,
                    verbose=1,
                    batch_size=batch_size,
                    validation_data=Gen_test)

plt.subplot(2, 2, 1)
plt.title(label='ошибка')
plt.plot(history.history['val_loss'], label='Test')
plt.plot(history.history['loss'], label='Train')
model.save(model_folder + model_name)
plt.legend()
plt.subplot(1, 2, 2)
plt.title(label='точность')
plt.plot(history.history['val_accuracy'], label='Test')
plt.plot(history.history['accuracy'], label='Train')
plt.legend()
plt.savefig(graph_folder + model_name + '.jpg')
plt.show()
