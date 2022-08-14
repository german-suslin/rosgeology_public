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

# Список ошибок
range_ggkp_loss = 0.1
range_gk_loss = 1.5
range_pe_loss = 1
range_ds_loss = 50
range_dtp_loss = 20

errors = [range_ggkp_loss, range_gk_loss, range_pe_loss,
          range_ds_loss, range_dtp_loss]


# error_column_inx = [i for i in range(len(x_train[0]) - 3)]

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


class Generator(tensorflow.keras.utils.Sequence):
    '''
    Генератор батчей с x - x_len*batch_size и y - batch_size
    '''

    def __init__(self, x_data,
                 y_data_coll,
                 y_data_rest,
                 length,
                 batch_size,
                 x_columns,
                 y_columns,
                 only_colls=True):
        # Инициализируем и записывем переменные батча и сколько
        # понадобится предыдущих значений для предсказания, а также колонки, которые выбраны для анализа
        '''
        :param x_data: np.array
        Входные данные
        :param y_data_coll: np.array
        Данные по типу коллектора
        :param y_data_rest: np.array
        Данные по двум оставшимся показателям
        :param length: int
        Длина выборки для предсказания
        :param batch_size: int
        размер батча
        :param x_columns: list of ints [0-8]
        какие колонки задействуются
        :param y_columns: list of inst [0-3]
        пока не используется - но выбор колонок
        '''
        self.x_data = x_data[:, x_columns]
        self.y_data_coll = y_data_coll
        self.y_data_rest = y_data_rest
        self.x_columns = x_columns
        # self.y_columns = y_columns
        self.batch_size = batch_size
        self.length = length
        self.norm_fit = []
        self.norm_y_fit = []
        self.only_colls = only_colls

    def info(self):
        '''
        Вывод информации о датафрейма
        '''
        # print(self.df.head())
        print(f'\nРазмер: {self.x_data.shape, self.y_data_coll.shape, self.y_data_rest.shape}')

    def add_error(self, errors_indx, errors_value):
        # Добавляем ошибки к исходным данным
        for row in range(len(self.x_data)):
            for column in range(len(self.x_data[0])):
                self.x_data[row, column] = round(self.x_data[row, column] + \
                                                 random.uniform(-errors_value[column],
                                                                errors_value[column]), 3)
        print(self.x_data.shape)

    def normalize(self):
        # нормализация каждого столбца данных
        for i in range(len(self.x_columns)):
            xScaler = StandardScaler()
            xScaler.fit(self.x_data)
            self.norm_fit.append(xScaler)
            self.x_data = xScaler.transform(self.x_data)
        for i in range(2):
            yScaler = StandardScaler()
            yScaler.fit(self.y_data_rest)
            self.norm_y_fit.append(yScaler)
        # возвращает список нормализаторов, чтобы потом нормализировать
        # тестовые данные
        return self.norm_fit, self.norm_y_fit

    def normalize_test(self, norm_fit, norm_y_fit):
        for i in range(len(self.x_columns)):
            self.x_data = norm_fit[i].transform(self.x_data)
        for i in range(2):
            self.y_data_rest = norm_y_fit[i].transform(self.y_data_rest)

    def __get_data(self, x_batch, y_batch_coll, y_batch_rest):
        # Разбиваем наш батч на сеты
        # Определим максимальный индекс
        form = x_batch.shape[0] - self.length
        x = [x_batch[i:i + self.length] for i in range(form)]
        y_coll = [y_batch_coll[i] for i in range(form)]
        y_rest = [y_batch_rest[i] for i in range(form)]
        return np.array(x), np.array(y_coll), np.array(y_rest)

    def __len__(self):
        return self.x_data.shape[0] // (self.batch_size + self.length)

    def __getitem__(self, index):
        # Формирование выборки батчей
        # Берём значения от 0 до размера батча + длина выборки предсказания
        x_batch = self.x_data[index * self.batch_size:
                              (index + 1) * self.batch_size + self.length]

        y_batch_coll = self.y_data_coll[index * self.batch_size + self.length:
                                        (index + 1) * self.batch_size + self.length]

        y_batch_rest = self.y_data_rest[index * self.batch_size + self.length:
                                        (index + 1) * self.batch_size + self.length]
        print(y_batch_rest.shape, y_batch_coll.shape, x_batch.shape)
        x, y_coll, y_rest = self.__get_data(x_batch, y_batch_coll, y_batch_rest)

        if self.only_colls == True:
            return x, y_coll
        else:
            return x, y_rest
        # return x, [y_coll, y_rest]
