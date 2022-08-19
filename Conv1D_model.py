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
    LSTM, MaxPooling1D  # Стандартные слои

import matplotlib.pyplot as plt

from Generator import Generator, Worker
import pandas as pd
import numpy as np

# df = pd.read_csv('train_3.csv')
# print(df['Коллекторы'].unique())
# # df[df['Коллекторы']]
# df.loc[df['Коллекторы']==75, 'Коллекторы'] = 2
# print(df['Коллекторы'].unique())
# df.to_csv('train_3corr.csv', index=False)
# df = pd.read_csv('train_3corr.csv')
# print(df.head())

# Подгрузка данных
train_worker1 = Worker('train.csv')
train_worker2 = Worker('train_2.csv')
train_worker3 = Worker('train_3corr.csv')
test_worker = Worker('test.csv')
print('worker info')
train_worker1.info()
train_worker2.info()
train_worker3.info()
columns = ['GGKP_korr', 'GK_korr', 'PE_korr', 'DS_korr', 'DTP_korr', 'Wi_korr', 'BK_korr', 'BMK_korr']
print('get data')
x_train1 = train_worker1.get_x_data(columns)
y_train_coll1, y_train_rest1 = train_worker1.get_y_collektors()

x_train2 = train_worker2.get_x_data(columns)
y_train_coll2, y_train_rest2 = train_worker2.get_y_collektors()

x_train3 = train_worker3.get_x_data(columns)
y_train_coll3, y_train_rest3 = train_worker3.get_y_collektors()

x_val = test_worker.get_x_data(columns)
y_val_coll, y_val_rest = test_worker.get_y_collektors()

# Объединение и разбиение на обучающую и проверочную выборки
print('before concat:', x_train1.shape,
      x_train2.shape,
      x_train3.shape,
      y_train_coll1.shape,
      y_train_coll2.shape,
      y_train_coll3.shape)
print(y_train_coll3[:2])

x_data = np.concatenate([x_train1, x_train2, x_train3], axis=0)
y_data_colls = np.concatenate([y_train_coll1, y_train_coll2, y_train_coll3], axis=0)
y_data_rest = np.concatenate([y_train_rest1, y_train_rest2, y_train_rest3], axis=0)
y_data = np.concatenate([y_data_colls, y_data_rest], axis=1)
x_test, x_train, y_test, y_train = train_test_split(x_data, y_data, train_size=0.7, shuffle=False)
y_train_coll = y_train[:,:3]
y_test_coll = y_test[:,:3]
y_train_rest = y_train[:, 3:]
y_test_rest = y_test[:, 3:]

print('after concat:', x_train.shape,
      x_test.shape,
      y_train_coll.shape,
      y_test_coll.shape,
      y_train_rest.shape,
      y_test_rest.shape)

# Список ошибок
range_ggkp_loss = 0.1
range_gk_loss = 1.5
range_pe_loss = 1
range_ds_loss = 50
range_dtp_loss = 20

errors = [range_ggkp_loss, range_gk_loss, range_pe_loss, range_dtp_loss, 0, 0, 0]
x_columns = [0, 1, 2, 4, 5, 6, 7]
error_column_inx = [0, 1, 2, 4, 5, 6, 7]
print(error_column_inx)


# Параметры данных и эпохи обучения модели
lenght = 20
batch_size = 100
epochs = 100


# Создание генератора, нормализация данных
Gen = Generator(x_train,
                y_train_coll,
                y_train_rest,
                lenght,
                batch_size=batch_size,
                x_columns=x_columns,
                y_columns=[0], only_colls=True)
Gen.add_error(error_column_inx, errors)
norm_fit, norm_y_fit = Gen.normalize()
print('norm fit', len(norm_fit))
Gen_test = Generator(x_test,
                     y_test_coll,
                     y_test_rest,
                     lenght,
                     batch_size=batch_size,
                     x_columns=x_columns,
                     y_columns=[0], only_colls=True)
Gen_test.add_error(error_column_inx, errors)
Gen_test.normalize_test(norm_fit, norm_y_fit)

# Создание модели
input_model = Input(shape=(lenght, len(x_columns)))
conv = Conv1D(512, 3, activation='relu', padding='same')(input_model)
batch_normalized1 = BatchNormalization()(conv)
max_pool = MaxPooling1D()(batch_normalized1)
flatten = Flatten()(max_pool)
# dropout1 = Dropout(0.1)(batch_normalized1)
# flatten = Flatten()(dropout1)
# dense = Dense(1024, activation='linear')(flatten)
# batch_normalized2 = BatchNormalization()(dense)
# dropout2 = Dropout(0.3)(batch_normalized2)
output_coll = Dense(3, activation='softmax')(flatten)

# Путь сохранения модели и графиков
model_name = 'Conv1d_simple_n20'
folder = 'data/'
model_folder = folder + 'models/'
graph_folder = folder + 'graphs/'

# Компиляция модели
model = Model(input_model, output_coll, name=model_name)
model.summary()
model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=Adam(learning_rate=1e-5))

if __name__ == '__main__':
# Обучение модели
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





