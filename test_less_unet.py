from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Нормировщики
from sklearn.model_selection import train_test_split

import random
import tensorflow
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate, \
    Input, Dense, Dropout, BatchNormalization, \
    Flatten, Conv1D, Conv2D, \
    LSTM, MaxPooling1D, Activation, UpSampling1D  # Стандартные слои

import matplotlib.pyplot as plt

from Generator import Generator, Worker, accuracy_calculate
import pandas as pd
import numpy as np


# Подгрузка данных
train_worker0 = Worker('df_norm_ts.csv')
train_worker1 = Worker('train.csv')
train_worker2 = Worker('train_2.csv')
train_worker3 = Worker('train_3corr.csv')
test_worker = Worker('test.csv')

print('worker info')
train_worker0.info()
train_worker1.info()
train_worker2.info()
train_worker3.info()

columns = ['GGKP_korr', 'GK_korr', 'PE_korr', 'DS_korr', 'DTP_korr', 'Wi_korr', 'BK_korr', 'BMK_korr']
x_data = train_worker0.get_x_data(columns)
y_data_colls, y_data_rest = train_worker0.get_y_collektors()
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
print('before concat:', x_data.shape,
      y_data_colls.shape,
      y_data_rest.shape)
# print('before concat:', x_train1.shape,
#       x_train2.shape,
#       x_train3.shape,
#       y_train_coll1.shape,
#       y_train_coll2.shape,
#       y_train_coll3.shape)
# x_data = np.concatenate([x_train1, x_train2, x_train3], axis=0)
# y_data_colls = np.concatenate([y_train_coll1, y_train_coll2, y_train_coll3], axis=0)
# y_data_rest = np.concatenate([y_train_rest1, y_train_rest2, y_train_rest3], axis=0)
x_val = np.concatenate([x_val, y_val_coll], axis=1)
x_data = np.concatenate([x_data, y_data_colls], axis=1)
y_data = np.concatenate([y_data_colls, y_data_rest], axis=1)
x_test, x_train, y_test, y_train = train_test_split(x_data, y_data, train_size=0.2, shuffle=False)
y_train_coll = y_train[:,:3]
y_test_coll = y_test[:,:3]
y_train_rest = y_train[:, 3].reshape(-1, 1)
y_test_rest = y_test[:, 3].reshape(-1, 1)
y_val_rest = y_val_rest[:, 1].reshape(-1, 1)

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

errors = [range_ggkp_loss, range_gk_loss, range_pe_loss, range_dtp_loss, 0, 0, 0, 0, 0, 0, 0]
x_columns = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
norm_columns = [0, 1, 2, 3, 4, 5, 6]
error_column_inx = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
print(error_column_inx)


# Параметры данных и эпохи обучения модели
lenght = 16
batch_size = 500
epochs = 40
train_state = 'unet' # parallel or consistent

# Создание генератора, нормализация данных
Gen = Generator(x_train,
                y_train_coll,
                y_train_rest,
                lenght,
                batch_size=batch_size,
                x_columns=x_columns,
                y_columns=[0], only_colls=False)
# Gen.add_error(error_column_inx, errors)
norm_fit, norm_y_fit = Gen.normalize(columns=norm_columns)
print('norm fit', len(norm_fit), len(norm_y_fit))
Gen_test = Generator(x_test,
                     y_test_coll,
                     y_test_rest,
                     lenght,
                     batch_size=batch_size,
                     x_columns=x_columns,
                     y_columns=[0], only_colls=False)
# Gen_test.add_error(error_column_inx, errors)
Gen_test.normalize_test(norm_fit, norm_y_fit)
Gen.info()
Gen_test.info()


Gen_val = Generator(x_val,
                    y_val_coll,
                    y_val_rest,
                    lenght,
                    batch_size=len(x_val)-lenght,
                    x_columns=x_columns,
                    y_columns=[0], only_colls=False)
Gen_val.add_error(error_column_inx, errors)
Gen_val.normalize_test(norm_fit, norm_y_fit)
x_val_data = []
y_val_data = []
for x in Gen_val:
    x_val_data.append(x[0])
    y_val_data.append(x[1])
x_val_data = np.array(x_val_data)
y_val_data = np.array(y_val_data)
print('validation data shape', x_val_data.shape, y_val_data.shape)




# Создание модели (64, 32 / 128, 64 - большая выборка)
if train_state == 'consistent':
    input_model = Input(shape=(lenght, len(x_columns)))
    conv = Conv1D(128, 5, activation='relu', padding='same')(input_model)
    batch_normalized1 = BatchNormalization()(conv)
    max_pool = MaxPooling1D()(batch_normalized1)
    conv2 = Conv1D(64, 5, activation='relu', padding='same')(max_pool)
    batch_normalized2 = BatchNormalization()(conv2)
    max_pool = MaxPooling1D()(batch_normalized2)
    last_layer = Flatten()(max_pool)

# Создание модели
if train_state == 'parallel':
    input_model = Input(shape=(lenght, len(x_columns)))
    conv = Conv1D(256, 2, activation='relu', padding='same')(input_model)
    batch_normalized1 = BatchNormalization()(conv)
    # batch_normalized1 = Dropout(0.1)(batch_normalized1)
    max_pool1 = MaxPooling1D()(batch_normalized1)
    flatten1 = Flatten()(max_pool1)
    conv2 = Conv1D(128, 2, activation='relu', padding='same')(input_model)
    batch_normalized2 = BatchNormalization()(conv2)
    # batch_normalized2 = Dropout(0.1)(batch_normalized2)
    max_pool2 = MaxPooling1D()(batch_normalized2)
    flatten2 = Flatten()(max_pool2)
    conv3 = Conv1D(64, 2, activation='relu', padding='same')(input_model)
    batch_normalized3 = BatchNormalization()(conv3)
    # batch_normalized3 = Dropout(0.1)(batch_normalized3)
    max_pool3 = MaxPooling1D()(batch_normalized3)
    flatten3 = Flatten()(max_pool3)
    concat = concatenate([flatten1, flatten2, flatten3])
    last_layer = Dense(512, activation = 'relu')(concat)
    last_layer = Dropout(0.3)(last_layer)

# unet
if train_state == 'unet':
    # def lstm_unet(input_shape=(88, 120, 3)):

    input_model = Input(shape=(lenght, len(x_columns)))  # Создаем входной слой с размерностью input_shape
    # Block 1
    # lstm_out1 = LSTM(128, return_sequences=True)(input_model)
    x = Conv1D(16, 3, padding='same', name='block1_conv1')(input_model)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(16, 3, padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling1D(pool_size=2)(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    # lstm_out2 = LSTM(64, return_sequences=True)(x)
    x = Conv1D(32, 3, padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(32, 3, padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling1D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    # lstm_out3 = LSTM(32, return_sequences=True)(x)
    x = Conv1D(64, 3, padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(64, 3, padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(64, 3, padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    # UP 2
    x = UpSampling1D(size=2)(x)
    # x = Conv1DTranspose(256, 2, strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 256 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out])  # Объединем текущий слой со слоем block_3_out
    x = Conv1D(32, 3, padding='same')(x)  # Добавляем слой Conv2D с 256 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(32, 3, padding='same')(x)
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 3
    x = UpSampling1D(size=2)(x)
    # x = Conv1DTranspose(128, 2, strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_2_out
    x = Conv1D(16, 3, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(16, 3, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    last_layer = Flatten()(x)
    # last_layer = Dense(1, activation='linear')(x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов


output_coll = Dense(1, activation='sigmoid')(last_layer)

# Путь сохранения модели и графиков
model_name = 'test_{}_core_conv_n16_less'.format(train_state)
folder = 'data/'
model_folder = folder + 'models/'
graph_folder = folder + 'graphs/'


# Компиляция модели
model = Model(input_model, output_coll, name=model_name)
model.summary()
loss = "mae"
metrics = 'mse'
model.compile(loss=loss, metrics=[metrics], optimizer=Adam(learning_rate=1e-4))

# Callbacks
# создаём callback для сохранения лучшего результата и для уменьшения шага обучения при выходе на плато.
reduse_callback = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.2,patience=20,verbose=1,mode='min',min_lr=0.000001,cooldown=10,min_delta=0.01)
save_best_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    filepath=model_folder+model_name,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True, verbose=1)


if __name__ == '__main__':
    # Обучение модели
    history = model.fit(Gen,
                            epochs=epochs,
                            verbose=1,
                            batch_size=batch_size,
                            validation_data=Gen_test, callbacks=[reduse_callback, save_best_callback])

    model = load_model(model_folder + model_name, compile=False)
    model.compile(loss=loss, metrics=[metrics], optimizer=Adam(learning_rate=1e-4))

    print('validation loss {} ='.format(loss),
          accuracy_calculate(model, x_val_data[0], y_val_data[0], colls=False))
    plt.subplot(2, 2, 1)
    plt.title(label='ошибка')
    plt.plot(history.history['val_loss'], label='Test')
    plt.plot(history.history['loss'], label='Train')
    model.save(model_folder + model_name)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title(label='{}'.format(metrics))
    plt.plot(history.history['val_{}'.format(metrics)], label='Test')
    plt.plot(history.history['{}'.format(metrics)], label='Train')
    plt.legend()
    plt.savefig(graph_folder + model_name + '.jpg')
    plt.show()





