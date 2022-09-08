from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from Generator import Worker, Generator, accuracy_calculate, tpe
import matplotlib.pyplot as plt
import numpy as np

model_name = 'test_unet_core_unet_n16_second'
folder = 'data/'
model_folder = folder + 'models/'
graph_folder = folder + 'graphs/'
model = load_model(model_folder + model_name, compile=False)

model.compile(loss='mae', metrics=['mse'], optimizer=Adam(learning_rate=1e-4))
model.summary()
worker = Worker('test1000.csv')

columns = ['GGKP_korr', 'GK_korr', 'PE_korr', 'DS_korr', 'DTP_korr', 'Wi_korr', 'BK_korr', 'BMK_korr']
x_columns = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
norm_columns = [0, 1, 2, 3, 4, 5, 6]

x_data = worker.get_x_data(columns)
y_data_colls, y_data_rest = worker.get_y_collektors()
x_data = np.concatenate([x_data, y_data_colls], axis=1)
y_data = np.concatenate([y_data_colls, y_data_rest], axis=1)
y_coll = y_data[:, :3]
y_rest = y_data[:, 3].reshape(-1, 1)
lenght = 16
print('rest', y_data_rest)
Gen = Generator(x_data,
                y_coll,
                y_rest,
                lenght,
                batch_size=len(x_data) - lenght,
                x_columns=x_columns,
                y_columns=[0], only_colls=False)
norm_fit, norm_y_fit = Gen.normalize(columns=norm_columns)
x_val_data = []
y_val_data = []
for x in Gen:
    x_val_data.append(x[0])
    y_val_data.append(x[1])

x_val_data = np.array(x_val_data)
y_val_data = np.array(y_val_data)
# print('val', y_val_data[0, :2000])
print('validation data shape', x_val_data.shape, y_val_data.shape)
pred = model.predict(x_val_data[0])
# print('tpe', tpe(y_val_data[0], pred))
print(accuracy_calculate(model, x_val_data[0], y_val_data[0], colls=False))
plt.title(label='предсказание')
plt.plot(pred, label='предсказанное')
plt.plot(y_val_data[0], label='истиное')
model.save(model_folder + model_name)
plt.legend()
plt.show()