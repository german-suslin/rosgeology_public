from Generator import Worker, Generator, accuracy_calculate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np

def predict_kpef(model,
            x_data,
            y_data = None,
            x_columns = None,
            norm_columns = None,
            lenth = 16):
    '''

    '''
    if y_data:
        y_colls = y_data[:,:3]
        y_rest = y_data[:,3:]
    else:
        y_colls = np.zeros(shape=(x_data.shape[0],3))
        y_rest = np.zeros(shape=(x_data.shape[0],1))
    if x_columns:
        pass
    else:
        x_columns = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    if norm_columns:
        pass
    else:
        norm_columns = [0, 1, 2, 3, 4, 5, 6]
    Gen = Generator(x_data,
                    y_colls,
                    y_rest,
                    lenth,
                    batch_size=len(x_data) - lenth,
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
    print('validation data shape', x_val_data.shape, y_val_data.shape)
    pred = model.predict(x_val_data[0])
    return pred

if __name__ == '__main__':
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
    predict = predict_kpef(model, x_data)
    print(predict.shape)
    print(predict[:20])