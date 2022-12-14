from Generator import Worker, Generator, accuracy_calculate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np

def predict_kpef_knef(model,
            x_data:np.array,
            y_data:np.array = np.array([]),
            x_columns:list = [],
            norm_columns:list = [],
            lenth:int = 16,
            predict_value:str = 'kpef'):
    '''
    x_data: данные первых 8 колонок и, опционально для моделей, данные коллекторов
    в формате OHE

    y_data: опционально (по умолчанию - массив нулей той же длины, что и x_data - указать данные для сравнения с предсказанными
    если указан параметр - будет выведен рассчёт точности для предсказания модели

    x_columns: опционально (по умолчанию - 0, 1, 2, 4, 5, 6, 7, 8, 9, 10) - список числовых индексов колонок, который используется для выборки входных данных
    по умолчанию - с учётом данных для коллекторов в OHE (не считая третьей колонки)

    norm_columns: опционально (по умолчанию - 0, 1, 2, 3, 4, 5, 6) - список числовых индексов колонок, которые нужно нормализировать
    ВНИМАНИЕ - индексация не по исходному массиву данных, а по уже отобранному в x_columns

    lenth: по умолчанию = 16 - длинна сета для предсказания (параметр зависит от модели), по умолчанию - 16

    predict_value: по умолчанию - kpef какое значение предсказываем - kpef или knef.
    влияет только на сравнение данных - по
    '''
    # Если указаны эталонные выходные данные - отбирает их
    # в противном случае - создаёт массивы нулей нужных размеров
    # это необходимо для корректной работы генератора
    if len(y_data):
        y_colls = y_data[:,:3]
        if predict_value == 'kpef':
            y_rest = y_data[:,3].reshape(-1, 1)
        else:
            y_rest = y_data[:, 4].reshape(-1, 1)
    else:
        y_colls = np.zeros(shape=(x_data.shape[0],3))
        y_rest = np.zeros(shape=(x_data.shape[0],1))

    # Обработка случаев, когда указаны списки колонок для отбора и нормализации
    # и когда не указаны.
    if len(x_columns):
        pass
    else:
        x_columns = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]

    if len(norm_columns):
        pass
    else:
        norm_columns = [0, 1, 2, 3, 4, 5, 6]

    # Создание генератора, нормализация и заполнение массивов выходных и входных данных
    # генератор по умолчанию создаёт сеты данных для обучения модели,
    # поэтому в аргументы требует входные и выходные данные,
    # по этой же причине - возвращает и входные и выходные данные,
    # в случае, если мы не сравниваем с эталонными данными - принимает массивы нулей в качестве y_data
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

    # случай сравнения с эталонными данными.
    if len(y_data):
        print('loss mae',
              accuracy_calculate(model, x_val_data[0], y_val_data[0], colls=False, scaler=norm_fit))
    return pred

# if __name__ == '__main__':
#     model_name = 'test_unet_core_unet_n16_second'
#     folder = 'data/'
#     model_folder = folder + 'models/'
#     graph_folder = folder + 'graphs/'
#     model = load_model(model_folder + model_name, compile=False)
#
#     model.compile(loss='mae', metrics=['mse'], optimizer=Adam(learning_rate=1e-4))
#     # model.summary()
#     worker = Worker('test1000.csv')
#
#     columns = ['GGKP_korr', 'GK_korr', 'PE_korr', 'DS_korr', 'DTP_korr', 'Wi_korr', 'BK_korr', 'BMK_korr']
#     x_columns = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
#     norm_columns = [0, 1, 2, 3, 4, 5, 6]
#     x_data = worker.get_x_data(columns)
#     y_data_colls, y_data_rest = worker.get_y_collektors()
#
#     x_data = np.concatenate([x_data, y_data_colls], axis=1)
#     y_data = np.concatenate([y_data_colls, y_data_rest], axis=1)
#     print('y_data', y_data.shape)
#     predict = predict_kpef_knef(model, x_data, y_data=y_data)
#     print(predict.shape)
#     print(predict[:5])