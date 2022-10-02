from Generator import Worker, Generator, accuracy_calculate
import numpy as np

def predict_kpef(model,
            x_data,
            y_data,
            x_columns = None,
            norm_columns = None,
            lenth = 16):
    '''

    '''
    y_colls = y_data[:,:3]
    y_rest = y_data[:,3:]
    model.summary()
    if x_columns:
        pass
    else:
        x_columns = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    if norm_columns:
        pass
    else:
        norm_columns = [0, 1, 2, 3, 4, 5, 6]
    x_data = np.concatenate([x_data, y_colls], axis=1)
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