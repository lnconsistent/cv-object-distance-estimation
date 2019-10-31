import pandas as pd
import argparse

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# argparser = argparse.ArgumentParser(description='Get predictions of test set')
# argparser.add_argument('-m', '--modelname', default='model@1572128487',
#                        help='model name (.json)')
# argparser.add_argument('-w', '--weights', default='model@1572128487',
#                        help='weights filename (.h5)')
#
# args = argparser.parse_args()

# parse arguments
MODEL = 'model@1572128487'
WEIGHTS = 'model@1572128487'


def infer(df):
    # inference from df bboxes -> df est distances
    # get data
    X_test = df[['xmin', 'ymin', 'xmax', 'ymax']].values

    # load json and create model
    json_file = open('generated_files/{}.json'.format(MODEL), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json( loaded_model_json )

    # load weights into new model
    loaded_model.load_weights("generated_files/{}.h5".format(WEIGHTS))
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    y_pred = loaded_model.predict(X_test)

    # save predictions
    df_result = df
    df_result['zloc_pred'] = -100000

    for idx, row in df_result.iterrows():
        df_result.at[idx, 'zloc_pred'] = y_pred[idx]

    # df_result.to_csv('data/predictions.csv', index=False)
    return df_result
