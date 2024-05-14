import sys
import logging
from keras.models import Sequential, Model, Input, load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def save_np(arr, path):
    with open(path, 'wb') as f:
        np.save(f, arr)


def load_np(path):
    with open(path, 'rb') as f:
        arr = np.load(f)
    return arr

if __name__ == '__main__':
    mc_model = load_model(f'model_simulation/update_50days_softplus.h5')  

    logging.debug("load")


    # all_50days
    test_stbasis = pd.read_csv('all_new_data/stBasisAll_indCV_sm.csv')
    test_airnow = pd.read_csv('all_new_data/airnow_indCV.csv')
    test_humidity = pd.read_csv('all_new_data/humidity_indCV_up.csv')
    test_aerosol = pd.read_csv('all_new_data/aerosol_indCV_up.csv')

    standardScaler = StandardScaler()
    test_airnow_scaled  = pd.DataFrame(standardScaler.transform(test_airnow))
  
    mc_predictions_cnn_cov = []


    for j in range(100):
        CNN_prediction_Y_cv = mc_model.predict([test_stbasis,test_airnow_scaled,test_humidity,test_aerosol])
        response = np.array(CNN_prediction_Y_cv)
        mc_predictions_cnn_cov.append(response)

 
   save_np(mc_predictions_cnn_cov, f'output/update_all_softplus.ny') 
   

