import sys
import logging
from keras.models import Sequential, Model, Input, load_model
from utils import load_np, save_np
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    mc_model = load_model(f'model_simulation/update_ver5_50days_softplus_v4.h5')  # update_ver4_50days_softplus_v3   # update_ver3_50days_linear_v4
    # mc_model_ver5_50days_softplus

    logging.debug("load")


    # all_50days
    test_stbasis = pd.read_csv('all_new_data/stBasisAll_indCV_sm.csv')
    test_airnow = pd.read_csv('all_new_data/airnow_indCV.csv')
    test_humidity = pd.read_csv('all_new_data/humidity_indCV_up.csv')
    # test_aerosol = pd.read_csv('all_new_data/aerosol_indCV_up.csv')

    standardScaler = StandardScaler()
    # standardScaler.fit(test_humidity)
    standardScaler.fit(test_airnow)
    # test_humidity_scaled = pd.DataFrame(standardScaler.transform(test_humidity))
    test_airnow_scaled  = pd.DataFrame(standardScaler.transform(test_airnow))
    # train_stbasis = train_dataset_stbasis.reset_index(drop=True)
    # train_humidity = train_dataset_humid.reset_index(drop=True)




    mc_predictions_cnn_cov = []
    # cnn-cov

    for j in range(100):
        # CNN_prediction_Y_cv = mc_model.predict([test_stbasis,test_airnow_scaled,test_humidity,test_aerosol])
        CNN_prediction_Y_cv = mc_model.predict([test_stbasis, test_airnow_scaled, test_humidity])
        # CNN_prediction_Y_cv =mc_model.predict([test_stbasis,test_airnow_scaled,test_humidity_scaled,test_aerosol])
        # CNN_prediction_Y_cv =mc_model.predict([test_stbasis,test_airnow,test_humidity])
        response = np.array(CNN_prediction_Y_cv)
        mc_predictions_cnn_cov.append(response)

    #
    # for j in range(100):
    #     CNN_prediction_Y_cv =mc_model.predict([test_basis,test_airnow,test_humidity])
    #     response = np.array(CNN_prediction_Y_cv)
    #     mc_predictions_cnn_cov.append(response)




    save_np(mc_predictions_cnn_cov, f'output/update_all_ver5_softplus.ny') # prev : prediction_cov_0819_08
    # August : prediction_cov_0822_08
    # save_np(pred_test, f'output/prediction_test_cov.ny')
    # prediction_cov_all_ver2_nouvai_0822_08_softplus.ny

