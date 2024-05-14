import sys
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D, SpatialDropout1D, AlphaDropout, Conv2D, \
    MaxPooling2D, Conv1D, MaxPooling1D
from scipy.stats import *
from tensorflow.keras.optimizers import Adam, SGD
from utils import load_np, save_np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pyreadr


def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)


#
# def get_model(mc=False, act="relu"):
#     inp = Input(input_shape)
#     prev_inp = Input(input_shape2)
#     other_input = Input(input_shape3)
#     # other_input2 = Input(input_shape4)
#     z = Conv1D(64,
#                kernel_size=3,
#                strides=2,
#                activation='softmax')(inp)
#     z = get_dropout(z, p=0.3, mc=mc)
#     z = MaxPooling1D()(z)
#     z = Conv1D(32,
#                kernel_size=3,
#                strides=1,
#                activation='softmax')(z)
#     z = get_dropout(z, p=0.3, mc=mc)
#     z = MaxPooling1D()(z)
#     # z = Conv1D(16,
#     #            kernel_size=3,
#     #            strides=1,
#     #            activation='relu')(z)
#     # z = get_dropout(z, p=0.3, mc=mc)
#     # z = MaxPooling1D()(z)
#     z = Flatten()(z)
#     y = Dense(16, activation='softmax')(z)
#     y = get_dropout(y, p=0.3, mc=mc)
#     p = Dense(8, activation='relu')(z)
#     p = get_dropout(p, p=0.3, mc=mc)
#     # s = concatenate([p,prev_inp,other_input,other_input2])
#     s = concatenate([p,prev_inp,other_input])
#     out = Dense(1, activation='softplus')(s) #softplus
#     # model = Model(inputs=[inp,prev_inp,other_input,other_input2], outputs=out)
#     model = Model(inputs=[inp,prev_inp,other_input], outputs=out)
#     model.compile(optimizer=Adam(learning_rate=0.001),  #0.001 softplus
#                   loss='mse',
#                   metrics=['mse'])
#
#     return model


def get_model2(mc=False, act="relu"):
    inp = Input(input_shape)
    prev_inp = Input(input_shape2)
    other_input = Input(input_shape3)
    # other_input2 = Input(input_shape4)
    z = Conv1D(16,
               kernel_size=3,
               strides=2,
               activation='softmax')(inp)
    z = get_dropout(z, p=0.3, mc=mc)
    z = MaxPooling1D()(z)
    z = Conv1D(32,
               kernel_size=3,
               strides=1,
               activation='softmax')(z)
    z = get_dropout(z, p=0.3, mc=mc)
    z = MaxPooling1D()(z)
    # z = Conv1D(16,
    #            kernel_size=3,
    #            strides=1,
    #            activation='relu')(z)
    # z = get_dropout(z, p=0.3, mc=mc)
    # z = MaxPooling1D()(z)
    z = Flatten()(z)
    y = Dense(16, activation='softmax')(z)
    y = get_dropout(y, p=0.3, mc=mc)
    p = Dense(8, activation='softmax')(z)
    p = get_dropout(p, p=0.3, mc=mc)
    # s = concatenate([p,prev_inp,other_input,other_input2])
    s = concatenate([p,prev_inp,other_input])
    out = Dense(1, activation='softplus')(s) #softplus
    # model = Model(inputs=[inp,prev_inp,other_input,other_input2], outputs=out)
    model = Model(inputs=[inp,prev_inp,other_input], outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001),  #0.001 softplus
                  loss='mse',
                  metrics=['mse'])

    return model

if __name__ == '__main__':


    # train_stbasis = pd.read_csv('data/stbasis_train.csv')
    # train_humidity = pd.read_csv('new_data/humidity_indMod.csv')
    # train_aerosol = pd.read_csv('new_data/aerosol_indMod.csv')
    # train_airnow = pd.read_csv('new_data/airnow_indMod.csv')
    #
    # train_Y = pd.read_csv('new_data/z_indMod.csv')
    # print(train_stbasis.shape)
    # sys.exit(1)

    # train_stbasis = pd.read_csv('new_data/stBasisAll_indMod_144.csv') # 8640
    train_stbasis = pd.read_csv('all_new_data/stBasisAll_indMod_sm.csv') # 1320
    train_humidity = pd.read_csv('all_new_data/humidity_indMod_up.csv')
    # train_aerosol = pd.read_csv('all_new_data/aerosol_indMod_up.csv')
    train_airnow = pd.read_csv('all_new_data/airnow_indMod.csv')

    train_Y = pd.read_csv('all_new_data/z_indMod_up.csv')

    # standardScaler = StandardScaler()
    # standardScaler.fit(train_humidity)
    # train_humidity_scaled = pd.DataFrame(standardScaler.transform(train_humidity))
    #
    standardScaler = StandardScaler()
    standardScaler.fit(train_airnow)
    train_airnow_scaled = pd.DataFrame(standardScaler.transform(train_airnow))
    # # standardScaler = StandardScaler()
    # standardScaler.fit(train_aerosol)
    # train_aerosol_scaled = pd.DataFrame(standardScaler.transform(train_aerosol))

    # standardScaler = StandardScaler()
    # standardScaler.fit(train_stbasis)
    # train_stbasis_scaled = pd.DataFrame(standardScaler.transform(train_stbasis))
    # train_stbasis = train_dataset_stbasis.reset_index(drop=True)
    # train_humidity = train_dataset_humid.reset_index(drop=True)
    # train_aerosol = train_dataset_aerosol.reset_index(drop=True)
    # train_airnow = train_dataset_airnow.reset_index(drop=True)

    train_dataset = pd.concat([train_stbasis,train_airnow_scaled, train_humidity], axis=1,join="inner") # Ver2
    # train_dataset = pd.concat([train_stbasis,train_airnow_scaled, train_humidity,train_aerosol], axis=1,join="inner") # Ver2
    # train_dataset = pd.concat([train_stbasis, train_airnow_scaled, train_humidity_scaled, train_aerosol], axis=1,
    #                           join="inner")  # Ver2

    # print(train_dataset.shape)
    # sys.exit(1)
    train_x_now, test_x_now, train_y_now, test_y_now = train_test_split(train_dataset, train_Y,
                                                                        test_size=0.15,
                                                                        random_state=8)  # 0.25 x 0.8 = 0.2
    # print(train_stbasis.shape)
    # print(train_x_now.shape)

    train_basis = train_x_now.iloc[:, 0:1320]
    train_airnow = train_x_now.iloc[:, 1320]
    train_humidity = train_x_now.iloc[:, 1321]
    # train_aerosol = train_x_now.iloc[:, 1322]
    # print(train_basis.shape)
    # sys.exit(1)
    # print(train_prev.shape)
    # sys.exit(1)

    test_basis = test_x_now.iloc[:, 0:1320]
    test_airnow = test_x_now.iloc[:, 1320]
    test_humidity = test_x_now.iloc[:, 1321]
    # test_aerosol = test_x_now.iloc[:, 1322]


    input_shape = (1320, 1)
    input_shape2 = (1,)
    input_shape3 = (1,)
    # input_shape4 = (1,)
    batch_size = 10  # 10 for August 32 for September #버전4=10    32,   ver3 softplus 32/50
    epochs = 50  # 5 for August 30 for September #ver4,5 20    50, 100

    mc_model = get_model2(mc=True, act="relu")

    # train_prev = train_prev.reshape(1,-1)
    # train_basis = train_basis.reshape(1,-1)
    # test_prev = test_prev.reshape(1,-1)
    # test_basis = test_basis.reshape(1,-1)

    # h_mc = mc_model.fit([train_basis, train_airnow, train_humidity, train_aerosol], train_y_now,
    #                     batch_size=batch_size,
    #                     epochs=epochs,
    #                     verbose=1,
    #                     validation_data=([test_basis, test_airnow, test_humidity, test_aerosol], test_y_now))

    h_mc = mc_model.fit([train_basis, train_airnow, train_humidity], train_y_now,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=([test_basis, test_airnow, test_humidity], test_y_now))

    # h_mc = mc_model.fit([train_basis, train_airnow, train_humidity], train_y_now,
    #                     batch_size=batch_size,
    #                     epochs=epochs,
    #                     verbose=1,
    #                     validation_data=([test_basis, test_airnow, test_humidity], test_y_now))

    # h_mc = mc_model.fit([train_basis, train_airnow, train_humidity], train_y_now,
    #                     batch_size=batch_size,
    #                     epochs=epochs,
    #                     verbose=1,
    #                     validation_data=([test_basis, test_airnow, test_humidity], test_y_now))


    mc_model.save(f'model_simulation/update_ver5_50days_softplus_v4.h5')
    # mc_model_ver3_50days_linear

    print('finish!')