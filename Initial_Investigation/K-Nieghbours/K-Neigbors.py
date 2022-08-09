import os 
import inspect
import random 
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import metrics
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# Loading Data ################################################################
###############################################################################
adress = "Inverter Data Set.csv"
df = pd.read_csv(adress).sample(frac=1)
df2 = pd.read_csv(adress).sample(frac=1)
###############################################################################
dfx1 = df[['i_a_k',
            'i_a_k-1',
            'i_b_k',
            'i_b_k-1',
            'i_c_k',
            'i_c_k-1',
            'u_dc_k',
            'u_dc_k-1',
            'd_a_k-2',
            'd_a_k-3',
            'd_b_k-2',
            'd_b_k-3',
            'd_c_k-2',
            'd_c_k-3']]
dfx2 = df2[['u_a_k-1',
            'u_b_k-1',
            'u_c_k-1',
            'd_a_k-3',
            'd_b_k-3',
            'd_c_k-3',
            'i_a_k-3',
            'i_b_k-3',
            'i_c_k-3',
            'i_a_k-2',
            'i_b_k-2',
            'i_c_k-2',
            'u_dc_k-3',
            'u_dc_k-2']]
dfy1 = df.pop("u_a_k-1")
dfy2 = df.pop("u_b_k-1")
dfy3 = df.pop("u_c_k-1")
dfy1_1 = df2.pop("d_a_k-2")
dfy1_2 = df2.pop("d_b_k-2")
dfy1_3 = df2.pop("d_c_k-2")
#split data black box inverter model 
X_trainUak1, X_testUak1, y_trainUak1, y_testUak1 = train_test_split(dfx1,dfy1, test_size = 0.3)
X_trainUbk1, X_testUbk1, y_trainUbk1, y_testUbk1 = train_test_split(dfx1,dfy2, test_size = 0.3)
X_trainUck1, X_testUck1, y_trainUck1, y_testUck1 = train_test_split(dfx1,dfy3, test_size = 0.3)
#Split data black box compensation scheme 
X_trainDak2, X_testDak2, y_trainDak2, y_testDak2 = train_test_split(dfx2,dfy1_1, test_size = 0.3)
X_trainDbk2, X_testDbk2, y_trainDbk2, y_testDbk2 = train_test_split(dfx2,dfy1_2, test_size = 0.3)
X_trainDck2, X_testDck2, y_trainDck2, y_testDck2 = train_test_split(dfx2,dfy1_3, test_size = 0.3)
    

LinModUak1 = KNeighborsRegressor().fit(X_trainUak1,y_trainUak1)
LinModUbk1 = KNeighborsRegressor().fit(X_trainUbk1, y_trainUbk1)
LinModUck1 = KNeighborsRegressor().fit(X_trainUck1, y_trainUck1)

LinModDak2 = KNeighborsRegressor().fit(X_trainDak2, y_trainDak2)
LinModDbk2 = KNeighborsRegressor().fit(X_trainDbk2, y_trainDbk2)
LinModDck2 = KNeighborsRegressor().fit(X_trainDck2, y_trainDck2)


R2TrainUak1 = LinModUak1.score(X_trainUak1, y_trainUak1)
R2TrainUbk1 = LinModUbk1.score(X_trainUbk1, y_trainUbk1)
R2TrainUck1 = LinModUck1.score(X_trainUck1, y_trainUck1)

R2TestUak1 = LinModUak1.score(X_testUak1, y_testUak1)
R2TestUbk1 = LinModUbk1.score(X_testUbk1, y_testUbk1)
R2TestUck1 = LinModUck1.score(X_testUck1, y_testUck1)

MAETrainUak1 = mean_absolute_error(y_trainUak1, LinModUak1.predict(X_trainUak1)) 
MAETrainUbk1 = mean_absolute_error(y_trainUbk1, LinModUbk1.predict(X_trainUbk1))
MAETrainUck1 = mean_absolute_error(y_trainUck1, LinModUck1.predict(X_trainUck1))
MAETestUak1 = mean_absolute_error(y_testUak1, LinModUak1.predict(X_testUak1))
MAETestUbk1 = mean_absolute_error(y_testUbk1, LinModUbk1.predict(X_testUbk1))
MAETestUck1 = mean_absolute_error(y_testUck1, LinModUck1.predict(X_testUck1))

MSETrainUak1 = mean_squared_error(y_trainUak1, LinModUak1.predict(X_trainUak1))
MSETrainUbk1 = mean_squared_error(y_trainUbk1, LinModUbk1.predict(X_trainUbk1))
MSETrainUck1 = mean_squared_error(y_trainUck1, LinModUck1.predict(X_trainUck1))
MSETestUak1 = mean_squared_error(y_testUak1, LinModUak1.predict(X_testUak1))
MSETestUbk1 = mean_squared_error(y_testUbk1, LinModUbk1.predict(X_testUbk1))
MSETestUck1 = mean_squared_error(y_testUck1, LinModUck1.predict(X_testUck1))
RMSETrainUak1 = np.sqrt(MSETrainUak1)
RMSETrainUbk1 = np.sqrt(MSETrainUbk1)
RMSETrainUck1 = np.sqrt(MSETrainUck1)
RMSETestUak1 = np.sqrt(MSETestUak1)
RMSETestUbk1 = np.sqrt(MSETestUbk1)
RMSETestUck1 = np.sqrt(MSETestUck1)
print("########################################################################")
print("# Evaluation values for UaK-1, Ubk-1, Uck-1 ############################")
print("########################################################################")
print("R^2 (Uak-1) Train = {}".format(R2TrainUak1))
print("R^2 (Uak-1) Test  = {}".format(R2TestUak1))
print("R^2 (Ubk-1) Train = {}".format(R2TrainUbk1))
print("R^2 (Ubk-1) Test = {}".format(R2TestUbk1))
print("R^2 (Uck-1) Train = {}".format(R2TrainUck1))
print("R^2 (Uck-1) Test = {}".format(R2TestUck1))
print("########################################################################")
print("MAE (Uak-1) Train = {}".format(MAETrainUak1))
print("MAE (Uak-1) Test = {}".format(MAETestUak1))
print("MAE (Ubk-1) Train = {}".format(MAETrainUbk1))
print("MAE (Ubk-1) Test = {}".format(MAETestUbk1))
print("MAE (Uck-1) Train = {}".format(MAETrainUck1))
print("MAE (Uck-1) Test = {}".format(MAETestUck1))
print("########################################################################")
print("MSE (Uak-1) Train = {}".format(MSETrainUak1))
print("MSE (Uak-1) Test = {}".format(MSETestUak1))
print("MSE (Ubk-1) Train = {}".format(MSETrainUbk1))
print("MSE (Ubk-1) Test = {}".format(MSETestUbk1))
print("MSE (Uck-1) Train = {}".format(MSETrainUck1))
print("MSE (Uck-1) Test = {}".format(MSETestUck1))
print("########################################################################")
print("RMSE (Uak-1) Train = {}".format(RMSETrainUak1))
print("RMSE (Uak-1) Test = {}".format(RMSETestUak1))
print("RMSE (Ubk-1) Train = {}".format(RMSETrainUbk1))
print("RMSE (Ubk-1) Test = {}".format(RMSETestUbk1))
print("RMSE (Uck-1) Train = {}".format(RMSETrainUck1))
print("RMSE (Uck-1) Test = {}".format(RMSETestUck1))
print("########################################################################")
R2TrainDak2 = LinModDak2.score(X_trainDak2, y_trainDak2)
R2TrainDbk2 = LinModDbk2.score(X_trainDbk2, y_trainDbk2)
R2TrainDck2 = LinModDck2.score(X_trainDck2, y_trainDck2)
R2TestDak2 = LinModDak2.score(X_testDak2, y_testDak2)
R2TestDbk2 = LinModDbk2.score(X_testDbk2, y_testDbk2)
R2TestDck2 = LinModDck2.score(X_testDck2, y_testDck2)

MAETrainDak2 = mean_absolute_error(y_trainDak2, LinModDak2.predict(X_trainDak2)) 
MAETrainDbk2 = mean_absolute_error(y_trainDbk2, LinModDbk2.predict(X_trainDbk2))
MAETrainDck2 = mean_absolute_error(y_trainDck2, LinModDck2.predict(X_trainDck2))
MAETestDak2 = mean_absolute_error(y_testDak2, LinModDak2.predict(X_testDak2))
MAETestDbk2 = mean_absolute_error(y_testDbk2, LinModDbk2.predict(X_testDbk2))
MAETestDck2 = mean_absolute_error(y_testDck2, LinModDck2.predict(X_testDck2))

MSETrainDak2 = mean_squared_error(y_trainDak2, LinModDak2.predict(X_trainDak2))
MSETrainDbk2 = mean_squared_error(y_trainDbk2, LinModDbk2.predict(X_trainDbk2))
MSETrainDck2 = mean_squared_error(y_trainDck2, LinModDck2.predict(X_trainDck2))
MSETestDak2 = mean_squared_error(y_testDak2, LinModDak2.predict(X_testDak2))
MSETestDbk2 = mean_squared_error(y_testDbk2, LinModDbk2.predict(X_testDbk2))
MSETestDck2 = mean_squared_error(y_testDck2, LinModDck2.predict(X_testDck2))
RMSETrainDak2 = np.sqrt(MSETrainDak2)
RMSETrainDbk2 = np.sqrt(MSETrainDbk2)
RMSETrainDck2 = np.sqrt(MSETrainDck2)
RMSETestDak2 = np.sqrt(MSETestDak2)
RMSETestDbk2 = np.sqrt(MSETestDbk2)
RMSETestDck2 = np.sqrt(MSETestDck2)

print("########################################################################")
print("# Evaluation values for Dak-2, Dbk-2, Dck-2 ############################")
print("########################################################################")
print("R^2 (Dak-2) Train = {}".format(R2TrainDak2))
print("R^2 (Dak-2) Test = {}".format(R2TestDak2))
print("R^2 (Dbk-2) Train = {}".format(R2TrainDbk2))
print("R^2 (Dbk-2) Test = {}".format(R2TestDbk2))
print("R^2 (Dck-2) Train = {}".format(R2TrainDck2))
print("R^2 (Dck-2) Test = {}".format(R2TestDck2))
print("########################################################################")
print("MAE (Dak-2) Train = {}".format(MAETrainDak2))
print("MAE (Dak-2) Test = {}".format(MAETestDak2))
print("MAE (Dbk-2) Train = {}".format(MAETrainDbk2))
print("MAE (Dbk-2) Test = {}".format(MAETestDbk2))
print("MAE (Dck-2) Train = {}".format(MAETrainDck2))
print("MAE (Dck-2) Test = {}".format(MAETestDck2))
print("########################################################################")
print("MSE (Dak-2) Train = {}".format(MSETrainDak2))
print("MSE (Dak-2) Test = {}".format(MSETestDak2))
print("MSE (Dbk-2) Train = {}".format(MSETrainDbk2))
print("MSE (Dbk-2) Test = {}".format(MSETestDbk2))
print("MSE (Dck-2) Train = {}".format(MSETrainDck2))
print("MSE (Dck-2) Test = {}".format(MSETestDck2))
print("########################################################################")
print("RMSE (Dak-2) Train = {}".format(RMSETrainDak2))
print("RMSE (Dak-2) Test = {}".format(RMSETestDak2))
print("RMSE (Dbk-2) Train = {}".format(RMSETrainDbk2))
print("RMSE (Dbk-2) Test = {}".format(RMSETestDbk2))
print("RMSE (Dck-2) Train = {}".format(RMSETrainDck2))
print("RMSE (Dck-2) Test = {}".format(RMSETestDck2))
print("########################################################################")
    

    
    
    
    
file0.close()
file1.close()