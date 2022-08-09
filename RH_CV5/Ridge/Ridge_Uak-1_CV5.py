

##############################################################################
# Required Libraries #########################################################
##############################################################################
import numpy as np 
import pandas as pd 
import os 
import random
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_validate
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# Loading Data ################################################################
###############################################################################
df = pd.read_csv('Inverter Data Set.csv')

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
dfy1 = df.pop("u_a_k-1")
X_train, X_test, y_train, y_test = train_test_split(dfx1,dfy1, test_size = 0.3)
def RandomizedSearch():
    parameters = []
    Alpha = random.uniform(1.0, 1000.0)
    fitIntercept = random.choice([True,False])
    maxIter = random.randint(100, 100000)
    tolerance = random.uniform(1e-9,1e-3)
    solve = random.choice(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
    print("parameters = {}".format(parameters))
    parameters = [Alpha, 
                  fitIntercept,
                  maxIter,
                  tolerance, 
                  solve]
    return parameters

def RidgeRegCV(parameters, X_train, y_train, X_test, y_test): 
    model = linear_model.Ridge(alpha=parameters[0],
                             fit_intercept=parameters[1], 
                             max_iter=parameters[2], 
                             tol=parameters[3], 
                             solver = parameters[4])
    cvmodel = cross_validate(model, X_train, y_train, cv=5, 
                              scoring = ("r2", 
                                         "neg_mean_absolute_error",
                                         "neg_root_mean_squared_error",
                                         "neg_mean_absolute_percentage_error"),
                              return_train_score=True)
    print("###################################################################")
    print("# Results from CV 5 Cross Validation Using Multiple Metric")
    print("###################################################################")
    print("All Scores From CV5 = {}".format(cvmodel))
    ##########################################################################
    #Train Test Scores Raw####################################################
    ##########################################################################
    file1.write("R2 Train Scores = {}\n".format(cvmodel['train_r2']))
    file1.write("R2 Test Scores = {}\n".format(cvmodel['test_r2']))
    file1.write("MAE Train Scores = {}\n".format(abs(cvmodel['train_neg_mean_absolute_error'])))
    file1.write("MAE Test Scores = {}\n".format(abs(cvmodel['test_neg_mean_absolute_error'])))
    file1.write("RMSE Train Scores = {}\n".format(abs(cvmodel['train_neg_root_mean_squared_error'])))
    file1.write("RMSE Test Scores = {}\n".format(abs(cvmodel['test_neg_root_mean_squared_error'])))
    file1.write("MAPE Train Scores = {}\n".format(abs(cvmodel['train_neg_mean_absolute_percentage_error'])))
    file1.write("MAPE Test Scores = {}\n".format(abs(cvmodel['test_neg_mean_absolute_percentage_error'])))
    print("###################################################################")
    print("# Calculate Mean and Standard Deviation of Metric values ")
    print("###################################################################")
    AvrR2ScoreTrain = np.mean(cvmodel['train_r2'])
    StdR2ScoreTrain = np.std(cvmodel['train_r2'])
    AvrR2ScoreTest = np.mean(cvmodel['test_r2'])
    StdR2ScoreTest = np.std(cvmodel['test_r2'])
    AvrAllR2Score = np.mean([AvrR2ScoreTrain,AvrR2ScoreTest])
    StdAllR2Score = np.std([AvrR2ScoreTrain, AvrR2ScoreTest])
    
    AvrMAEScoreTrain = np.mean(abs(cvmodel['train_neg_mean_absolute_error']))
    StdMAEScoreTrain = np.std(abs(cvmodel['train_neg_mean_absolute_error']))
    AvrMAEScoreTest = np.mean(abs(cvmodel['test_neg_mean_absolute_error']))
    StdMAEScoreTest = np.std(abs(cvmodel['test_neg_mean_absolute_error']))
    AvrAllMAEScore = np.mean([AvrMAEScoreTrain, AvrMAEScoreTest])
    StdAllMAEScore = np.std([AvrMAEScoreTrain, AvrMAEScoreTest])
    
    AvrRMSEScoreTrain = np.mean(abs(cvmodel['train_neg_root_mean_squared_error']))
    StdRMSEScoreTrain = np.std(abs(cvmodel['train_neg_root_mean_squared_error']))
    AvrRMSEScoreTest = np.mean(abs(cvmodel['test_neg_root_mean_squared_error']))
    StdRMSEScoreTest = np.std(abs(cvmodel['test_neg_root_mean_squared_error']))
    AvrAllRMSEScore = np.mean([AvrRMSEScoreTrain, AvrRMSEScoreTest])
    StdAllRMSEScore = np.std([AvrRMSEScoreTrain, AvrRMSEScoreTest])
    
    AvrMAPEScoreTrain = np.mean(abs(cvmodel['train_neg_mean_absolute_percentage_error']))
    StdMAPEScoreTrain = np.std(abs(cvmodel['train_neg_mean_absolute_percentage_error']))
    AvrMAPEScoreTest = np.mean(abs(cvmodel['test_neg_mean_absolute_percentage_error']))
    StdMAPEScoreTest = np.std(abs(cvmodel['test_neg_mean_absolute_percentage_error']))
    AvrAllMAPEScore = np.mean([AvrMAPEScoreTrain, AvrMAPEScoreTest])
    StdAllMAPEScore = np.std([AvrMAPEScoreTrain, AvrMAPEScoreTest])
    print("CV-R^2 Score = {}".format(AvrAllR2Score))
    print("CV-STD R^2 Score = {}".format(StdAllR2Score))
    print("CV-MAE Score = {}".format(AvrAllMAEScore))
    print("CV-STD MAE Score = {}".format(StdAllMAEScore))
    print("CV-RMSE Score ={}".format(AvrAllRMSEScore))
    print("CV-STD RMSE Score = {}".format(StdAllRMSEScore))
    print("CV-MAPE Score = {}".format(AvrAllMAPEScore))
    print("CV-STD MAPE Score = {}".format(StdAllMAPEScore))
    file1.write("##############################################################\n"+\
                "AvrR2Score Train = {}\n".format(AvrR2ScoreTrain)+\
                "StdR2Score Train = {}\n".format(StdR2ScoreTrain)+\
                "AvrR2Score Test = {}\n".format(AvrR2ScoreTest)+\
                "StdR2Score Test = {}\n".format(StdR2ScoreTest)+\
                "AvrAllR2Score = {}\n".format(AvrAllR2Score)+\
                "StdAllR2Score = {}\n".format(StdAllR2Score)+\
                "AvrMAEScore Train = {}\n".format(AvrMAEScoreTrain)+\
                "StdMAEScore Train = {}\n".format(StdMAEScoreTrain)+\
                "AvrMAEScore Test = {}\n".format(AvrMAEScoreTest)+\
                "StdMAEScore Test ={}\n".format(StdMAEScoreTest)+\
                "AvrAllMAEScore = {}\n".format(AvrAllMAEScore)+\
                "StdAllMAEScore = {}\n".format(StdAllMAEScore)+\
                "AvrRMSEScore Train = {}\n".format(AvrRMSEScoreTrain)+\
                "StdRMSEScore Train = {}\n".format(StdRMSEScoreTrain)+\
                "AvrRMSEScore Test = {}\n".format(AvrRMSEScoreTest)+\
                "StdRMSEScore Test = {}\n".format(StdRMSEScoreTest)+\
                "AvrAllRMSEScore = {}\n".format(AvrAllRMSEScore)+\
                "StdAllRMSEScore = {}\n".format(StdAllRMSEScore)+\
                "AvrMAPEScore Train ={}\n".format(AvrMAPEScoreTrain)+\
                "StdMAPEScore Train = {}\n".format(StdMAPEScoreTrain)+\
                "AvrMAPEScore Test = {}\n".format(AvrMAPEScoreTest)+\
                "StdMAPEScore Test = {}\n".format(StdMAPEScoreTest)+\
                "AvrAllMAPEScore = {}\n".format(AvrAllMAPEScore)+\
                "StdAllMAPeScore = {}\n".format(StdAllMAPEScore)+\
                "###############################################################\n")
    if AvrAllR2Score > 0.995:
        print("###############################################################")
        print(" Final Evaluation")
        print("################################################################")
        file1.write("###############################################################\n")
        file1.write(" Final Evaluation\n")
        file1.write("################################################################\n")
        model.fit(X_train,y_train)
        R2Test = model.score(X_test,y_test)
        MAETest = mean_absolute_error(y_test, model.predict(X_test))
        RMSETest = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        MAPETest = mean_absolute_percentage_error(y_test, model.predict(X_test))
        print("R^2 Test = {}".format(R2Test))
        print("MAE Test = {}".format(MAETest))
        print("RMSE Test = {}".format(RMSETest))
        print("MAPE Test = {}".format(MAPETest))
        file1.write("########################################################\n")
        file1.write("model R^2 Test = {}\n".format(R2Test))
        file1.write("MAE Test = {}\n".format(MAETest))
        file1.write("RMSE Test = {}\n".format(RMSETest))
        file1.write("MAPE Test = {}\n".format(MAPETest))
        file1.write("########################################################\n")
        file1.flush()
        return R2Test
    else:
        return AvrAllR2Score

name = "RidgeCV5_Uak-1"
file0 = open("{}_parameters.data".format(name), "w")
file1 = open("{}_results.data".format(name), "w")
# test = KNNParRandomSearch()
# print(test)
k = 0
while True:
    print("Current Iteration = {}".format(k))
    Param = RandomizedSearch()
    test = RidgeRegCV(Param,X_train,y_train, X_test, y_test)
    k+=1
    if test > 0.995:
        print("Solution is Found!!")
        file1.write("Solution is Found!")
        file1.flush()
        break 
    else:
        continue
file0.close()
file1.close()    