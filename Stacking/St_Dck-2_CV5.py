

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
from sklearn.linear_model import Ridge
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor 
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_validate
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# Loading Data ################################################################
###############################################################################

df = pd.read_csv('Inverter Data Set.csv')
dfx1 = df[['u_a_k-1',
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

dfy1 = df.pop("d_c_k-2")
X_train, X_test, y_train, y_test = train_test_split(dfx1,dfy1, test_size = 0.3)

def StackingEnsamble(X_train,X_test, y_train, y_test):
    def ARDRandomizedGridSearch():
        parametersList = []
        NumIter = random.randint(100,1000)
        tolerance = round(np.random.uniform(1e-30,1e-25),30)
        alpha1 = round(np.random.uniform(1e-20,1e-1),20)
        alpha2 = round(np.random.uniform(1e-20,1e-1),20)
        lambda1 = round(np.random.uniform(1e-20,1e-1),20)
        lambda2 = round(np.random.uniform(1e-20,1e-1),20)
        computeScore = random.choice([True,False])
        thresholdLambda = random.randint(1000,100000)
        verbose = True #random.choice([True,False])
        parametersList = [NumIter,
                          tolerance,
                          alpha1,
                          alpha2,
                          lambda1,
                          lambda2,
                          computeScore, 
                          thresholdLambda, 
                          verbose]
        print("list of randomly chosen parameters = {}".format(parametersList))
        file0.write("ARD_PARAMETERS = " + str(NumIter)+"\t"+\
                    str(tolerance)+"\t"+\
                    str(alpha1)+"\t"+\
                    str(alpha2)+"\t"+\
                    str(lambda1)+"\t"+\
                    str(lambda2)+"\t"+\
                    str(computeScore)+"\t"+\
                    str(thresholdLambda)+"\t"+\
                    str(verbose)+"\n")
        file0.flush()
        return parametersList 
    def BayesianRandomizedSearch():
        parameters = []
        nIter = random.randint(500,1000)
        tolerance = random.uniform(1e-4,1e-3)
        alpha1 = random.uniform(1e-5,1e-1)
        alpha2 = random.uniform(1e-5,1e-1)
        lambda1 = random.uniform(1e-5,1e-1)
        lambda2 = random.uniform(1e-5,1e-1)
        #alphaInit = random.unform()
        lambdaInit = random.choice([None, random.uniform(0,10)])
        computeScore = random.choice([True, False])
        fitIntercept = random.choice([True, False])
        verbose = random.choice([True, False])
        parameters = [nIter,
                      tolerance,
                      alpha1, 
                      alpha2,
                      lambda1, 
                      lambda2,
                      lambdaInit,
                      computeScore, 
                      fitIntercept,
                      verbose]
        print("Chosen parameters = {}".format(parameters))
        file0.write("Bayes_PARAM = " + str(nIter)+"\t"+\
                     str(tolerance)+"\t"+\
                     str(alpha1)+"\t"+\
                     str(alpha2)+"\t"+\
                     str(lambda1)+"\t"+\
                     str(lambda2)+"\t"+\
                     str(lambdaInit)+"\t"+\
                     str(computeScore)+"\t"+\
                     str(verbose)+"\n")
        file0.flush()
        return parameters
    def ElasticNetRandomSearch():
        parameters = []
        Alpha = random.uniform(-10,10)
        l1Ratio = random.uniform(0,1) 
        fitIntercept = random.choice([True,False])
        Precompute = False#random.choice([True,False]) 
        maxIter = random.randint(10000,100000)
        Tolerance = random.uniform(1e-30,1e-5)
        warmStart = False#random.choice([True, False])
        randomState = random.randint(0,50)
        Selection = random.choice(['cyclic', 'random'])
        parameters = [Alpha, 
                      l1Ratio, 
                      fitIntercept, 
                      Precompute, 
                      maxIter, 
                      Tolerance,
                      warmStart,
                      randomState,
                      Selection]
        print("ElasticNet Param = {}".format(parameters))
        file0.write("ElasticNet = " + str(Alpha)+ "\t"+\
                    str(l1Ratio) + "\t"+\
                    str(fitIntercept) + "\t"+\
                    str(Precompute) + "\t"+\
                    str(maxIter) + "\t"+\
                    str(Tolerance) + "\t"+\
                    str(warmStart) + "\t"+\
                    str(randomState) + "\t"+\
                    str(Selection) + "\n")
        file0.flush()
        return parameters
    def LassRandomParam():
        parameters = []
        Alpha = random.uniform(0.1, 10.0)
        fitIntercept = random.choice([True, False])
        maxIter = random.randint(1000,10000)
        tolerance = random.uniform(1e-30, 1e-10)
        warmStart = False #random.choice([True, False])
        randomState = random.choice([None, random.randint(0,50)])
        selection = random.choice(['cyclic', 'random'])
        parameters = [Alpha,
                      fitIntercept,
                      maxIter, 
                      tolerance,
                      warmStart,
                      randomState, 
                      selection]
        print("Chosen parameters LASSO  = {}".format(parameters))
        file0.write("Lasso Param = \t" +\
                    str(Alpha)+"\t"+\
                    str(fitIntercept)+"\t"+\
                    str(maxIter)+"\t"+\
                    str(tolerance)+"\t"+\
                    str(warmStart)+"\t"+\
                    str(randomState)+"\t"+\
                    str(selection)+"\n")
        file0.flush()
        return parameters
    def LinearRandomizedGridSearch():
        parameters = []
        fitIntercept = random.choice([True, False])
        parameters = [fitIntercept]
        print("Chosen Parameters Linear = {}".format(parameters[0]))
        file0.write("Linear = " + str(fitIntercept)+"\n")
        file0.flush()
        return parameters 
    def MLPParRandomSearch():
        parameters = []
        def hidLayerSize():
            numHidLayers = random.randint(2,5)
            HLS = []
            for i in range(numHidLayers):
                HLS.append(random.randint(10,200))
            return tuple(HLS)
        #Choosing activation function 
        ActFun = random.choice(['identity', 'logistic', 'tanh', 'relu'])
        #Choosing solver 
        Solver = random.choice(['lbfgs', 'adam'])#'lbfgs', 'sgd', 'adam'])
        #Choosing Alpha parameter L2 penalty parameter
        Alpha = random.uniform(1e-6, 1e-2)
        #Choosing Batch_size vlaue 
        BatchSize = random.randint(200,300)
        #Choosing learning rate
        LearnRate = random.choice(['constant', 'invscaling', 'adaptive'])
        #Choosing maximum number of iterations 
        MaxIter = random.randint(200,2000)
        #Choosing Tolerance 
        Tol = random.uniform(10e-10, 1e-4)
        #Maximum number of iterations without change 
        while True:
            nIter = random.randint(10,1000)
            if nIter < MaxIter:
                print("nIter smaller than MaxIter")
                break
            else:
                continue
        ######################################################
        #Appending Randomly Selected Data into Parameters list
        ######################################################
        parameters.append(hidLayerSize())
        parameters.append(ActFun)
        parameters.append(Solver)
        parameters.append(Alpha)
        parameters.append(BatchSize)
        parameters.append(LearnRate)
        parameters.append(MaxIter)
        parameters.append(Tol)
        parameters.append(nIter)
        #####################################################
        # Writing parameters into file0 _parameters.dat
        #####################################################
        file0.write("MLP_PARAM = \t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(parameters[0],
                                                                  parameters[1],
                                                                  parameters[2],
                                                                  parameters[3],
                                                                  parameters[4],
                                                                  parameters[5],
                                                                  parameters[6],
                                                                  parameters[7],
                                                                  parameters[8]))
        file0.flush()
        return parameters 
    def HUBRandomizedSearch():
        parameters = []
        Epsilon = random.uniform(1.1, 10)
        maxIter = random.randint(10000, 100000)
        Alpha = random.uniform(1e-10, 1e-3)
        warmStart = False#random.choice([True, False])
        fitIntercept = True#random.choice([True, False])
        tolerance = random.uniform(1e-30, 1e-20)
        parameters = [Epsilon, 
                      maxIter, 
                      Alpha, 
                      warmStart, 
                      fitIntercept, 
                      tolerance]
        print("Chosen parameters HUB = {}".format(parameters))
        file0.write("HUB = "+str(Epsilon)+"\t"+\
                    str(maxIter)+"\t"+\
                    str(Alpha)+"\t"+\
                    str(warmStart)+"\t"+\
                    str(fitIntercept)+"\t"+\
                    str(tolerance)+"\n")
        file0.flush()
        return parameters
    def RidgeRandomSearch():
        parameters = []
        Alpha = random.uniform(1.0, 1000.0)
        fitIntercept = random.choice([True,False])
        maxIter = random.randint(100, 100000)
        tolerance = random.uniform(1e-9,1e-3)
        solve = random.choice(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
        file0.write("Ridge = "+ str(Alpha)+"\t"+\
                    str(fitIntercept)+"\t"+\
                    str(maxIter)+"\t"+\
                    str(tolerance)+"\t"+\
                    str(solve)+"\n")
        file0.flush()
        parameters = [Alpha, 
                      fitIntercept,
                      maxIter,
                      tolerance, 
                      solve]
        print("Ridge parameters = {}".format(parameters))
        return parameters

    ###########################################################################
    # Calling Random Parameter functions 
    ###########################################################################
    ARDParam = ARDRandomizedGridSearch()
    BayesianParam = BayesianRandomizedSearch()
    ElasticNetParam = ElasticNetRandomSearch()
    LassoParam = LassRandomParam()
    LinParam = LinearRandomizedGridSearch()
    MLPParam = MLPParRandomSearch()
    HubParam = HUBRandomizedSearch()

    RdigeParam = RidgeRandomSearch()
    ###########################################################################
    # Calling Estimator functions and setting Parameter values
    ###########################################################################
    estimators = [('Ridge', Ridge(alpha=RdigeParam[0],
                             fit_intercept=RdigeParam[1], 
                             max_iter=RdigeParam[2], 
                             tol=RdigeParam[3], 
                             solver = RdigeParam[4])),
                  ('ARD', ARDRegression(n_iter = ARDParam[0],
                                        tol = ARDParam[1],
                                        alpha_1 = ARDParam[2], 
                                        alpha_2 = ARDParam[3],
                                        lambda_1 = ARDParam[4],
                                        lambda_2 = ARDParam[5], 
                                        compute_score = ARDParam[6],
                                        threshold_lambda = ARDParam[7], 
                                        verbose = ARDParam[8])),
                  ('Br', BayesianRidge(n_iter=BayesianParam[0], 
                                                   tol=BayesianParam[1], 
                                                   alpha_1=BayesianParam[2], 
                                                   alpha_2=BayesianParam[3], 
                                                   lambda_1=BayesianParam[4], 
                                                   lambda_2=BayesianParam[5],  
                                                   lambda_init=BayesianParam[6], 
                                                   compute_score=BayesianParam[7], 
                                                   fit_intercept=BayesianParam[8], 
                                                   verbose=BayesianParam[9])),
                  ('ENR', ElasticNet(alpha=ElasticNetParam[0],
                       l1_ratio=ElasticNetParam[1],
                       fit_intercept=ElasticNetParam[2],
                       precompute=ElasticNetParam[3],
                       max_iter=ElasticNetParam[4], 
                       tol=ElasticNetParam[5], 
                       warm_start = ElasticNetParam[6],
                       random_state = ElasticNetParam[7],
                       selection=ElasticNetParam[8])),
                  ('Lr', Lasso(alpha=LassoParam[0],
                                           fit_intercept=LassoParam[1], 
                                           max_iter=LassoParam[2], 
                                           tol=LassoParam[3], 
                                           warm_start=LassoParam[4],  
                                           random_state=LassoParam[5], 
                                           selection=LassoParam[6])),
                  ('LinR', LinearRegression(fit_intercept=LinParam[0])),
                  ('MLP', MLPRegressor(hidden_layer_sizes=MLPParam[0],
                                       activation= MLPParam[1],
                                       solver= MLPParam[2],
                                       alpha= MLPParam[3],
                                       batch_size = MLPParam[4],
                                       learning_rate = MLPParam[5],
                                       max_iter = MLPParam[6],
                                       tol = MLPParam[7],
                                       n_iter_no_change = MLPParam[8],
                                       verbose = True)),
                  ('Hubb', HuberRegressor(epsilon=HubParam[0], 
                                                    max_iter=HubParam[1], 
                                                    alpha=HubParam[2], 
                                                    warm_start=HubParam[3], 
                                                    fit_intercept=HubParam[4], 
                                                    tol=HubParam[5]))]

    model = StackingRegressor(estimators=estimators, 
                            final_estimator = None)
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
    if AvrAllR2Score > 0.99:
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
        file1.write("Final R^2 Test = {}\n".format(R2Test))
        file1.write("Final MAE Test = {}\n".format(MAETest))
        file1.write("Final RMSE Test = {}\n".format(RMSETest))
        file1.write("Final MAPE Test = {}\n".format(MAPETest))
        file1.write("########################################################\n")
        file1.flush()
        return R2Test
    else:
        return AvrAllR2Score
name = "Dck-2"
file0 = open("{}_CV5_Param.dat".format(name),"w")
file1 = open("{}_CV5_score.dat".format(name),'w')
while True:
    res = StackingEnsamble(X_train, X_test, y_train, y_test)
    if res > 0.999:
        print("Solution is Found!")
        break
    else:
        continue
file1.close()