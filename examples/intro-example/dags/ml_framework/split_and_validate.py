from sklearn.naive_bayes import GaussianNB
from sklearn import tree
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import numpy as np
import pandas as pd
# import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn import metrics
import time    
# import math

from ml_framework.neuralNetwork import neuralNetwork
from ml_framework.calcMeasures import calcMeasures
# def calcMeasures(testY,predicted,predicted_proba):
#     tn, fp, fn, tp = confusion_matrix(testY, predicted).ravel()
#     precision = tp/(tp+fp)
#     if math.isnan(precision):
#         precision = 0
#     recall = tp/(tp+fn)
#     if precision == 0 :
#         f1 = 0
#     else :
#         f1 = 2*(precision*recall)/(precision+recall)
    
#     classification_error = (fp + fn)/(tp+tn+fp+fn)
#     confidence_interval = 1.96 * np.sqrt( (classification_error * (1 - classification_error)) / (tp+tn+fp+fn))
#     accuracy =  (tp+tn)/(tp+tn+fp+fn)
#     roc_auc_score_model = accuracy
#     if len(np.unique(predicted)) == 1 :
#         roc_auc_score_mode = 0
#     else :
#         roc_auc_score_model = roc_auc_score(predicted == 1,testY == 1)
#     sensitivity = tp / (tp + fn)
#     specificity = tn / (tn + fp)
#     log_loss_score = log_loss(testY,predicted_proba)
#     brier_score = brier_score_loss(testY,predicted_proba[:,1])
#     # class_ratio = (testY == "yes").sum()/len(testY)

#     model_score = {'f1_score' : f1, "recall" : recall,"specificity": specificity, "precision" : precision,
#          "roc_auc_score": roc_auc_score_model, "log_loss_score": log_loss_score, "accuracy" : accuracy,"brier_score" : brier_score} 
#     print('Confidence Interval ',classification_error, '+/-  ', confidence_interval  )
#     for (score_name,score) in model_score.items():
#         print(score_name,' : ', score)
#     return model_score

    
def predict_and_score(clf,train,test,trainY,testY):
    predicted = clf.predict(train)
    predicted_proba = clf.predict_proba(train)
    model_name = (str(type(clf)).split('.')[-1].replace("'>",""))

    print('Train')
    calcMeasures(trainY,predicted,predicted_proba)
    print('Test')
    predicted = clf.predict(test)
    t1 = time.time()
    predicted_proba = clf.predict_proba(test)
    time_for_pred = round(time.time()-t1, 3)
    model_score = calcMeasures(testY,predicted,predicted_proba)
    # log_loss_array.append(model_score['log_loss_score'])
    model_score['time_for_pred'] = time_for_pred
#     storeScores[model_name] = model_score
    return model_score

# def kfold_fit(clf,train,trainY):
#     splits = 5
#     kf = StratifiedKFold(n_splits=splits)
#     kf.get_n_splits(train,trainY)
#     average_score = {}
#     for (train_idx,test_idx) in kf.split(train,trainY):
#         clf.fit(train.iloc[train_idx],trainY.iloc[train_idx])
#         model_score = predict_and_score(clf,train.iloc[train_idx],train.iloc[test_idx],trainY.iloc[train_idx],trainY.iloc[test_idx])
#         for (score_name,score) in model_score.items():
#             if average_score.get(score_name) :
#                 average_score[score_name] += score
#             else :
#                 average_score[score_name] = score
#     for (score_name,score) in average_score.items():
#         average_score[score_name] = score/splits
#     model_name = (str(type(clf)).split('.')[-1].replace("'>",""))
#     storeScores[model_name] = average_score
#     return clf 


# def nn_kfold(train,trainY,column_pred='+_-'):
#     es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)
    
# #     model.fit(train, trainY, batch_size = bsize, epochs = 1000, verbose = 1,validation_data = (test, testY)
# #               ,callbacks=[es])
#     splits = 5
#     bsize = 32

#     kf = StratifiedKFold(n_splits=splits)
#     kf.get_n_splits(train,trainY.loc[:,[column_pred]])
#     average_score = {}
#     for train_idx,test_idx in kf.split(train,trainY.loc[:,[column_pred]]) :
#         model = nn_model(train.shape[1])
#         model.fit(train.iloc[train_idx], trainY.iloc[train_idx], batch_size = bsize, epochs = 1000, verbose = 1,
#                   validation_data = (train.iloc[test_idx], trainY.iloc[test_idx])
#               ,callbacks=[es])

#         t1 = time.time()
#         prob = model.predict(train.iloc[test_idx])
#         time_taken = round(time.time()-t1, 3)

#         class_pred = np.argmax(prob,axis=1) 
#         model_score = calcMeasures(trainY.iloc[test_idx][column_pred], class_pred, prob )
#         model_score['time_for_pred'] = time_taken
#         for (score_name,score) in model_score.items():
#             if average_score.get(score_name) :
#                 average_score[score_name] += score
#             else :
#                 average_score[score_name] = score
#     for (score_name,score) in average_score.items():
#         average_score[score_name] = score/splits

# #         log_loss_array.append(model_score['log_loss_score'])
#     storeScores['NeuralNetwork'] = average_score
    

def lda(train,test,trainY,testY):
    clf = LinearDiscriminantAnalysis()
    clf = clf.fit(train,trainY)
    model_score = predict_and_score(clf,train,test,trainY,testY)
    return clf,model_score

def qda(train,test,trainY,testY):
    clf = QuadraticDiscriminantAnalysis()
    clf = clf.fit(train,trainY)
    model_score = predict_and_score(clf,train,test,trainY,testY)    

    return clf,model_score


def logisticRegressionClassifier(train,test,trainY,testY):
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(train, trainY)
    clf = clf.fit(train, trainY)
    # print(scores)
    model_score = predict_and_score(clf,train,test,trainY,testY)    

    return clf,model_score

  
    
def randomForestClassifier(train,test,trainY,testY):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0)
    clf = clf.fit(train, trainY)
    model_score = predict_and_score(clf,train,test,trainY,testY)
    return clf,model_score


def adaBoost(train,test,trainY,testY):
    clf = AdaBoostClassifier(n_estimators=100)
    clf = clf.fit(train, trainY)
    model_score = predict_and_score(clf,train,test,trainY,testY)    

    return clf,model_score


def svmClassifier(train,test,trainY,testY):
    clf = SVC(gamma='scale',probability=True)
    clf = clf.fit(train, test)
    model_score = predict_and_score(clf,train,test,trainY,testY) 
    return clf,model_score

    
def naiveBayes(train,test,trainY,testY):
    gnb = GaussianNB()
    gnb = gnb.fit(train, trainY)
    model_score = predict_and_score(gnb,train,test,trainY,testY)
    return gnb,model_score

# def svmClassifier(train,test,trainY,testY):
#     clf = SVC(probability=True)
#     clf = clf.fit(train, trainY)
#     predict_and_score(clf,train,test,trainY,testY)
#     return clf

def decisionTree(train,test,trainY,testY):
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(train, trainY)
    model_score = predict_and_score(clf,train,test,trainY,testY)    

    return clf,model_score

def predict_and_score_nn(model,test,testY):
    op = model.predict(test)
    calcMeasures(testY,op,op)
 



def score(index,algo):
    train = pd.read_csv(f'/usr/local/airflow/train_{index}.csv')
    test = pd.read_csv(f'/usr/local/airflow/test_{index}.csv')
    trainY = pd.read_csv(f'/usr/local/airflow/trainY_{index}.csv')
    testY = pd.read_csv(f'/usr/local/airflow/testY_{index}.csv')
    if algo == 'neuralNetwork' :
        model,score = globals()[algo](train,test,trainY,testY)
    else :
        model,score = globals()[algo](train,test,trainY.iloc[:,[1]],testY.iloc[:,[1]])
    return score
