import numpy as np
import math
from sklearn.metrics import confusion_matrix,roc_auc_score,log_loss,brier_score_loss

def calcMeasures(testY,predicted,predicted_proba):
    tn, fp, fn, tp = confusion_matrix(testY, predicted).ravel()
    precision = tp/(tp+fp)
    if math.isnan(precision):
        precision = 0
    recall = tp/(tp+fn)
    if precision == 0 :
        f1 = 0
    else :
        f1 = 2*(precision*recall)/(precision+recall)
    
    classification_error = (fp + fn)/(tp+tn+fp+fn)
    confidence_interval = 1.96 * np.sqrt( (classification_error * (1 - classification_error)) / (tp+tn+fp+fn))
    accuracy =  (tp+tn)/(tp+tn+fp+fn)
    roc_auc_score_model = accuracy
    if len(np.unique(predicted)) == 1 :
        roc_auc_score_mode = 0
    else :
        roc_auc_score_model = roc_auc_score(predicted == 1,testY == 1)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    log_loss_score = log_loss(testY,predicted_proba)
    brier_score = brier_score_loss(testY,predicted_proba[:,1])
    # class_ratio = (testY == "yes").sum()/len(testY)

    model_score = {'f1_score' : f1, "recall" : recall,"specificity": specificity, "precision" : precision,
         "roc_auc_score": roc_auc_score_model, "log_loss_score": log_loss_score, "accuracy" : accuracy,"brier_score" : brier_score} 
    print('Confidence Interval ',classification_error, '+/-  ', confidence_interval  )
    for (score_name,score) in model_score.items():
        print(score_name,' : ', score)
    return model_score
