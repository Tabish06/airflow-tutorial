# from copy import deepcopy as copy

# simplicity_hash = {
#     'adaBoost' : 9.9,
#     'naiveBayes' : 10,
#     'randomForestClassifier': 9.9,
#     'decisionTree' : 10,
#     'logisticRegressionClassifier' : 10,
#     'svmClassifier' : 9.9,
#     'neuralNetwork' : 9.8,
#     'lda' : 10,
#     'qda' : 10
# }


# speed_hash = {
#     0.1 : 10,
#     0.5 : 9,
#     1 : 8,
#     2.5 : 7
# }
# weightage_hash = {
#     'f1_score' : 20,
#     'recall' : 10,
#     'specificity' : 10,
#     'precision' : 10,
#     'roc_auc_score' : 20,
# #     'log_loss_score' : 5,
#     'brier_score' : 10,
#     'accuracy' : 10,
#     'simplicity' : 5,
#     'time_for_pred' : 5
# }

log_loss_array=[]
def calcLogScore(log_loss_hash,value):
    return log_loss_hash[value]

def calcSpeedScore(speed,speed_hash):
    for (key,value) in speed_hash.items() :
        if speed < float(key) :
            return float(value)
    return 7

def isWeightThere(weightage_hash,score_name):
    return weightage_hash.get(score_name)

def update_f1_roc_balance(class_ratio,weightage_hash) :
    if weightage_hash['roc_auc_score'] != 0 and weightage_hash['f1_score'] != 0 :
        if class_ratio > 0.3 and class_ratio < 0.8 :
            weightage_hash['roc_auc_score'] += 5
            weightage_hash['f1_score'] -= 5
        else :
            weightage_hash['roc_auc_score'] -= 5
            weightage_hash['f1_score'] += 5   

def calculateWeightedScore(weightage_hash,scores,simplicity_hash,speed_hash,model_name):
    weightedScore = 0
    for (score_name, score) in scores.items():
        if isWeightThere(weightage_hash,score_name):
            if score_name == 'brier_score' :
                weightedScore += (weightage_hash[score_name] - (weightage_hash[score_name] * score))
            # elif score_name == 'log_loss_score':
            #     weightedScore += (calcLogScore(log_loss_normalized_score,score)/(max(log_loss_array) + min(log_loss_array) )) * weightage_hash[score_name]
            elif score_name == 'time_for_pred':
                    weightedScore += weightage_hash[score_name] * calcSpeedScore(score,speed_hash) / 10
            else :
                weightedScore += weightage_hash[score_name] * score 
    weightedScore += (simplicity_hash[model_name]/10) * weightage_hash['simplicity']
    return weightedScore


def calculateScore(storeScores,log_loss_array,simplicity_hash,weightage_hash,speed_hash,class_ratio) :
    # weightage_hash = copy(weightage_hash)
    class_ratio = float(class_ratio)
    printScores(storeScores)
    update_f1_roc_balance(class_ratio,weightage_hash)
    ## Code to normalize log loss
    if weightage_hash.get('log_loss_score'):
        max_loss = max(log_loss_array)
        log_loss_array.sort()
        log_loss_normalized_score = {}
        agg_loss = 0

        for (idx,value) in enumerate(log_loss_array) :
            if idx == 0 :
                log_loss_normalized_score[value] = max_loss
            else :
                agg_loss += value - (log_loss_array[idx - 1])
                log_loss_normalized_score[value] =  max_loss - agg_loss
    #######
    
    ##Code to calculate final score
    final_score = {}
    for (model_name,scores) in storeScores.items() :
        final_score[model_name] = calculateWeightedScore(weightage_hash,scores,simplicity_hash,speed_hash,model_name)
    print("========================FINAL SCORE============================")
    maxScore = 0 
    winner = None
    for (model_name,collated_score) in final_score.items() :
        if collated_score > maxScore :
            maxScore = collated_score
            winner = model_name
        print(model_name,'                   \t\t', round(collated_score,2))
    print(" ==========================AND THE WINNER IS========================")
    print(winner)
    print("====================================================================")
    return final_score


def printScores(storeScores):
    print("name\tf1 \trec \tspeci \tprec \troc \tlglss \tacc \tbrier \tpred_t ")
    for (classifier, scores) in storeScores.items() :
        print(classifier[0:6],end="\t")
        for (score_name,value) in scores.items() :
            print(round(value,3),end="\t")
        print()
