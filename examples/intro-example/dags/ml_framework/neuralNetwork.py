
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout,LeakyReLU,Activation
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import numpy as np
from ml_framework.calcMeasures import calcMeasures
import time
def nn_model(dim):
    model = Sequential()
    #Layer 1
    model.add(Dense(10, input_dim = dim,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
    #Layer 2
    model.add(Dense(5,activation='relu'))
    model.add(Dropout(0.25))
    
    
    #output layer
    model.add(Dense(3,activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(5,activation='relu'))
    model.add(Dropout(0.5))

#     model.add(Dense(100,activation='tanh'))
    
    model.add(Dense(2,activation='softmax'))
    
#     model.add(Activation('softmax'))
    optimizer = optimizers.RMSprop(lr=0.00001)

    model.compile(optimizer = optimizer,loss = 'categorical_crossentropy',metrics = ['accuracy'])
    #Fit/Train the model
    return model

def neuralNetwork(train,test,trainY,testY):
    bsize = 32

    model = nn_model(train.shape[1])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    model.fit(train, trainY, batch_size = bsize, epochs = 20, verbose = 1,validation_data = (test, testY)
              ,callbacks=[es])

    t1 = time.time()
    prob = model.predict(test)
    time_taken = round(time.time()-t1, 3)

    prob[:,[0, 1]] = prob[:,[1, 0]]


    class_pred = np.argmax(prob,axis=1)
    # print(testY.iloc[:,[1]])
    # B = np.where(testY.iloc[:,[1]] <= 0.5, 1, 0)
    
    model_score = calcMeasures(testY.iloc[:,[1]], class_pred, prob )
    model_score['time_for_pred'] = time_taken

    # log_loss_array.append(model_score['log_loss_score'])
    # storeScores['NeuralNetwork'] = model_score
    return model,model_score
