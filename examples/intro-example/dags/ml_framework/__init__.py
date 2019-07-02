import lazy_import
lazy_import.lazy_module("tensorflow")
lazy_import.lazy_module("tensorflow.keras.layers")
lazy_import.lazy_callable("tensorflow.keras.layers.Dense")
lazy_import.lazy_callable("tensorflow.keras.layers.Dropout")
lazy_import.lazy_callable("tensorflow.keras.layers.LeakyReLU")
lazy_import.lazy_callable("tensorflow.keras.layers.Activation")
lazy_import.lazy_callable("tensorflow.keras.callbacks.EarlyStopping")
lazy_import.lazy_callable("tensorflow.keras.models.Sequential")
lazy_import.lazy_module("tensorflow.keras.regularizers")
lazy_import.lazy_module("tensorflow.keras.optimizers")
lazy_import.lazy_callable("sklearn.naive_bayes.GaussianNB")
lazy_import.lazy_module("sklearn.tree")
lazy_import.lazy_callable("sklearn.linear_model.LogisticRegression")
lazy_import.lazy_callable("sklearn.ensemble.RandomForestClassifier")
lazy_import.lazy_callable("sklearn.ensemble.AdaBoostClassifier")
lazy_import.lazy_callable("sklearn.metrics.confusion_matrix")
lazy_import.lazy_callable("sklearn.metrics.roc_auc_score")
lazy_import.lazy_callable("sklearn.metrics.log_loss")
lazy_import.lazy_callable("sklearn.metrics.brier_score_loss")

# from sklearn.naive_bayes import GaussianNB
# from sklearn import tree
# # from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Dense,Dropout,LeakyReLU,Activation
# from tensorflow.keras.callbacks import EarlyStopping

# from tensorflow.keras.models import Sequential
# from tensorflow.keras import regularizers
# from tensorflow.keras import optimizers