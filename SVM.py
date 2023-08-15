from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import EDA 
import numpy as np 

def svm():
    param_grid = {
        'C': [0.1, 1, 10],           # Regularization parameter
        'kernel': ['linear', 'rbf'], # Kernel type
        'gamma': ['scale', 'auto', 0.1, 1],  # Kernel coefficient for 'rbf' and 'poly'
        'degree': [2, 3, 4],         # Degree of polynomial kernel function
        'class_weight': ['balanced', {0: 0.1, 1: 0.9}, {0: 0.2, 1: 0.8}]
    }

    svm = SVC(
        C=0.1,             # Regularization parameter. You might want to explore values like 0.1, 1, 10, and 100.
        kernel='rbf',      # Radial basis function kernel. You can also try 'linear', 'poly', and 'sigmoid'.
        gamma='scale',     # Kernel coefficient. If this doesn't work well, you can try 'auto' or specific float values.
        class_weight=None, # Since you've balanced the training dataset, we can keep this as None. If performance is off, consider 'balanced'.
        probability=True   # If you need probabilities (useful for ROC/AUC or precision-recall curves).
    )
    svm.fit(EDA.x_train_resampled, EDA.y_train_resampled)

    y_pred_GBM = svm.predict(EDA.x_val)
    print(classification_report(EDA.y_val, y_pred_GBM))
    print("AUC: ", roc_auc_score(EDA.y_val, y_pred_GBM))
#svm

