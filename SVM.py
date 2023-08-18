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
        C=0.1,             
        kernel='rbf',      # Radial basis function kernel. 
        gamma='scale',      
        class_weight=None, # Since we've balanced the training dataset, we can keep this as None
        probability=True   
    )
    svm.fit(EDA.x_train_resampled, EDA.y_train_resampled)

    y_val_pred_SVM = svm.predict(EDA.x_val)
    print(classification_report(EDA.y_val, y_val_pred_SVM))
    print("AUC: ", roc_auc_score(EDA.y_val, y_val_pred_SVM))


    

    # Get the predicted probabilities for the positive class (default)
    y_pred_prob_svm = svm.predict_proba(EDA.x_val)[:, 1]

    # Compute the false positive rate, true positive rate, and threshold for the ROC curve
    fpr_svm, tpr_svm, thresholds_roc_svm = roc_curve(EDA.y_val, y_pred_prob_svm)

    # Compute the precision, recall, and threshold for the Precision-Recall curve
    precision_svm, recall_svm, thresholds_pr_svm = precision_recall_curve(EDA.y_val, y_pred_prob_svm)
                                                                        
    # Compute the area under the ROC curve
    roc_auc_svm = auc(fpr_svm, tpr_svm)

    # Compute the area under the Precision-Recall curve
    pr_auc_svm = auc(recall_svm, precision_svm)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_svm, tpr_svm, label='SVM (AUC = %0.2f)' % roc_auc_svm)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
    plt.legend(loc="lower right")
    plt.show()

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_svm, precision_svm, label='SVM (AUPRC = %0.2f)' % pr_auc_svm)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - SVM')
    plt.legend(loc="lower left")
    plt.show()


    J = tpr_svm - fpr_svm
    idx = np.argmax(J)
    best_threshold = thresholds_roc_svm[idx]

    print("Best threshold:", best_threshold)

    print('Results on Test Dataset:')
    y_test_pred_svm = svm.predict(EDA.x_test)
    print(classification_report(EDA.y_test, y_test_pred_svm))
    print("AUC: ", roc_auc_score(EDA.y_test, y_test_pred_svm))


#svm()

