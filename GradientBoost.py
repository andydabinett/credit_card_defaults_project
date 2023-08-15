from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import EDA 
import numpy as np 

#GBM's are an ensemble learning method. The idea is to use a number of "weak learners"
# to create one strong learner. Weak learners are fit to the residuals of the previous iteration. 


def gradient_boost():
    param_grid_gbm = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
        'subsample': [0.8, 0.9, 1.0]
    }
    #While we will not use GridSearch for GBM, it is good to have a parameter grid to reference 
    #when we are picking and choosing our hyperparameters. 

    #High learning rate makes each individual tree influence the final output more and more. This can make the ensemble method 
    #learn faster, but it also makes it more prone to overfitting. 

    #In practice, setting a smaller learning_rate and using it in combination with a higher number of trees (n_estimators) 
    # is a common strategy to achieve better generalization. 


    model = GradientBoostingClassifier(max_depth = 3, n_estimators = 200, learning_rate = 0.01, random_state=42)
    model.fit(EDA.x_train_resampled, EDA.y_train_resampled)


    y_pred_GBM = model.predict(EDA.x_val)
    print(classification_report(EDA.y_val, y_pred_GBM))
    print("AUC: ", roc_auc_score(EDA.y_val, y_pred_GBM))



    # Get the predicted probabilities for the positive class (default)
    y_pred_prob_gbm = model.predict_proba(EDA.x_val)[:, 1]

    # Compute the false positive rate, true positive rate, and threshold for the ROC curve
    fpr_gbm, tpr_gbm, thresholds_roc_gbm = roc_curve(EDA.y_val, y_pred_prob_gbm)

    # Compute the precision, recall, and threshold for the Precision-Recall curve
    precision_gbm, recall_gbm, thresholds_pr_gbm = precision_recall_curve(EDA.y_val, y_pred_prob_gbm)
                                                                        
    # Compute the area under the ROC curve
    roc_auc_gbm = auc(fpr_gbm, tpr_gbm)

    # Compute the area under the Precision-Recall curve
    pr_auc_gbm = auc(recall_gbm, precision_gbm)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_gbm, tpr_gbm, label='Gradient Boosting Machines (AUC = %0.2f)' % roc_auc_gbm)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - GBM')
    plt.legend(loc="lower right")
    plt.show()

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_gbm, precision_gbm, label='Gradient Boosting Machines (AUPRC = %0.2f)' % pr_auc_gbm)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - GBM')
    plt.legend(loc="lower left")
    plt.show()

    #Why the discrepency between first AUC and graph AUC? 
    #The second method, which uses the predicted probabilities, is the standard approach for computing AUC because it provides a comprehensive 
    # view of the model's performance across all thresholds. The first method, with binary predictions, limits the AUC to a single point on the ROC curve.


    #We want to get the threshold closest to the top-left corner of the ROC graph. So we will use Eucliden distance 
    # to find out what the threshold is. 
    # Calculate the distance to the top-left corner for each threshold
  #  distances = np.sqrt((0 - fpr_gbm)**2 + (1 - tpr_gbm)**2)

    # Get the threshold for the point closest to the top-left corner
   # optimal_threshold = thresholds_roc_gbm[np.argmin(distances)]
  #  print(f"Optimal threshold: {optimal_threshold}")

    
    
    #Change the threshold in our model to implement the optimal one: 
   # def predict_with_threshold(probabilities, threshold):
    #    """Return class predictions using the given threshold."""
   #     return [1 if prob > threshold else 0 for prob in probabilities]

    
   # y_pred_optimal = predict_with_threshold(y_pred_prob_gbm, optimal_threshold)
   # print(classification_report(EDA.y_val, y_pred_optimal))
   # print("AUC: ", roc_auc_score(EDA.y_val, y_pred_optimal))

    #^^ This did not work, so we will scrap it (comment it out)


#gradient_boost()