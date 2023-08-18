from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import EDA 
import numpy as np 


def random_forest():
    model = RandomForestClassifier(random_state=42)
    model.fit(EDA.x_train_resampled, EDA.y_train_resampled)

    param_grid = {
        'n_estimators': [50, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
    }

    #Note: You can use combinatorics to see how many models your GridSearch will fit based on the number of parameters you have.
    #Originally, we had 288 combinations, which overwhelemed the GPU. 



    tuning = GridSearchCV(estimator=model, param_grid=param_grid, 
                            cv=5, n_jobs=-1, verbose=2, scoring='accuracy')  #Automatically uses cross-validation 

    tuning.fit(EDA.x_train_resampled, EDA.y_train_resampled)

    best_params = tuning.best_params_
    best_rf = tuning.best_estimator_

    print(best_params)
    #Based on the parameters returned, it seems our model may have overfit:
    #{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}
    #We will compare these results versus the default parameters 

    y_validation_predictions = tuning.predict(EDA.x_val)
    y_validation_predictions2 = model.predict(EDA.x_val)


    from sklearn.metrics import accuracy_score

    accuracy1 = accuracy_score(EDA.y_val, y_validation_predictions)
    print(f'Accuracy Score 1: {accuracy1}')
    accuracy2 = accuracy_score(EDA.y_val, y_validation_predictions2)
    print(f'Accuracy Score 2: {accuracy2}')


    val_report1 = classification_report(EDA.y_val, y_validation_predictions)
    print(f"Val Report 1 classification: \n{val_report1}")
    print("AUC: ", roc_auc_score(EDA.y_val, y_validation_predictions))

    
    val_report2 = classification_report(EDA.y_val, y_validation_predictions2)
    print(f"Val Report 2 classification: \n{val_report2}")
    print("AUC: ", roc_auc_score(EDA.y_val, y_validation_predictions2))

    
    
    #Now, having decided that we will use 'model' instead of 'tuning', we get predicted probabilities and plot proper ROC curve and evaluate on test dataset 

    # Get the predicted probabilities for the positive class (default)
    y_pred_prob_rf = model.predict_proba(EDA.x_val)[:, 1]

    # Compute the false positive rate, true positive rate, and threshold for the ROC curve
    fpr_rf, tpr_rf, thresholds_roc_rf = roc_curve(EDA.y_val, y_pred_prob_rf)

    # Compute the precision, recall, and threshold for the Precision-Recall curve
    precision_rf, recall_rf, thresholds_pr_rf = precision_recall_curve(EDA.y_val, y_pred_prob_rf)
                                                                        
    # Compute the area under the ROC curve
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    # Compute the area under the Precision-Recall curve
    pr_auc_rf = auc(recall_rf, precision_rf)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
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
    plt.plot(recall_rf, precision_rf, label='Random Forest (AUPRC = %0.2f)' % pr_auc_rf)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - GBM')
    plt.legend(loc="lower left")
    plt.show()


    J = tpr_rf - fpr_rf
    idx = np.argmax(J)
    best_threshold = thresholds_roc_rf[idx]

    print("Best threshold:", best_threshold)

    print('Results on Test Dataset:')
    y_test_pred_rf = model.predict(EDA.x_test)
    print(classification_report(EDA.y_test, y_test_pred_rf))
    print("AUC: ", roc_auc_score(EDA.y_test, y_test_pred_rf))




#random_forest()




