from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np 
import EDA 

def Decision_Tree():

    
    def hyperparameter_tuning():
        def train_model(params, x_train, y_train, x_val, y_val):
            model = DecisionTreeClassifier(**params)
            model.fit(x_train, y_train)
            y_val_pred = model.predict(x_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            return model, val_accuracy

        
        params = {'random_state': 42}
        model, val_accuracy = train_model(params, EDA.x_train_resampled, EDA.y_train_resampled, EDA.x_val, EDA.y_val)

        
        print(f'Validation Accuracy: {val_accuracy}')

        
        param_grid = {
            'max_depth': [10, 15, 20, 25], 
            'min_samples_split': [2, 5, 10], 
            'min_samples_leaf': [1, 2, 3, 4, 5],
            }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(EDA.x_train_resampled, EDA.y_train_resampled)

        # Print the best parameters and the best score
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_}')

        # Fit the model with the best parameters and validate
        model_best, val_accuracy_best = train_model(grid_search.best_params_, EDA.x_train_resampled, EDA.y_train_resampled, EDA.x_val, EDA.y_val)

        print(f'Validation Accuracy (best parameters): {val_accuracy_best}')

    #hyperparameter_tuning()

    
    #After hyperparameter tuning with the above function, we get the following results: 
    #Best parameters: {'max_depth': 25, 'min_samples_leaf': 1, 'min_samples_split': 2}
    #Train Score: 0.76
    #Validation Score: 0.69
    #Test Score: 0.69


    #To avoid having to run Grid Search every time we run this program, I will create and fit a new model with the apporpriate parameters. 
    tuned_model = DecisionTreeClassifier(max_depth = 25, min_samples_leaf = 1, min_samples_split = 2, random_state=42)

    tuned_model.fit(EDA.x_train_resampled, EDA.y_train_resampled)

    y_val_predictions_tuned = tuned_model.predict(EDA.x_val)

    print('Tuned Model Results: ')
    print(classification_report(EDA.y_val, y_val_predictions_tuned))
    print("AUC: ", roc_auc_score(EDA.y_val, y_val_predictions_tuned))
    print(confusion_matrix(EDA.y_val, y_val_predictions_tuned))


    #Our tuned model performs slightly bette then default models on the validation set.
    # However, because our tuned model was chosen based on performance on the validation set, 
    # it is possible this is due to overfitting.


    # Get the predicted probabilities for the positive class (default)
    y_pred_prob_dt = tuned_model.predict_proba(EDA.x_val)[:, 1]

    # Compute the false positive rate, true positive rate, and threshold for the ROC curve
    fpr_dt, tpr_dt, thresholds_roc_dt = roc_curve(EDA.y_val, y_pred_prob_dt)

    # Compute the precision, recall, and threshold for the Precision-Recall curve
    precision_dt, recall_dt, thresholds_pr_dt = precision_recall_curve(EDA.y_val, y_pred_prob_dt)
                                                                        
    # Compute the area under the ROC curve
    roc_auc_dt = auc(fpr_dt, tpr_dt)

    # Compute the area under the Precision-Recall curve
    pr_auc_dt = auc(recall_dt, precision_dt)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_dt, tpr_dt, label='Decision Tree (AUC = %0.2f)' % roc_auc_dt)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Decision Tree')
    plt.legend(loc="lower right")
    plt.show()

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_dt, precision_dt, label='Decision Tree (AUPRC = %0.2f)' % pr_auc_dt)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Decision Tree')
    plt.legend(loc="lower left")
    plt.show()

    #Why the discrepency between first AUC and graph AUC? 
    #The second method, which uses the predicted probabilities, is the standard approach for computing AUC because it provides a comprehensive 
    # view of the model's performance across all thresholds. The first method, with binary predictions, limits the AUC to a single point on the ROC curve.
    # Thus, the first method has a lower AUC 


    J = tpr_dt - fpr_dt
    idx = np.argmax(J)
    best_threshold = thresholds_roc_dt[idx]

    print("Best threshold:", best_threshold)

    print('Results on Test Dataset with Tuned Model:')
    y_test_pred_dt = tuned_model.predict(EDA.x_test)
    y_test_pred_dt = (y_test_pred_dt > best_threshold).astype(int)
    print(classification_report(EDA.y_test, y_test_pred_dt))
    print("AUC: ", roc_auc_score(EDA.y_test, y_test_pred_dt))

    
    #Compare to a default model to see if overfitting occured in our validation dataset 
    default_model = DecisionTreeClassifier(random_state=42)
    default_model.fit(EDA.x_train_resampled, EDA.y_train_resampled)

    y_pred_default = default_model.predict(EDA.x_test)
    print('Results on Test Dataset with Default Model: ')
    print(classification_report(EDA.y_test, y_pred_default))
    print("AUC: ", roc_auc_score(EDA.y_test, y_pred_default))


    #The difference in performance between the tuned model and the default model is negligible



Decision_Tree()


