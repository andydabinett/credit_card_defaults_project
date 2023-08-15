from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import EDA 

def logistic_regression():

        def train_model(params, x_train, y_train, x_val, y_val):
                model = LogisticRegression(**params)
                model.fit(x_train, y_train)
                y_val_pred = model.predict(x_val)
                val_report = classification_report(y_val, y_val_pred) #Using classification report instead of accuracy. 
                return model, val_report                              #This will provide the performance of the model in more depth than just accuracy. 


        params1 = {'random_state': 42}
        model1, val_report1 = train_model(params1, EDA.x_train_resampled, EDA.y_train_resampled, EDA.x_val, EDA.y_val)

        # Initialize an empty dictionary to store models and their performances
        model_performances = {}

        model_performances[str(params1)] = val_report1 #Add new key-value pair to the dictionary. dictionary[new_key] = new_value
        print(f"Report 1 for parameters {params1}: \n{val_report1}")

        # define the hyperparameters to tune
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
                'penalty': ['l1', 'l2', 'none'], 
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        #After creating this param grid, decided I decided to try another method besides GridSearchCV that would use less computation, given the size of this dataset.
        #Instead, I will try manually testing parameters, and trying to deduce which ones performed the best based on varying results. 

        params2 = {'penalty': 'l1', 'C': 1.0, 'solver': 'liblinear', 'random_state': 42}
        model2, val_report2 = train_model(params2, EDA.x_train_resampled, EDA.y_train_resampled, EDA.x_val, EDA.y_val)
        print(f"Report 2 for parameters {params2}: \n{val_report2}")


        params3 = {'penalty': 'l2', 'C': 10, 'solver': 'lbfgs', 'random_state': 42}
        model3, val_report3 = train_model(params3, EDA.x_train_resampled, EDA.y_train_resampled, EDA.x_val, EDA.y_val)
        print(f"Report 3 for parameters {params3}: \n{val_report3}")

        #Tinkering with the parameters allows us to be a bit more precise on accounts marked as defaults. 
        #This lowered our recall a bit (meaning some true defaults might go missed) but increased our F1 score. 
        #We will use model 2 for testing. 
        #Ultimately, the conclusion from tinkering with the hyper-parameters was that while we can optimize for certain metrics (i.e. precision vs recall),
        # we won't ultimately improve the performance of our model. 


        y_test_predictions = model2.predict(EDA.x_test)

        test_report = classification_report(EDA.y_test, y_test_predictions)
        print(f"Test classification: \n{test_report}")
        print("AUC: ", roc_auc_score(EDA.y_test, y_test_predictions))
        #Performed nearly identical to validation set. 

        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        import matplotlib.pyplot as plt

        # Get the predicted probabilities for the positive class (fraud) from the logistic regres
        y_pred_prob_lr = model2.predict_proba(EDA.x_test)[:, 1]
        # Compute the false positive rate, true positive rate, and threshold for the ROC curve
        fpr, tpr, thresholds_roc = roc_curve(EDA.y_test, y_pred_prob_lr)
        # Compute the precision, recall, and threshold for the Precision-Recall curve
        precision, recall, thresholds_pr = precision_recall_curve(EDA.y_test, y_pred_prob_lr)
        # Compute the area under the ROC curve
        roc_auc = auc(fpr, tpr)
        # Compute the area under the Precision-Recall curve
        pr_auc = auc(recall, precision)
        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        # Plot the Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Logistic Regression (AUPRC = %0.2f)' % pr_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()


#logistic_regression()