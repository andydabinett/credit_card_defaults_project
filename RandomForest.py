from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import EDA 


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
    #Now, we have 2 * 3 * 2 * 2 = 48 models. 


    tuning = GridSearchCV(estimator=model, param_grid=param_grid, 
                            cv=5, n_jobs=-1, verbose=2, scoring='accuracy')  #Automatically uses cross-validation 

    tuning.fit(EDA.x_train_resampled, EDA.y_train_resampled)

    best_params = tuning.best_params_
    best_rf = tuning.best_estimator_

    print(best_params)
    #Based on the parameters returned, it seems our model may have overfit:
    #{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}

    y_validation_predictions = tuning.predict(EDA.x_val)


    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(EDA.y_val, y_validation_predictions)

    test_report = classification_report(EDA.y_val, y_validation_predictions)
    print(f"Test classification: \n{test_report}")
    print("AUC: ", roc_auc_score(EDA.y_val, y_validation_predictions))

#random_forest()




