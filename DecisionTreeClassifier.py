from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import matplotlib.pyplot as plt
import EDA 

def Decision_Tree():

    def train_model(params, x_train, y_train, x_val, y_val):
        model = DecisionTreeClassifier(**params)
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        return model, val_accuracy

    # Initialize the model
    params = {'random_state': 42}
    model, val_accuracy = train_model(params, EDA.x_train_resampled, EDA.y_train_resampled, EDA.x_val, EDA.y_val)

    # Print Validation Accuracy
    print(f'Validation Accuracy: {val_accuracy}')

    # Tune Hyperparameters
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

    # Print Validation Accuracy with the best parameters
    print(f'Validation Accuracy (best parameters): {val_accuracy_best}')

    # Use the model for prediction on test set
    y_test_predictions = model_best.predict(EDA.x_test)
    test_accuracy = accuracy_score(EDA.y_test, y_test_predictions)
    print(f'Test Accuracy: {test_accuracy}')

    # Print a classification report and confusion matrix for more detailed performance analysis
    print(classification_report(EDA.y_test, y_test_predictions))
    print(confusion_matrix(EDA.y_test, y_test_predictions))

    # Plot feature importances
    plt.figure(figsize=(10, 5))
    plt.barh(EDA.feature_names, model_best.feature_importances_)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

#Decision_Tree()


