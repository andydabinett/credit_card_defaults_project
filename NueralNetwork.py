from tensorflow import keras
from keras import layers
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import EDA
from keras_tuner import RandomSearch, HyperParameters

#To avoid having an over-complicated (and thus, overfitted) model, we will omit hyper-parameter tuning using Keras Tuner. 
#This is a basic sequential neural network 


def neuralnetwork():
    model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(EDA.x_train_resampled.shape[1],)), 
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(EDA.x_train_resampled, EDA.y_train_resampled, epochs=30, batch_size=16)


    y_pred_ann = model.predict(EDA.x_val)
    y_pred_ann = (y_pred_ann > 0.5).astype(int)

    print(classification_report(EDA.y_val, y_pred_ann))
    print("AUC: ", roc_auc_score(EDA.y_val, y_pred_ann))


    from sklearn.metrics import confusion_matrix


    # Compute the confusion matrix
    cm = confusion_matrix(EDA.y_val, y_pred_ann)
    print(cm)

    # Extract counts
    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]

    # Number of positive predictions and negative predictions
    positive_predictions = true_positives + false_positives
    negative_predictions = true_negatives + false_negatives

    print((positive_predictions, negative_predictions))


#neuralnetwork()

