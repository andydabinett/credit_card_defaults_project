from tensorflow import keras
from keras import layers
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import EDA

def nerualnetwork():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(EDA.x_train_resampled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(EDA.x_train_resampled, EDA.y_train_resampled, epochs=30, batch_size=16)


    y_pred_ann = model.predict(EDA.x_val)
    y_pred_ann = (y_pred_ann > 0.5).astype(int)

    print(classification_report(EDA.y_val, y_pred_ann))
    print("AUC: ", roc_auc_score(EDA.y_val, y_pred_ann))

#neuralnetwork()
