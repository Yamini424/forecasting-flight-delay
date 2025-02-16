import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Attention
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib

data = pd.read_csv('/flight_delay_predict.csv')


features = ['AirTime', 'Distance', 'CRSDepTime', 'Month', 'DayOfWeek', 'Year', 'Origin', 'Dest']
data = data.dropna(subset=features + ['ArrDelay'])
X = data[features]
y = (data['ArrDelay'] > 15).astype(int)
categorical_features = ['Origin', 'Dest']
numeric_features = ['AirTime', 'Distance', 'CRSDepTime', 'Month', 'Year',  'DayOfWeek']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
X_processed = preprocessor.fit_transform(X)
joblib.dump(preprocessor, 'preprocessor.pkl')
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
X_train_processed = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_processed = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
X_train_processed = X_train_processed.astype('float32')
X_test_processed = X_test_processed.astype('float32')
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')
inputs = layers.Input(shape=(1, X_train_processed.shape[2]))
lstm_1 = LSTM(64, return_sequences=True)(inputs)
attn_output = Attention()([lstm_1, lstm_1])
dropout_1 = Dropout(0.2)(attn_output)
lstm_2 = LSTM(32, return_sequences=False)(dropout_1)
dropout_2 = Dropout(0.2)(lstm_2)
dense_1 = Dense(16, activation='relu')(dropout_2)
output = Dense(1, activation='sigmoid')(dense_1)
model = Model(inputs=inputs, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n✅ Model Created Successfully!")
history = model.fit(X_train_processed, y_train, epochs=3, batch_size=64, validation_data=(X_test_processed, y_test))
test_loss, test_accuracy = model.evaluate(X_test_processed, y_test)

print(f"\n✅ Model Evaluation on Test Data - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
model.save('flight_delay_model.h5')

