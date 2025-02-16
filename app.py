from flask import Flask, jsonify, request
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
model = load_model('flight_delay_model.h5', compile=False)  
preprocessor = joblib.load('preprocessor.pkl') 

@app.route('/api/flight/delay/predict', methods=['POST'])
def predict():
    data = request.json
    params = [
        data['AirTime'],
        data['Distance'],
        data['CRSDepTime'],
        data['DayOfWeek'],
        data['DayOfMonth'],
        data['Year'],
        data['Month'],
        data['Origin'],
        data['Dest'],
    ]

    user_input = pd.DataFrame([params],
                          columns=['AirTime', 'Distance', 'CRSDepTime', 'DayOfWeek', 'DayOfMonth', 'Year', 'Month', 'Origin', 'Dest'])
    user_input_processed = preprocessor.transform(user_input)

    # Reshape the input for LSTM (1 timestep, n features)
    user_input_processed = user_input_processed.reshape(1, 1, user_input_processed.shape[1])
    # Making the prediction with the trained model
    predictions = model.predict(user_input_processed)
    delay_minutes = predictions[0][0]
    data = {"message": f"The flight is delayed by {delay_minutes:.2f} minutes." if delay_minutes >= 0.5 else "The flight is not delayed."}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
