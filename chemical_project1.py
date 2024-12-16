import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, jsonify, request
import numpy as np
import joblib

app = Flask(__name__)


@app.route('/train_model', methods=['GET'])
def train():

        file_path = 'Historical Alarm Cases.xlsx'
        df = pd.read_excel(file_path)
        print(df.head())
        print(df.info())

        x = df.iloc[:, 1:7]
        y = df['Spuriosity Index(0/1)']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=60)

        lr = LogisticRegression()
        lr.fit(X_train, y_train)

        predict = lr.predict(X_test)
        accuracy = accuracy_score(y_test, predict)
        print("Accuracy:", accuracy)

        joblib.dump(lr, 'train.pkl')
        return jsonify({"message": "Model trained successfully", "accuracy": accuracy})

@app.route('/test_model', methods=['POST'])
def test():

        pkl_file = joblib.load('train.pkl')
        test_data = request.get_json()
        required_keys = ['Ambient Temperature', 'Calibration',
                         'Unwanted substance deposition', 'Humidity',
                         'H2S Content', 'detected by']

        for key in required_keys:
            if key not in test_data:
                return jsonify({"error": f"Missing key in JSON: {key}"}), 400

        my_test_data = [
            test_data['Ambient Temperature'],
            test_data['Calibration'],
            test_data['Unwanted substance deposition'],
            test_data['Humidity'],
            test_data['H2S Content'],
            test_data['detected by']
        ]

        test_array = np.array(my_test_data, dtype=float).reshape(1, -1)
        y_pred = pkl_file.predict(test_array)
        prediction = "False Alarm, No Danger" if y_pred[0] == 1 else "True Alarm, Danger"

        return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
