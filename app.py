from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Import custom transformers before loading
import __main__
from utils.transformers import TimeToDecimal, ToStringTransformer

__main__.TimeToDecimal = TimeToDecimal
__main__.ToStringTransformer = ToStringTransformer

app = Flask(__name__)

model = joblib.load("flight_delay_xgbregressor_model.pkl")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([data])

        prediction = model.predict(df)

        return jsonify({
            "success": True,
            "prediction": round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
