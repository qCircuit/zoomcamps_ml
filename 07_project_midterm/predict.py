import pandas as pd
import pickle
from flask import Flask, request, jsonify

# load saved model
model_file = "model.pkl"
with open(model_file, "rb") as f:
    model = pickle.load(f)

app = Flask("phone")

@app.route("/predict", methods=["POST"])
def predict():
    phone = request.get_json()
    phone = pd.DataFrame([phone])
    ypr = model.predict(phone)

    result = {
        "phone price range": float(ypr)
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)