import pickle
from flask import Flask, request, jsonify
 
model_file = "model_C=1.0.bin"

with open(model_file, 'rb') as f:
    dv, model = pickle.load(f)

app = Flask("churn")

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    x=dv.transform([customer])
    ypred = model.predict_proba(x)[0,1]
    churn = ypred >= 0.5

    result = {
        "churn_probability": float(ypred),
        "churn": bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)