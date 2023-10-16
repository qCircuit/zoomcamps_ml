import pickle
from flask import Flask, request, jsonify

app = Flask("credit")

with open("dv.bin", 'rb') as f:
    dv = pickle.load(f)
with open("model2.bin", 'rb') as f:
    model = pickle.load(f)

# client = {"job": "retired", "duration": 445, "poutcome": "success"}

@app.route("/predict", methods=["POST"])
def predict():
    client = request.get_json()

    x=dv.transform([client])
    yprob = model.predict_proba(x)[0, 1]
    ypred = model.predict(x)[0]

    result = {
        "probability": float(yprob),
        "credit": str(ypred)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)