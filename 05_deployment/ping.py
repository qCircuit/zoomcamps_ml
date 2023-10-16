from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    print(1)
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=9898)