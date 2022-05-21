import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)
model = pickle.load(open('model.pkl','rb'))
ureaModel = pickle.load(open('urea.pkl','rb'))
mopModel = pickle.load(open('mop.pkl','rb'))
tspModel = pickle.load(open('tsp.pkl','rb'))

@app.route("/")
def index():
    return '<h1>Hellow</h1>'


# ROUTES
@app.route('/hello', methods=['GET'])
def getHello():
	return 'This is a GET !'

@app.route('/hello', methods=['POST'])
def postHello():
    data=request.get_json(force=True)
    print(data['data'])
    prediction = model.predict([data['data']])
    output = prediction[0]
    return jsonify(output)
    return 'This is a PUT request!'

@app.route('/hello', methods=['PUT'])
def putHello():
	return 'This is a PUT request!'

@app.route('/hello', methods=['DELETE'])
def deleteHello():
	return 'This is a DELETE request!'



@app.route('/prediction', methods=['POST'])
def postPrediction():
    data=request.get_json(force=True)
    output={}
    UreaOutput=0
    MopOutput=0
    TspOutput=0

    prediction = ureaModel.predict([data['data']])
    UreaOutput = prediction[0]   

    prediction = tspModel.predict([data['data']])
    MopOutput = prediction[0] 


    prediction = mopModel.predict([data['data']])
    TspOutput = prediction[0]    

    output = [{
        "urea" : UreaOutput,
        "mop" : MopOutput,
        "tsp" : TspOutput,
    }]
    print(data['data'])

    return jsonify(output)
    return 'This is a POST request!'




if __name__ == "__main__":
  app.run()