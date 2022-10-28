from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("forest.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    temp = data['temperature']
    oxy = data['oxygen']
    hum = data['humidity']
    final = [np.array([temp,oxy,hum])]
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    return output

if __name__ == '__main__':
    app.run(debug=True)
