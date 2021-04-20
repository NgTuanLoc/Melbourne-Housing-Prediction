import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from model import SVM, RF, KNN, full_pipeline, data, features, test, SVM_Grid, KNN_Grid, RF_Random, convert

app = Flask(__name__)


@app.route('/index')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    X = [item for item in request.form.values()]
    X = [float(X[i]) if X[i].isnumeric() else np.nan for i in range(8) ] + [str(X[i]) for i in range(8, 15)]
    X = pd.DataFrame(data=[X], columns=features)
    item = data.head(1)
    for i in features:
        item[i] = X[i]
    final_features = full_pipeline.transform(item)
    prediction_svm = convert(SVM_Grid.best_estimator_.predict(final_features))
    prediction_knn = convert(KNN_Grid.best_estimator_.predict(final_features))
    prediction_rf = convert(RF_Random.predict(final_features))

    
    
    return render_template('index.html', prediction_svm="(SVM) House's Price should be {}".format(prediction_svm), prediction_knn="(KNN) House's Price should be {}".format(prediction_knn), prediction_rf="(RF) House's Price should be {}".format(prediction_rf) )
    

@app.route('/results',methods=['POST'])
def results():
    data = request.get_json(force=True)
    
    prediction = RF.predict(full_pipeline.transform(np.array(list(data.values()))))

    output = prediction
    return jsonify(output)



if __name__ == "__main__":
    app.run(debug=True)