from flask import Flask, request
import joblib
import numpy
import sklearn

MODEL_FOREST_PATH ='mlmodels/model_forest2_0.pkl'
MODEL_CATBOOST_PATH ='mlmodels/model_catboost2_0.pkl'
SCALER_X_path = 'mlmodels/scaler_x2_0.pkl'
SCALER_Y_path = 'mlmodels/scaler_y2_0.pkl'


app = Flask(__name__)

@app.route('/predict_price', methods=['GET'])
def predict():
    args = request.args
    model_version = args.get('model_version', default=-1, type=int)
    floor = args.get('floor', default=-1, type=int)
    open_plan = args.get('open_plan', default=-1, type = int )
    rooms = args.get('rooms', default=-1, type=int)
    area = args.get('area', default=-1, type=float)
    renovation = args.get('renovation', default=-1, type=int)

    model_1 = joblib.load(MODEL_CATBOOST_PATH)
    model_2 = joblib.load(MODEL_FOREST_PATH)
    sc_x = joblib.load(SCALER_X_path)
    sc_y = joblib.load(SCALER_Y_path)
    params = [floor, open_plan, rooms, area, renovation]

    X = numpy.array([params]).reshape(1,-1)
    X = sc_x.transform(X)

    if model_version == 1:
        result1 = model_1.predict(X)
        result1 = sc_y.inverse_transform(result1.reshape(1,-1))

        if any([i == -1 for i in params]):
            return '500 Internal server error', 500

        return str(result1[0][0])
    if model_version == 2:
        result2 = model_2.predict(X)
        result2 = sc_y.inverse_transform(result2.reshape(1,-1))

        if any([i == -1 for i in params]):
            return '500 Internal server error', 500

        return str(result2[0][0])

if __name__ == "__main__":
    app.run(debug=True, port=5444, host='0.0.0.0')