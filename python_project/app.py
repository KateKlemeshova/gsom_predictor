from flask import Flask, request
import joblib
import numpy
import sklearn

MODEL_FOREST_PATH ='mlmodels/model_forest.pkl'
MODEL_CATBOOST_PATH ='mlmodels/model_catboost.pkl'
SCALER_X_path = 'mlmodels/scaler_x.pkl'
SCALER_Y_path = 'mlmodels/scaler_y.pkl'


app = Flask(__name__)

@app.route('/predict_price', methods=['GET'])
def predict():
    args = request.args
    open_plan = args.get('open_plan', default=-1, type = int )
    rooms = args.get('rooms', default=-1, type=int)
    area = args.get('area', default=-1, type=float)
    renovation = args.get('renovation', default=-1, type=int)

    #response ='open_plan:{}, rooms:{}, area:{}, renovation:{}'.format(open_plan, rooms, area, renovation)
    model_1 =joblib.load(MODEL_FOREST_PATH)
    sc_x = joblib.load(SCALER_X_path)
    sc_y = joblib.load(SCALER_Y_path)

    X = numpy.array([open_plan, rooms, area, renovation]).reshape(1,-1)
    X = sc_x.transform(X)
    result = model_1.predict(X)
    result = sc_y.inverse_transform(result.reshape(1,-1))

    return str(result[0][0])




if __name__ == "__main__":
    app.run(debug=True, port=5444, host='0.0.0.0')