from flask import Flask, render_template, request, url_for
import pickle

# app = Flask(__name__)
app = Flask(__name__, static_folder='static')


# Load the trained machine learning model (you need to replace 'model.pkl' with your actual model file)
with open('rf_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route("/")
def open_app():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    company = float(request.form['company'])
    production_year = float(request.form['production_year'])
    mileage = float(request.form['mileage'])
    engine_type = float(request.form['engine_type'])
    gear_type = float(request.form['gear_type'])

    predicted_price = predict_price(model, company, production_year, mileage, engine_type, gear_type)
    return f'Predicted Car Price: ${predicted_price: }'


def predict_price(model,company, production_year, mileage, engine_type, gear_type):
    predicted_price = model.predict([[company, production_year, mileage, engine_type, gear_type]])
    return predicted_price[0]


port = 8000

if __name__ == "__main__":
    app.run(debug=True, port=port)
