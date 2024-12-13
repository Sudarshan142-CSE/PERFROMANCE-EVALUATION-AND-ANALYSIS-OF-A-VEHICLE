# app.py
import pickle
import numpy as np
from flask import Flask, render_template, request

# Load the Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    model2 = pickle.load(file)




# Create the Flask app
app = Flask(__name__)

# Mapping for categorical variables
brand_mapping = {
    'Tesla ': 0, 'Volkswagen ': 1, 'Polestar ': 2, 'BMW ': 3,
    'Honda ': 4, 'Lucid ': 5, 'Peugeot ': 6, 'Audi ': 7,
    'Mercedes ': 8, 'Nissan ': 9, 'Hyundai ': 10, 'Porsche ': 11,
    'MG ': 12, 'Mini ': 13, 'Opel ': 14, 'Skoda ': 15,
    'Volvo ': 16, 'Kia ': 17, 'Renault ': 18, 'Mazda ': 19,
    'Lexus ': 20, 'CUPRA ': 21, 'SEAT ': 22, 'Lightyear ': 23,
    'Aiways ': 24, 'DS ': 25, 'Citroen ': 26, 'Jaguar ': 27,
    'Ford ': 28, 'Byton ': 29, 'Sono ': 30, 'Smart ': 31,
    'Fiat ': 32
}

model_mapping = {
    'Model 3 Long Range Dual Motor': 0, 'ID.3 Pure': 1, '2': 2, 
    # Add all other models as per your dataset
}

rapid_charge_mapping = {'Yes': 0, 'No': 1}
power_train_mapping = {'AWD': 0, 'RWD': 1, 'FWD': 2}
plug_type_mapping = {
    'Type 2 CCS': 0, 'Type 2 CHAdeMO': 1, 'Type 2': 2, 'Type 1 CHAdeMO': 3
}
body_style_mapping = {
    'Sedan': 0, 'Hatchback': 1, 'Liftback': 2, 'SUV': 3,
    'Pickup': 4, 'MPV': 5, 'Cabrio': 6, 'SPV': 7, 'Station': 8
}
segment_mapping = {
    'D': 0, 'C': 1, 'B': 2, 'F': 3, 'A': 4, 'E': 5, 'N': 6, 'S': 7
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Fetch user input
        brand = request.form['brand']
        model = request.form['model']
        accel_sec = float(request.form['accel_sec'])
        top_speed = float(request.form['top_speed'])
        range_km = float(request.form['range_km'])
        fast_charge = float(request.form['fast_charge'])
        rapid_charge = request.form['rapid_charge']
        power_train = request.form['power_train']
        plug_type = request.form['plug_type']
        body_style = request.form['body_style']
        segment = request.form['segment']
        seats = int(request.form['seats'])
        price_euro = float(request.form['price_euro'])

        # Convert categorical inputs to numerical
        input_data = np.array([[
            brand_mapping[brand], model_mapping[model], accel_sec,
            top_speed, range_km, fast_charge, rapid_charge_mapping[rapid_charge],
            power_train_mapping[power_train], plug_type_mapping[plug_type],
            body_style_mapping[body_style], segment_mapping[segment],
            seats, price_euro
        ]])

        # Make prediction
        prediction = model2.predict(input_data)

        return render_template('index.html', prediction=prediction[0],
                               brand_mapping=brand_mapping,
                               model_mapping=model_mapping)

    return render_template('index.html', prediction=None,
                           brand_mapping=brand_mapping,
                           model_mapping=model_mapping)

if __name__ == '__main__':
    app.run(debug=True)
