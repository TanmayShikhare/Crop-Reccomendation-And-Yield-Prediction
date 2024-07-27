from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib
from prettytable import PrettyTable

app = Flask(__name__)   # Initializing flask
# Loading our model:
model = pickle.load(open('models/RandomForest.pkl', "rb"))
model1 =joblib.load('notebook/RandomForestR.joblib')



@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        
        # Nitrogen
        nitrogen = float(request.form["nitrogen"])
        
        # Phosphorus
        phosphorus = float(request.form["phosphorus"])
        
        # Potassium
        potassium = float(request.form["potassium"])
        
        # Temperature
        temperature = float(request.form["temperature"])
        
        # Humidity Level
        humidity = float(request.form["humidity"])
        
        # PH level
        phLevel = float(request.form["ph-level"])
        
        # Rainfall
        rainfall = float(request.form["rainfall"])
        
        # Making predictions from the values:
        probas = model.predict_proba([[nitrogen, phosphorus, potassium, temperature, humidity, phLevel, rainfall]])[0]
        classes = np.argsort(probas)[::-1][:5]  # get top 5 classes
        class_names = ['rice', 'blackgram', 'banana', 'jute', 'coconut', 'apple', 'papaya', 'muskmelon', 'grapes',
                       'watermelon', 'kidneybeans', 'mango', 'mothbeans', 'pomegranate', 'coffee', 'cotton',
                       'orange', 'lentil', 'chickpea', 'mungbeans', 'maize', 'pigeonpeas']
        crop_recommendations = []
        for class_idx in classes:
            crop_name = class_names[class_idx].capitalize()
            if (crop_name == "Rice" or crop_name == "Blackgram" or crop_name == "Pomegranate" or crop_name == "Papaya"
                or crop_name == "Cotton" or crop_name == "Orange" or crop_name == "Coffee" or crop_name == "Chickpea"
                or crop_name == "Mothbeans" or crop_name == "Pigeonpeas" or crop_name == "Jute" or crop_name == "Mungbeans"
                or crop_name == "Lentil" or crop_name == "Maize" or crop_name == "Apple"):
                season = "Kharif- it should be sown at the beginning of the rainy season, between April and May."
            elif (crop_name == "Kidneybeans" or crop_name == "Coconut" or crop_name == "Grapes" or crop_name == "Banana"):
                season = "Rabi-it should be sown at the end of the monsoon and beginning of the winter season, between September and October."
            elif (crop_name == "Watermelon" or crop_name == "Muskmelon"):
                season = "Zaid- it should be sown between the Kharif and Rabi season, i.e., between March and June."
            else:
                season = "No specific season"
            crop_recommendations.append((crop_name, season))
        x = PrettyTable()
        x.field_names = ["Crop", "Best Time to Plant"]
        for crop, planting_time in crop_recommendations:
            x.add_row([crop, planting_time])
        crop_table = x.get_html_string()


    return render_template('CropResult.html', prediction_text=crop_recommendations, crop_table=crop_table)
    
        
@app.route("/predictyield", methods=["GET", "POST"])
def predictyield():
    if request.method == "POST":
       
        
        # Get the input values from the form
     Area = request.form['Area']
     Item = request.form['Item']
     average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
     avg_temp = float(request.form['avg_temp'])
     pesticides_tonnes = float(request.form['pesticides_tonnes'])
    
    # Create a dictionary with the input values
    input_data = {'Area': [Area], 'Item': [Item], 'average_rain_fall_mm_per_year': [average_rain_fall_mm_per_year], 
              'avg_temp': [avg_temp], 'pesticides_tonnes': [pesticides_tonnes]}

    # Create a Pandas data frame from the input data
    input_df = pd.DataFrame(input_data)
    model1_cols = input_df.columns.tolist()

    # Convert categorical variables to one-hot encoding
    categorical_cols = ['Area', 'Item']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)

   

    # Reorder columns to match the order in the trained model
    input_df = input_df[model1_cols]

    # Make the prediction
    yield_prediction = model1.predict(input_df)

    # Format the output
    output = round(float(yield_prediction[0]), 2)
    
    # Render the prediction to the user
    return render_template('Cropyield.html', yield_prediction=output)

if __name__ == '__main__':
    app.run(debug=True)

        