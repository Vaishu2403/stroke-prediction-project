from flask import Flask, render_template, request, url_for
import joblib
import os
import numpy as np
import pickle


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('home.html', about_url=url_for('about'), contact_url=url_for('contact'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    print("Form submitted")
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level,
                  bmi, smoking_status]).reshape(1, -1)

    scaler_path = os.path.join('C:/Users/AMAR PAAPU/Desktop/Project/Stroke Prediction', 'models/scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    x = scaler.transform(x)

    model_path = os.path.join('C:/Users/AMAR PAAPU/Desktop/Project/Stroke Prediction', 'models/dt.sav')
    dt = joblib.load(model_path)

    Y_pred = dt.predict(x)

    if Y_pred[0] == 0:
        return render_template('nostroke.html', about_url=url_for('about'), contact_url=url_for('contact'))
    else:
        return render_template('stroke.html', about_url=url_for('about'), contact_url=url_for('contact'))
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)
    

