from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Define your model training code here using the latest scikit-learn version
# Example:
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load or create your dataset
data = load_iris()
X = data.data
y = data.target

# Create and train a DecisionTreeClassifier model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model using joblib
joblib.dump(model, 'fetal_health1.pkl')

@app.route("/")
def f():
    return render_template("index.html")

@app.route("/inspect")
def inspect():
    return render_template("inspect.html")

@app.route('/home', methods=['POST'])
def home():
    if request.method == 'POST':
        input_data = request.form
        print(input_data)  # Debugging line

        # Access the form data and convert it to float
        feature1 = float(input_data.get('feature1'))
        feature2 = float(input_data.get('feature2'))
        feature3 = float(input_data.get('feature3'))
        feature4 = float(input_data.get('feature4'))
    
        # Make predictions using the loaded model
        input_features = [feature1, feature2, feature3,feature4]
        output = model.predict([input_features])

        # Process the output and return it
        res =  str(output[0])
        if res=="1":
            result="Normal"
        elif res=="2":
            result="Suspect"
        elif res=="3":
            result="Pathological"
        return render_template('output.html', output=result)


if __name__ == "__main__":
    app.run(debug=True)
