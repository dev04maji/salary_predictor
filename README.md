 Create a Virtual Environment:
python -m venv .venv

.venv\Scripts\activate


Install Dependencies:

pip install -r requirements.txt

Train the Model (Only if model.joblib is missing):

python train.py


This will generate the saved model:

model.joblib

Run the Web App:
python app.py


After running, open:

👉 http://127.0.0.1:5000/

The web app will load in your browser.

Use the App:

Enter job details

Click Predict Salary

View predicted salary instantly

Requirements:

Installed from requirements.txt:

flask
scikit-learn
pandas
joblib
