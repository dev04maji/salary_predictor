from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load("model.joblib")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
   
    job_title = request.form.get("job_title", "")
    years_experience = float(request.form.get("years_experience", 0))
    location = request.form.get("location", "")
    education_level = request.form.get("education_level", "")
    remote_ratio = float(request.form.get("remote_ratio", 0))

    
    X = pd.DataFrame([{
        "job_title": job_title,
        "years_experience": years_experience,
        "location": location,
        "education_level": education_level,
        "remote_ratio": remote_ratio
    }])

    pred = model.predict(X)[0]
   
    predicted_salary = round(float(pred), 2)

    return render_template("index.html", predicted_salary=predicted_salary,
                           form_data={"job_title":job_title,"years_experience":years_experience,
                                      "location":location,"education_level":education_level,"remote_ratio":remote_ratio})

if __name__ == "__main__":
    app.run(debug=True)
