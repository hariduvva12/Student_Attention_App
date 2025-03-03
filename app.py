from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
import requests
from flask_session import Session
import bcrypt
import google.generativeai as genai

app = Flask(__name__)

# Flask session configuration
app.secret_key = "supersecretkey"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["student_attention"]
users_collection = db["users"]
collection = db["attention_data"]

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

EMOTION_LABELS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"
}

def generate_summary(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(text)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def classify_emotion(hr, vr):
    if hr >= 0 and vr >= 0:
        return "High Arousal - Pleasant", "Excited"
    elif hr < 0 and vr >= 0:
        return "Low Arousal - Pleasant", "Relaxed"
    elif hr < 0 and vr < 0:
        return "Low Arousal - Unpleasant", "Bored"
    else:
        return "High Arousal - Unpleasant", "Anxious"

@app.route("/")
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = users_collection.find_one({"username": username})
        if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))
        return "Invalid Credentials", 401
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    
    date = request.form["date"]
    subject = request.form["subject"]
    start_time = request.form["start_time"]
    end_time = request.form["end_time"]
    
    try:
        df = pd.read_excel(file)
        if "HR" in df.columns and "VR" in df.columns:
            df["Engagement"] = np.sqrt(df["HR"]**2 + df["VR"]**2)
            df[["Outer Label", "Emotion Quadrant"]] = df.apply(lambda row: classify_emotion(row["HR"], row["VR"]), axis=1, result_type='expand')
            df["Drowsiness Level"] = df["EAR"].apply(lambda x: "Drowsy" if x < 0.2 else "Alert") if "EAR" in df.columns else "N/A"
        
        row_data = {
            "date": date,
            "subject": subject,
            "start_time": start_time,
            "end_time": end_time,
            "data": df.to_dict(orient="records")
        }
        collection.insert_one(row_data)
        return redirect(url_for("classwise"))
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/classwise", methods=["GET"])
def classwise():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    data = list(collection.find())
    return render_template("classwise.html", data=data)
@app.route("/analytics/<entry_id>")
def analytics(entry_id):
    entry = collection.find_one({"_id": ObjectId(entry_id)})
    if not entry:
        return "Data not found", 404
    
    df = pd.DataFrame(entry["data"])
    graphs = {}
    
    for column in df.select_dtypes(include=["number"]).columns:
        plt.figure(figsize=(5, 3))
        plt.plot(df[column])
        plt.title(column)
        plt.xlabel("Index")
        plt.ylabel(column)
        
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        graphs[column] = base64.b64encode(img.getvalue()).decode()
    
    avg_hr = df["HR"].mean() if "HR" in df.columns else "N/A"
    avg_vr = df["VR"].mean() if "VR" in df.columns else "N/A"
    avg_ear = df["EAR"].mean() if "EAR" in df.columns else "N/A"
    engagement = df["Engagement"].mean() if "Engagement" in df.columns else "N/A"
    emotion = df["Emotion Quadrant"].mode()[0] if "Emotion Quadrant" in df.columns else "N/A"
    outer_label = df["Outer Label"].mode()[0] if "Outer Label" in df.columns else "N/A"
    drowsiness_level = df["Drowsiness Level"].mode()[0] if "Drowsiness Level" in df.columns else "N/A"

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except ValueError:
            return default

    avg_hr = safe_float(avg_hr)
    avg_vr = safe_float(avg_vr)
    avg_ear = safe_float(avg_ear)

    return render_template(
        "analytics.html",
        subject=entry["subject"],
        date=entry["date"],
        graphs=graphs,
        avg_hr=avg_hr,
        avg_vr=avg_vr,
        avg_ear=avg_ear,
        engagement=engagement,
        emotion=emotion,
        outer_label=outer_label,
        drowsiness_level=drowsiness_level
    )

@app.route("/summary/<entry_id>")
def summary(entry_id):
    entry = collection.find_one({"_id": ObjectId(entry_id)})
    if not entry:
        return "Data not found", 404

    df = pd.DataFrame(entry["data"])
    avg_hr = df["HR"].mean() if "HR" in df.columns else "N/A"
    avg_vr = df["VR"].mean() if "VR" in df.columns else "N/A"
    avg_ear = df["EAR"].mean() if "EAR" in df.columns else "N/A"

    if "Emotion Label Index" in df.columns:
        most_common_emotion_index = df["Emotion Label Index"].mode()[0]
        most_common_emotion = EMOTION_LABELS.get(most_common_emotion_index, "Unknown")
    else:
        most_common_emotion = "N/A"
    
    most_common_drowsiness = df["Drowsiness Level"].mode()[0] if "Drowsiness Level" in df.columns else "N/A"

    prompt = f"""
    This is the student attention data of a class session:
    - Average Heart Rate (HR): {avg_hr}
    - Average Valence Response (VR): {avg_vr}
    - Average Eye Aspect Ratio (EAR): {avg_ear}
    - Most Common Emotion: {most_common_emotion}
    - Most Common Drowsiness Level: {most_common_drowsiness}

    Based on this data:
    - What can you infer about the attention levels of the class?
    - Are students showing signs of drowsiness?
    - What does the emotional trend suggest about engagement?
    - Provide a brief summary of the class's attentiveness.
    """
    
    generated_summary = generate_summary(prompt)
    print("Generated Summary:", generated_summary)

    return jsonify({
        "summary": generated_summary,
        "avg_hr": avg_hr,
        "avg_vr": avg_vr,
        "avg_ear": avg_ear,
        "most_common_emotion": most_common_emotion,
        "most_common_drowsiness": most_common_drowsiness
    })

if __name__ == "__main__":
    app.run(debug=True)
