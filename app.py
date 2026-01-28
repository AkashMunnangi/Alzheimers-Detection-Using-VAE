import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

from config import Config, Database
from model import VAE, LatentClassifier

# ===============================
# FLASK CONFIG
# ===============================
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_SIZE
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ===============================
# DEVICE
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# LOAD CLASS NAMES
# ===============================
with open("models/class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# ===============================
# LOAD MODELS
# ===============================
vae = VAE(latent_dim=128).to(DEVICE)
vae.load_state_dict(torch.load("models/vae.pth", map_location=DEVICE))
vae.eval()

classifier = LatentClassifier(latent_dim=128).to(DEVICE)
classifier.load_state_dict(torch.load("models/classifier.pth", map_location=DEVICE))
classifier.eval()

# ===============================
# PRETRAINED RESNET (OOD GATE)
# ===============================
resnet = resnet18(pretrained=True).to(DEVICE)
resnet.eval()

# ===============================
# TRANSFORMS
# ===============================
mri_transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

ood_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ===============================
# DATASET-FREE IMAGE REJECTION
# ===============================
def is_valid_brain_image(image: Image.Image) -> bool:
    """
    Dataset-free OOD rejection using ImageNet entropy
    """
    x = ood_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = resnet(x)
        probs = F.softmax(logits, dim=1)

    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
    confidence = probs.max().item()

    print(f"[OOD CHECK] Entropy={entropy:.2f}, Confidence={confidence:.2f}")

    # STRICT thresholds
    if entropy > 4.5:
        return False
    if confidence < 0.35:
        return False

    return True

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        db = Database.get_db()
        user = db.users.find_one({"username": username})

        if not user or user["password"] != password:
            flash("Invalid username or password", "danger")
            return redirect(url_for("login"))

        session["user"] = username
        return redirect(url_for("home"))

    return render_template("auth.html")

@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username")
    email = request.form.get("email")
    password = request.form.get("password")

    db = Database.get_db()
    if db.users.find_one({"username": username}):
        flash("Username already exists", "danger")
        return redirect(url_for("login"))

    db.users.insert_one({
        "username": username,
        "email": email,
        "password": password
    })

    flash("Registration successful. Please login.", "success")
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    file = request.files.get("file")
    if not file or file.filename == "":
        flash("No file selected", "danger")
        return redirect(url_for("home"))

    if not Config.allowed_file(file.filename):
        flash("Invalid file type", "danger")
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    image = Image.open(filepath).convert("RGB")

    # ===============================
    # HARD IMAGE REJECTION (KEY FIX)
    # ===============================
    if not is_valid_brain_image(image):
        os.remove(filepath)
        flash(
            "Invalid image detected. Please upload a valid brain MRI scan only.",
            "danger"
        )
        return redirect(url_for("home"))

    # ===============================
    # ALZHEIMER PREDICTION
    # ===============================
    tensor = mri_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mu, _ = vae.encode(tensor)
        outputs = classifier(mu)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    results = [
        {"class": CLASS_NAMES[i], "probability": round(float(probs[i] * 100), 2)}
        for i in range(len(CLASS_NAMES))
    ]

    results.sort(key=lambda x: x["probability"], reverse=True)

    # LOW CONFIDENCE REJECTION
    if results[0]["probability"] < 60:
        os.remove(filepath)
        flash(
            "Unclear MRI scan. Please upload a clearer brain MRI image.",
            "danger"
        )
        return redirect(url_for("home"))

    prediction = results[0]["class"]

    # ===============================
    # SAVE RESULT
    # ===============================
    db = Database.get_db()
    db.predictions.insert_one({
        "username": session["user"],
        "prediction": prediction,
        "probabilities": results,
        "timestamp": datetime.utcnow()
    })

    return render_template(
        "result.html",
        prediction=prediction,
        results=results,
        image_path=filepath
    )

# ===============================
# RUN APP
# ===============================
if __name__ == "__main__":
    Database.initialize()
    app.run(debug=True)
