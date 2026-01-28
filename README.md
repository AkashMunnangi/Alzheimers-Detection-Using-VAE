# 🧠 Alzheimer’s Detection Using Variational Autoencoder (VAE)

## About the Project

Alzheimer’s disease affects memory and thinking ability, and early diagnosis plays an important role in patient care.
This project focuses on detecting Alzheimer’s disease using **brain MRI images** with the help of machine learning.

Instead of using a traditional CNN model, a **Variational Autoencoder (VAE)** is used to learn important features from MRI images. These features are then used to classify the image as **Alzheimer’s** or **Normal**.
The trained model is integrated into a **Flask web application**, where users can upload MRI images and get predictions.



## Why VAE?

* VAE helps in learning meaningful latent features from MRI images
* It is more robust to noise compared to direct image classification
* Works well for medical image analysis


## Project Execution

Youtube Link: https://youtu.be/vZDnyEG-_wM?si=cQ1pUBXvgnnWJjgb


## Tech Stack

* Programming Language: Python 3.10

* Deep Learning Framework: PyTorch (used for building and training the VAE and classifier models)

* Web Framework: Flask (used to create the web application and handle user requests)

* Database: MongoDB Atlas (used to store user details and prediction history)

* Frontend: HTML, CSS, JavaScript (used to build the user interface)

* Model Type: Variational Autoencoder (VAE) with a latent space classifier

Tools: VS Code / PyCharm

## Project Structure

```
Alzheimers-Detection-Using-VAE/
├── app.py
├── train.py
├── model.py
├── config.py
├── requirements.txt
├── models/
├── templates/
├── static/
├── train/
├── test/
└── README.md
```


## How the Project Works

1. User registers and logs into the system
2. MRI image is uploaded through the web interface
3. Uploaded image is validated to ensure it is a valid MRI
4. VAE extracts features from the image
5. Classifier predicts Alzheimer’s or Normal
6. Result is displayed and stored in the database



## Dataset

* Brain MRI images
* Two classes:

  * Alzheimer’s
  * Normal



## How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/Alzheimers-Detection-Using-VAE.git
cd Alzheimers-Detection-Using-VAE
```

### Step 2: Install required libraries

```bash
pip install -r requirements.txt
```

### Step 3: Set environment variables

Create a `.env` file and add:

```
MONGODB_URI=your_mongodb_atlas_connection_string
DATABASE_NAME=alzheimer_db name from your mongodb atlas
SECRET_KEY=your_secret_key
```

### Step 4: Train the model

```bash
python train.py

I reccomend you to use GPU for faster training speed. CPU will to too much time!!!
```

### Step 5: Run the application

```bash
python app.py
```

Open browser and go to:

```
http://127.0.0.1:5000
```



## Testing

* Valid MRI images give correct predictions
* Non-MRI images are rejected before prediction



## Note

This project is developed for **academic purposes** and is intended to act as a **decision-support system**, not a replacement for medical diagnosis.



## Author

Akash Munnangi
B.Tech – Computer Science Engineering
akashmunnangi@gmail.com

