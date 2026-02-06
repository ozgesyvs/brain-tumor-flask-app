import json
import os
import secrets
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from keras import models
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from werkzeug.security import (  # Import for hashing
    check_password_hash,
    generate_password_hash,
)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///myapp.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "testkey"  # Generates a secure 32-character hexadecimal string
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'login'  # Redirect to login if unauthorized access
login_manager.init_app(app)


# Database Models
class User(UserMixin, db.Model):
    """
    Represents a user in the database with unique username, password, email, and optional full name.
    Attributes:
        id (int): Primary key for the user.
        username (str): Unique username for the user.
        password (str): Password for the user.
        email (str): Unique email address for the user.
        full_name (str): Optional full name of the user.
        joined_date (datetime): Date and time when the user joined.
        predictions (relationship): Relationship to predictions made by the user.
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    full_name = db.Column(db.String(120), nullable=True)
    joined_date = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)



class Prediction(db.Model):
    """
    Model representing a prediction in the database.

    Attributes:
        id (int): Unique identifier for the prediction.
        user_id (int): Foreign key referencing the user associated with the prediction.
        filename (str): Name of the file for which the prediction was made.
        predicted_label (str): The label predicted for the file.
        confidence (float): Confidence level of the prediction.
        uploaded_time (datetime): Time when the prediction was uploaded.
        detection_time (float): Time taken for detection, if available.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(120), nullable=False)
    predicted_label = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    uploaded_time = db.Column(db.DateTime, default=datetime.utcnow)
    detection_time = db.Column(db.Float, nullable=True)

with app.app_context():
    db.create_all()

#Loading trained model
model = models.load_model('static/models/xception11.keras', compile=False)

# Labels for the tumor types
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Path to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
JSON_LOG_FILE = 'static/predictions_log.json'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@login_manager.user_loader
def load_user(user_id):
    """
    Loads a user from the database based on the user ID.
    Parameters:
        user_id (int): The ID of the user to load.
    Returns:
        User: The user object corresponding to the user ID.
    """
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    '''
    Register new users with unique credentials and store them in the database.
    Handles POST requests to register a new user with a unique username and email.
    If successful, redirects to the login page; otherwise, displays appropriate error messages.
    '''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_confirmation = request.form['password_confirmation']  # Get confirmation password
        email = request.form['email']
        full_name = request.form.get('full_name')

        if password != password_confirmation:
            flash('Passwords do not match.')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already exists.')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password, email=email, full_name=full_name)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    '''
    Route for user login functionality.
    Handles GET and POST requests for user login.
    If POST request, verifies user credentials and logs in if valid.
    Returns appropriate response messages.
    '''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()  # Fetch the user

        # Check if the user exists and the password is correct
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        
        flash('Login failed. Check your username and/or password.')
    return render_template('login.html')


@app.route('/profile')
@login_required
def profile():
    '''
    Route for displaying the user's profile page with their predictions.

    Returns:
        HTML template for the user's profile page with their predictions.
    '''
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    return render_template('profile.html', predictions=user_predictions)

@app.route('/logout')
@login_required
def logout():
    '''
    Logout the current user and redirect to the login page.

    Returns:
        redirect: Redirects to the login page.
    '''
    logout_user()
    return redirect(url_for('login'))

# Image Preparation and Prediction
def prepare_image(image, target_size=(150, 150)):
    '''
    Preprocesses an image for model prediction by resizing it to the target size and expanding its dimensions.

    Parameters:
        image (numpy.ndarray): The input image to be prepared.
        target_size (tuple): The target size to resize the image to. Defaults to (150, 150).

    Returns:
        numpy.ndarray: The prepared image ready for model prediction.
    '''
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    '''
    Route to render the index.html template when the root URL is accessed.
    '''
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
    '''
    Endpoint for making predictions based on uploaded images.

    Handles POST requests with image files, processes the image for prediction, saves the prediction results, and returns a rendered template with prediction details.

    Returns:
        str: Rendered HTML template with prediction details.
    '''
    send_notification_js = None
    
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    if image is None:
        return "Error loading image", 400

    image = prepare_image(image)

    start_time = time.time()
    predictions = model.predict(image)
    detection_time = time.time() - start_time

    predicted_class = np.argmax(predictions)
    predicted_label = labels[predicted_class]
    confidence = predictions[0][predicted_class] * 100
    confidence_formatted = f"{confidence:.2f}"

    if current_user.is_authenticated:
        new_prediction = Prediction(
            user_id=current_user.id,
            filename=filename,
            predicted_label=predicted_label,
            confidence=confidence_formatted,
            uploaded_time=datetime.now(),
            detection_time=round(detection_time, 2)
        )
        db.session.add(new_prediction)
        db.session.commit()
        send_notification_js = 'sendNotification();'
        return render_template('result.html', 
                               predicted_label=predicted_label, 
                               confidence=confidence_formatted,
                               uploaded_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                               detection_time=round(detection_time, 2),
                               filename=filename, send_notification_js=send_notification_js)
    else:
        # Save the prediction to JSON for guests
        guest_prediction = {
            "filename": filename,
            "predicted_label": predicted_label,
            "confidence": confidence_formatted,
            "uploaded_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_time": round(detection_time, 2)
        }

        # Append to JSON file
        if os.path.exists(JSON_LOG_FILE):
            with open(JSON_LOG_FILE, 'r+') as json_file:
                data = json.load(json_file)
                data.append(guest_prediction)
                json_file.seek(0)
                json.dump(data, json_file, indent=4)
        else:
            with open(JSON_LOG_FILE, 'w') as json_file:
                json.dump([guest_prediction], json_file, indent=4)

        return render_template('result.html', 
                               predicted_label=predicted_label, 
                               confidence=confidence_formatted,
                               uploaded_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                               detection_time=round(detection_time, 2),
                               filename=filename)

@app.route('/download/<int:prediction_id>')
@login_required
def download(prediction_id):
    '''
    Route for downloading a PDF file of a prediction result.

    Args:
        prediction_id (int): The unique identifier of the prediction.

    Returns:
        File: PDF file containing prediction details and additional notes.
    '''
    prediction = Prediction.query.get_or_404(prediction_id)
    pdf_filename = f"{prediction.filename.split('.')[0]}.pdf"
    
    # Create a PDF file
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.setTitle(f"Prediction Result for {prediction.filename}")

    # Add content to the PDF
    c.drawString(100, 750, f"Prediction Result for {prediction.filename}")
    c.drawString(100, 730, f"Predicted Label: {prediction.predicted_label}")
    c.drawString(100, 710, f"Confidence: {prediction.confidence}%")
    c.drawString(100, 690, f"Uploaded Time: {prediction.uploaded_time}")
    c.drawString(100, 670, f"Detection Time: {prediction.detection_time} seconds")
    
    # Additional details can be added here
    c.drawString(100, 650, "Additional Notes:")
    c.drawString(100, 630, "This PDF document contains the results of your MRI prediction.")
    c.drawString(100, 610, "If you have any questions, please contact your healthcare provider.")
    
    # Add the uploaded MRI image to the PDF
    image_path = os.path.join('static', 'uploads', prediction.filename)  # Construct the image path
    c.drawImage(image_path, 100, 400, width=200, height=200)  # Adjust width and height as needed

    # Include a footer or any additional sections as needed
    c.drawString(100, 370, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.save()

    # Return the PDF file as an attachment
    return send_file(pdf_filename, as_attachment=True)
@app.route('/about')
def about():
    '''
    Render the about.html template when the route '/about' is accessed.
    '''
    return render_template('about.html')

@app.route('/metrics')
def metrics():
    '''
    Render the metrics.html template for the '/metrics' route.
    '''
    return render_template('metrics.html')

@app.route('/dataset')
def dataset():
    '''
    Render the dataset.html template when the route '/dataset' is accessed.
    '''
    return render_template('dataset.html')

@app.route('/results')
def results():
    '''
    Render the results page with sorted predictions based on uploaded time.
    '''
    predictions = []
    if os.path.exists(JSON_LOG_FILE):
        with open(JSON_LOG_FILE, 'r') as json_file:
            predictions = json.load(json_file)
        predictions = sorted(predictions, key=lambda x: x['uploaded_time'], reverse=True)
    return render_template('results.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=False, port=5001)  # 5001 genelde boş olur.

