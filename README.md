# Brain Tumor Detection Web Application

This is a web-based application built using Flask and TensorFlow for detecting brain tumors in MRI images. Users can upload MRI images, and the application will predict the presence of different types of tumors with corresponding confidence scores. Registered users can save their predictions and download them as PDFs. Guests can make predictions and view them in a log.

## Features

- **User Authentication**: Users can register, log in, and view their prediction history.
- **Image Upload & Tumor Detection**: Upload MRI images and get predictions based on a pre-trained deep learning model.
- **Prediction Logging**: Users can view their past predictions on their profile page. Guests' predictions are saved in a JSON file.
- **PDF Download**: Users can download their prediction results as PDF files.
- **Responsive Design**: The application is styled using Bootstrap for modern and responsive user experience.

## Technologies Used

- **Backend**: Flask, Flask-Login, Flask-SQLAlchemy
- **Machine Learning**: TensorFlow/Keras (Xception Model)
- **Database**: SQLite
- **Frontend**: Bootstrap, HTML, CSS
- **Image Processing**: OpenCV, NumPy
- **PDF Generation**: ReportLab

## Setup & Installation

### Requirements

- Python==3.10
- Flask==3.0.3
- TensorFlow==2.17.0
- Keras==3.6.0
- opencv-python==4.10.0.82
- numpy==1.26.4
- Flask-Login==0.6.3
- Flask-SQLAlchemy==3.1.1
- reportlab==4.2.5

### Installation Steps

1. Clone the repository:
   ```bash
   cd brain-tumor-detection
    ```
2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
3. Activate the virtual environment:
    - Windows:
      ```bash
      venv\Scripts\activate
      ```
    - macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
5. Run the application:
    ```bash
    python app.py
    ```

### Application structure
```
├── app.py                   # Main Flask application
├── wsgi.py                  # WSGI entry point for production servers
├── instance/                # Instance folder for SQLite database
├── notebooks/               # Jupyter notebooks for model experimentation
├── static/                  # Static files (CSS, JS, images, etc.)
│   ├── css/                 # CSS stylesheets
│   ├── js/                  # JavaScript files
│   ├── icons/               # PWA and site icons
│   ├── images/              # UI images
│   ├── models/              # Trained Keras model
│   ├── external_test_data/  # External test images
│   ├── test/                # Local test images
│   └── uploads/             # User-uploaded images
├── templates/               # HTML templates
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
├── linux_requirements.txt   # Python dependencies for Linux
└── README.md                # This README file
```

### Usage

1. Register a new account or log in with an existing account.
2. Upload an MRI image of a brain scan.
3. Click the "Predict" button to get the prediction results.
4. View the prediction results and download them as a PDF file.
5. Log out of the account when done.
