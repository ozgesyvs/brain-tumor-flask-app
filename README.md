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

- Python 3.x
- Flask
- TensorFlow/Keras
- OpenCV
- NumPy
- Flask-Login
- Flask-SQLAlchemy
- ReportLab

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

├── app.py                   # Ana Flask uygulaması (Geliştirme sunucusu)
├── wsgi.py                  # Prodüksiyon sunucusu (Gunicorn vb.) giriş noktası
├── instance/                # SQLite veritabanı klasörü
├── notebooks/               # Model eğitim ve analiz defterleri (Jupyter)
├── static/
│   ├── css/                 # Stil dosyaları (Bootstrap vb.)
│   ├── js/                  # JavaScript dosyaları
│   ├── icons/               # PWA ve site ikonları
│   ├── images/              # Genel arayüz görselleri
│   ├── models/              # Eğitilmiş model dosyaları (xception11.keras)
│   ├── external_test_data/  # Dış kaynaklı deneme MRI görselleri
│   ├── test/                # Yerel test görselleri
│   └── uploads/             # Kullanıcıların yüklediği görseller 
├── templates/               # HTML sayfaları 
├── .gitignore               # Git'e gönderilmeyecek dosyaların listesi
├── requirements.txt         # Gerekli Python kütüphaneleri listesi
├── linux_requirements.txt   # Linux sunucular için optimize edilmiş kütüphane listesi
└── README.md                # Proje dökümantasyonu

### Usage

1. Register a new account or log in with an existing account.
2. Upload an MRI image of a brain scan.
3. Click the "Predict" button to get the prediction results.
4. View the prediction results and download them as a PDF file.
5. Log out of the account when done.