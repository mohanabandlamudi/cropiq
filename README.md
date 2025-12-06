
# CROPIQ an AI-Powered Crop Assistant System for India: District and Month-Specific Insights for Optimized Agricultural Practices

## Overview

This project develops an AI-driven crop recommendation system tailored for Indian farmers. By analyzing key environmental factors like temperature, humidity, pH, and rainfall, the system provides district and month-specific crop recommendations. The primary goal is to empower farmers with data-driven insights, enhancing agricultural productivity and sustainability across India. The system ranks the best crops to plant based on given conditions, facilitating informed decision-making for optimized farming practices.

## Features

- **District and Month-Specific Recommendations**: Automatically retrieves rainfall data based on selected district and month
- **Data Preprocessing**: Combines multiple datasets, handles duplicates, normalizes/standardizes features, and encodes categorical data
- **Model Training**: Trains multiple machine learning models: Decision Tree, Gaussian Naive Bayes, Support Vector Machine (SVM), Logistic Regression, Random Forest, XGBoost, and K-Nearest Neighbors (KNN)
- **Automatic Model Selection**: Selects the best-performing model based on test accuracy
- **Model Evaluation**: Evaluates models using accuracy, precision, recall, and F1 score with detailed classification reports
- **Top-N Recommendations**: Provides top 5 crop recommendations with confidence scores
- **Visualization**: Visualizes model accuracies using Seaborn and Matplotlib
- **Web Interface**: User-friendly Streamlit application for real-time crop recommendations

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/mohanabandlamudi/cropiq
   ```
   ```sh
   cd cropiq
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model

1. Open `recommendation.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells in order:
   - **Cell 1**: Installs xgboost and imports required libraries
   - **Cell 2**: Loads and converts district rainfall data to Parquet format
   - **Cell 3**: Combines crop recommendation datasets and saves as Parquet
   - **Cell 4**: Trains all models, evaluates them, selects the best model, and saves:
     - `crop_recommendation_model.pkl` - The best trained model
     - `scaler.pkl` - StandardScaler for feature scaling
     - `label_encoder.pkl` - LabelEncoder for crop labels
     - `model_accuracies.csv` - Model performance metrics
   - **Cell 5**: Visualizes model accuracies

### Step 2: Run the Application

1. Ensure the following files are in the same directory as `app.py`:
   - `crop_recommendation_model.pkl`
   - `scaler.pkl`
   - `label_encoder.pkl`
   - `district_wise_rainfall_normal.parquet`

2. Run the Streamlit application:
   ```sh
   streamlit run app.py
   ```

3. Access the application in your web browser at `http://localhost:8501`

4. Use the application:
   - Select your district and month (rainfall will be auto-filled)
   - Input values for Nitrogen (N), Phosphorus (P), Potassium (K), pH, temperature, and humidity
   - Click "Predict" to get top 5 crop recommendations with confidence scores

## Project Structure

```
.
├── app.py                              # Streamlit web application
├── recommendation.ipynb                # Jupyter notebook for model training
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
│
├── Data Files:
├── district_wise_rainfall_normal.csv   # Raw district-wise rainfall data
├── district_wise_rainfall_normal.parquet  # Processed rainfall data (Parquet format)
├── Crop_recommendation.csv             # Crop recommendation dataset 1
├── Crop_recommendation_1.csv            # Crop recommendation dataset 2
├── Combined_Crop_recommendation.parquet # Combined crop recommendation dataset
│
└── Model Files (generated after training):
├── crop_recommendation_model.pkl       # Trained best model
├── scaler.pkl                          # StandardScaler for feature scaling
├── label_encoder.pkl                   # LabelEncoder for crop labels
└── model_accuracies.csv                # Model performance metrics
```

## Data Preprocessing

- **Data Combination**: Combines multiple crop recommendation datasets and removes duplicates
- **Feature Scaling**: Uses StandardScaler to normalize features (N, P, K, temperature, humidity, ph, rainfall) for consistent input scales
- **Label Encoding**: Encodes crop labels using LabelEncoder for model training
- **Data Format**: Converts CSV files to Parquet format for efficient storage and faster loading

## Model Training and Evaluation

- **Train-Test Split**: Splits data into 80% training and 20% testing sets with random state for reproducibility
- **Multiple Models**: Trains 7 different classification models with default parameters:
  - Decision Tree
  - Gaussian Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest
  - XGBoost
  - K-Nearest Neighbors (KNN)
- **Model Selection**: Automatically selects the best-performing model based on test accuracy
- **Model Evaluation**: Provides detailed classification reports with precision, recall, and F1-score for each crop class
- **Model Persistence**: Saves the best model, scaler, and label encoder using joblib for deployment

## Input Features

The system uses the following environmental and soil factors:
- **N (Nitrogen)**: Soil nitrogen content (0-100)
- **P (Phosphorus)**: Soil phosphorus content (0-100)
- **K (Potassium)**: Soil potassium content (0-100)
- **Temperature**: Average temperature in °C
- **Humidity**: Relative humidity percentage (0-100)
- **pH**: Soil pH level (0-14)
- **Rainfall**: Monthly rainfall in mm (auto-filled based on district and month)

## Output

The system provides:
- **Top 5 Crop Recommendations**: Ranked list of best crops to plant
- **Confidence Scores**: Probability scores for each recommendation (0-100%)
- **27 Crop Classes**: Supports recommendations for 27 different crop types including rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee, Soyabeans, beans, peas, groundnuts, and cowpeas

## Visualization

- **Model Comparison**: Bar chart visualization comparing accuracies of all trained models
- **Performance Metrics**: Detailed accuracy scores for each model displayed in the visualization

## Technical Details

- **Python Version**: 3.x
- **Key Libraries**: 
  - scikit-learn (model training and evaluation)
  - xgboost (gradient boosting classifier)
  - pandas (data manipulation)
  - streamlit (web interface)
  - matplotlib & seaborn (visualization)
  - joblib (model persistence)

## Deployment

### Step 1: Push Project to GitHub

1. **Initialize Git Repository** (if not already done):
   ```sh
   git init
   ```

2. **Add all files to Git**:
   ```sh
   git add .
   ```
   
   > **Note**: Make sure you have trained the model and generated all `.pkl` files before committing. The `.gitignore` file will exclude `venv/` and other unnecessary files.

3. **Create initial commit**:
   ```sh
   git commit -m "Initial commit: CROPIQ - AI-Powered Crop Recommendation System"
   ```

4. **Create a new repository on GitHub**:
   - Go to [GitHub](https://github.com) and sign in
   - Click the "+" icon in the top right corner
   - Select "New repository"
   - Repository name: `cropiq`
   - Description: "AI-Powered Crop Recommendation System for India"
   - Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

5. **Connect local repository to GitHub**:
   ```sh
   git remote add origin https://github.com/mohanabandlamudi/cropiq.git
   git branch -M main
   git push -u origin main
   ```

6. **Verify files are uploaded**:
   - Check your GitHub repository to ensure all required files are present:
     - `app.py`
     - `recommendation.ipynb`
     - `requirements.txt`
     - `README.md`
     - `crop_recommendation_model.pkl`
     - `scaler.pkl`
     - `label_encoder.pkl`
     - `district_wise_rainfall_normal.parquet`
     - All CSV and Parquet data files

### Step 2: Deploy on Streamlit Cloud

1. **Sign up/Login to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Authorize Streamlit Cloud to access your GitHub repositories

2. **Deploy your app**:
   - Click "New app" button
   - Select your GitHub repository: `mohanabandlamudi/cropiq`
   - Branch: `main` (or `master` if you used that)
   - Main file path: `app.py`
   - App URL: Choose a custom subdomain (e.g., `cropiq` or `crop-recommendation`)
   - Click "Deploy!"

3. **Wait for deployment**:
   - Streamlit Cloud will automatically:
     - Install dependencies from `requirements.txt`
     - Run your `app.py` file
     - Provide you with a public URL (e.g., `https://cropiq.streamlit.app`)

4. **Verify deployment**:
   - Once deployed, test the application:
     - Select a district and month
     - Input values for N, P, K, pH, temperature, and humidity
     - Click "Predict" to verify recommendations work correctly

5. **Update your app** (for future changes):
   - Make changes to your code locally
   - Commit and push to GitHub:
     ```sh
     git add .
     git commit -m "Your commit message"
     git push
     ```
   - Streamlit Cloud will automatically redeploy your app with the latest changes

### Important Notes for Deployment

- **File Size Limits**: 
  - GitHub has a 100MB file size limit per file
  - If your `.pkl` or `.parquet` files are too large, consider using Git LFS (Large File Storage) or hosting them elsewhere
  - Streamlit Cloud has a 1GB repository size limit

- **Required Files for Deployment**:
  - ✅ `app.py` - Main Streamlit application
  - ✅ `requirements.txt` - Python dependencies
  - ✅ `crop_recommendation_model.pkl` - Trained model
  - ✅ `scaler.pkl` - Feature scaler
  - ✅ `label_encoder.pkl` - Label encoder
  - ✅ `district_wise_rainfall_normal.parquet` - Rainfall data
  - ✅ All CSV/Parquet data files used by the app

- **Optional Files** (not required for deployment but useful):
  - `recommendation.ipynb` - For retraining models
  - `README.md` - Documentation
  - `.gitignore` - Git ignore rules

- **Troubleshooting**:
  - If deployment fails, check the logs in Streamlit Cloud dashboard
  - Ensure all dependencies in `requirements.txt` are correct
  - Verify all `.pkl` and `.parquet` files are committed to GitHub
  - Check that file paths in `app.py` match the repository structure

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

