# Child Malnutrition Predictive Dashboard

## Overview
This project is a predictive analytics dashboard for tracking child malnutrition trends. It uses simulated community health data to predict malnutrition risk levels (Healthy, Stunted, Wasted, etc.) and provides a dashboard for visualization and reporting.

## Project Structure
- `app.py`: The main Streamlit dashboard application.
- `data_generator.py`: Script to generate synthetic training data (`child_health_data.csv`).
- `model_trainer.py`: Script to train the Random Forest model (`malnutrition_model.pkl`).
- `requirements.txt`: List of Python libraries required.

## Installation & Setup

### 1. Install Python
Ensure you have Python installed (version 3.8+ recommended).

### 2. Install Libraries
Open your terminal or command prompt in this directory and run:

```bash
pip install -r requirements.txt
```

### 3. Generate Data & Train Model
Before running the dashboard, you need to create the data and train the model. Run these commands in order:

```bash
# Generate synthetic data
python data_generator.py

# Train the predictive model
python model_trainer.py
```

### 4. Run the Dashboard
Start the application using Streamlit:

```bash
python -m streamlit run app.py
```

> [!NOTE]
> If `streamlit run app.py` fails with a "command not found" error, use `python -m streamlit run app.py` instead.

The application will open in your default web browser.

## Features
- **Dashboard Overview**: View key metrics, malnutrition rates, and regional breakdowns.
- **Predictive Analytics**: Input a child's health data to predict their malnutrition risk in real-time.
- **Reports & Alerts**: Identify high-risk regions and view critical cases needing immediate attention.
