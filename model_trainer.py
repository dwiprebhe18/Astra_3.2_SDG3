import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def train_model():
    # Load data
    try:
        df = pd.read_csv('child_health_data.csv')
    except FileNotFoundError:
        print("Data file not found. Please run data_generator.py first.")
        return

    # Preprocessing
    le_gender = LabelEncoder()
    le_region = LabelEncoder()
    le_socio = LabelEncoder()
    le_sanitation = LabelEncoder()
    
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Region'] = le_region.fit_transform(df['Region'])
    df['Socio_Economic_Status'] = le_socio.fit_transform(df['Socio_Economic_Status'])
    df['Sanitation_Access'] = le_sanitation.fit_transform(df['Sanitation_Access'])

    # Features and Target
    X = df[['Age_Months', 'Gender', 'Height_cm', 'Weight_kg', 'Region', 'Socio_Economic_Status', 'Sanitation_Access']]
    y = df['Malnutrition_Status']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save artifacts
    joblib.dump(clf, 'malnutrition_model.pkl')
    
    # Save encoders to handle user input in the app
    encoders = {
        'gender': le_gender,
        'region': le_region,
        'socio': le_socio,
        'sanitation': le_sanitation
    }
    joblib.dump(encoders, 'encoders.pkl')
    print("Model and encoders saved.")

if __name__ == "__main__":
    train_model()
