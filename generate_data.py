import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

def generate_data(num_records=1000):
    data = []
    
    # Define possible values
    regions = ['North', 'South', 'East', 'West', 'Central']
    genders = ['Male', 'Female']
    socio_economic_statuses = ['Low', 'Medium', 'High']
    sanitation_access_levels = ['Poor', 'Fair', 'Good']
    
    for _ in range(num_records):
        age_months = random.randint(6, 60) # 6 months to 5 years
        gender = random.choice(genders)
        region = random.choice(regions)
        socio_economic = random.choice(socio_economic_statuses)
        sanitation = random.choice(sanitation_access_levels)
        
        # Simulate health metrics with some logic
        # Baseline distributions
        height_cm = np.random.normal(85, 10) # rough average for toddlers
        weight_kg = np.random.normal(12, 3)  # rough average for toddlers
        
        # Adjust based on age (simple linear approximation + noise)
        height_cm = 50 + (age_months * 0.8) + np.random.normal(0, 2)
        weight_kg = 3 + (age_months * 0.25) + np.random.normal(0, 1)
        
        # Introduce malnutrition factors
        if socio_economic == 'Low' or sanitation == 'Poor':
            if random.random() < 0.4: # Higher chance of stunting/underweight
                height_cm *= 0.9
                weight_kg *= 0.85
        
        # Determine Status (Simplified Logic based on WHO standards concepts)
        # BMI-for-age is better, but using simple weight/height ratios for demo
        bmi = weight_kg / ((height_cm/100) ** 2)
        
        status = 'Healthy'
        if bmi < 13.5:
            status = 'Severely Wasted'
        elif bmi < 15:
            status = 'Wasted'
        elif weight_kg < (3 + age_months * 0.2) * 0.8: # Very rough underweight check
            status = 'Underweight'
        
        # Randomly assign 'Stunted' based on height-for-age (roughly)
        expected_height = 50 + (age_months * 0.8)
        if height_cm < expected_height * 0.9:
             # Can be co-morbid, but let's overwrite for classification simplicity in this demo
             # or keep 'Status' as the primary malnutrition type
             if status == 'Healthy':
                 status = 'Stunted'

        data.append({
            'Child_ID': fake.uuid4(),
            'Age_Months': age_months,
            'Gender': gender,
            'Height_cm': round(height_cm, 1),
            'Weight_kg': round(weight_kg, 1),
            'Region': region,
            'Socio_Economic_Status': socio_economic,
            'Sanitation_Access': sanitation,
            'Malnutrition_Status': status
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_data(2000)
    df.to_csv('child_health_data.csv', index=False)
    print("Synthetic data generated: child_health_data.csv")
    print(df.head())
