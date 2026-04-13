import random
import csv

random.seed(42)

countries = ['USA', 'UK', 'Germany', 'Brazil', 'Australia', 'France', 'Singapore', 'India', 'Japan', 'Canada']
industries = ['Manufacturing', 'Finance', 'Tech', 'Retail', 'Education', 'Healthcare']
work_modes = ['Remote', 'On-site', 'Hybrid']
genders = ['Male', 'Female', 'Non-binary']
burnout_risks = ['Low', 'Medium', 'High']
mental_health_options = ['Yes', 'No']

def get_productivity_level(score):
    if score <= 40:
        return 'Baja'
    elif score <= 70:
        return 'Media'
    else:
        return 'Alta'

data = []
for i in range(1, 1501):
    employee_id = f"EMP_{i:04d}"
    age = random.randint(22, 59)
    gender = random.choice(genders)
    country = random.choice(countries)
    industry = random.choice(industries)
    work_mode = random.choice(work_modes)
    work_hours = random.randint(30, 64)
    stress_level = random.randint(1, 10)
    sleep_hours = round(random.uniform(4.0, 10.0), 1)
    productivity_score = random.randint(40, 100)
    physical_activity_hours = round(random.uniform(0.0, 10.0), 1)
    mental_health_support_access = random.choice(mental_health_options)
    burnout_risk = random.choice(burnout_risks)
    productivity_level = get_productivity_level(productivity_score)
    
    data.append([
        employee_id, age, gender, country, industry, work_mode,
        work_hours, stress_level, sleep_hours, productivity_score,
        physical_activity_hours, mental_health_support_access, burnout_risk,
        productivity_level
    ])

with open(r'C:\Users\Usuario\MARIANA\Clases\Ingenieria de datos\Trabajos\Seguimiento 2\cleaned_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Employee_ID', 'Age', 'Gender', 'Country', 'Industry', 'Work_Mode',
        'Work_Hours_Per_Week', 'Stress_Level', 'Sleep_Hours', 'Productivity_Score',
        'Physical_Activity_Hours', 'Mental_Health_Support_Access', 'Burnout_Risk',
        'productivity_level'
    ])
    writer.writerows(data)

print(f"Created cleaned_dataset.csv with {len(data)} rows")