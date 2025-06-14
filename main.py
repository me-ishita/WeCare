from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import ast

# Flask app
app = Flask(__name__)

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load model
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Helper function to get all content based on disease
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication'].values[0]
    if isinstance(med, str):
        med = ast.literal_eval(med)

    die = diets[diets['Disease'] == dis]['Diet'].values[0]
    if isinstance(die, str):
        die = ast.literal_eval(die)

    wrkout = workout[workout['disease'] == dis]['workout']

    # Strip spaces from values in 'Disease' column (just in case)
    doctors_df['Disease'] = doctors_df['Disease'].str.strip()

    # Now filter based on the predicted disease
    try:
        doc_list = doctors_df[doctors_df['Disease'].str.lower() == dis.lower()][
            ['Doctor', 'Specialization', 'Email', 'Location']
        ].to_dict(orient='records')
    except KeyError as e:
        print("Column not found in DataFrame:", e)
        doc_list = []
    return desc, pre, med, die, wrkout, doc_list

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
data = [
    [15, "Fungal infection", "Dr. Ananya Mehra", "ananya.mehra@wecareclinic.in", "Mumbai", "Dermatologist"],
    [4, "Allergy", "Dr. Rohit Kapoor", "rohit.kapoor@allergycenter.in", "Delhi", "Immunologist"],
    [16, "GERD", "Dr. Sneha Rathi", "sneha.rathi@gastrocare.in", "Pune", "Gastroenterologist"],
    [9, "Chronic cholestasis", "Dr. Arjun Rao", "arjun.rao@liverinstitute.in", "Hyderabad", "Hepatologist"],
    [14, "Drug Reaction", "Dr. Kavita Sharma", "kavita.sharma@dermacare.in", "Bangalore", "Dermatologist"],
    [33, "Peptic ulcer diseae", "Dr. Nikhil Jain", "nikhil.jain@digestivecare.in", "Chennai", "Gastroenterologist"],
    [1, "AIDS", "Dr. Reema Joshi", "reema.joshi@hivaidscare.in", "Kolkata", "Infectious Disease Specialist"],
    [12, "Diabetes", "Dr. Varun Sinha", "varun.sinha@endoclinic.in", "Ahmedabad", "Endocrinologist"],
    [17, "Gastroenteritis", "Dr. Neha Arora", "neha.arora@gutclinic.in", "Surat", "Gastroenterologist"],
    [6, "Bronchial Asthma", "Dr. Karan Malhotra", "karan.malhotra@pulmocare.in", "Jaipur", "Pulmonologist"],
    [23, "Hypertension", "Dr. Tanya Verma", "tanya.verma@cardioclinic.in", "Noida", "Cardiologist"],
    [30, "Migraine", "Dr. Alok Nanda", "alok.nanda@neurocenter.in", "Bhopal", "Neurologist"],
    [7, "Cervical spondylosis", "Dr. Snehal Patil", "snehal.patil@spineclinic.in", "Nagpur", "Orthopedic Surgeon"],
    [32, "Paralysis (brain hemorrhage)", "Dr. Priya Dey", "priya.dey@neurocare.in", "Patna", "Neurologist"],
    [28, "Jaundice", "Dr. Harshita Singh", "harshita.singh@liverclinic.in", "Lucknow", "Hepatologist"],
    [29, "Malaria", "Dr. Sameer Das", "sameer.das@tropicalmed.in", "Guwahati", "General Physician"],
    [8, "Chicken pox", "Dr. Anil Khatri", "anil.khatri@infectionclinic.in", "Chandigarh", "Infectious Disease Specialist"],
    [11, "Dengue", "Dr. Meenal Desai", "meenal.desai@feverclinic.in", "Ranchi", "General Physician"],
    [37, "Typhoid", "Dr. Dev Sharma", "dev.sharma@carehospital.in", "Indore", "General Physician"],
    [40, "hepatitis A", "Dr. Shreya Iyer", "shreya.iyer@hepatocare.in", "Bhubaneswar", "Hepatologist"],
    [19, "Hepatitis B", "Dr. Aditya Menon", "aditya.menon@liverhealth.in", "Thiruvananthapuram", "Hepatologist"],
    [20, "Hepatitis C", "Dr. Ishita Kaul", "ishita.kaul@livercare.in", "Raipur", "Hepatologist"],
    [21, "Hepatitis D", "Dr. Mohit Shekhar", "mohit.shekhar@hepatology.in", "Amritsar", "Hepatologist"],
    [22, "Hepatitis E", "Dr. Parul Jain", "parul.jain@liverdoctor.in", "Visakhapatnam", "Hepatologist"],
    [3, "Alcoholic hepatitis", "Dr. Vikram Joshi", "vikram.joshi@liverrehab.in", "Varanasi", "Hepatologist"],
    [36, "Tuberculosis", "Dr. Namita Bansal", "namita.bansal@tbclinic.in", "Dehradun", "Pulmonologist"],
    [10, "Common Cold", "Dr. Akash Tripathi", "akash.tripathi@generalclinic.in", "Agra", "General Physician"],
    [34, "Pneumonia", "Dr. Sheetal Roy", "sheetal.roy@lungcare.in", "Srinagar", "Pulmonologist"],
    [13, "Dimorphic hemmorhoids(piles)", "Dr. Ajay Nair", "ajay.nair@proctocare.in", "Kozhikode", "Proctologist"],
    [18, "Heart attack", "Dr. Radhika Nambiar", "radhika.nambiar@heartcare.in", "Mangalore", "Cardiologist"],
    [39, "Varicose veins", "Dr. Yashwant Kulkarni", "yash.kulkarni@vascularclinic.in", "Nashik", "Vascular Surgeon"],
    [26, "Hypothyroidism", "Dr. Charu Saxena", "charu.saxena@endoclinic.in", "Faridabad", "Endocrinologist"],
    [24, "Hyperthyroidism", "Dr. Nidhi Prakash", "nidhi.prakash@endoclinic.in", "Gwalior", "Endocrinologist"],
    [25, "Hypoglycemia", "Dr. Ravi Shankar", "ravi.shankar@glucosecare.in", "Meerut", "Endocrinologist"],
    [31, "Osteoarthristis", "Dr. Lakshmi Pillai", "lakshmi.pillai@jointcare.in", "Hubli", "Rheumatologist"],
    [5, "Arthritis", "Dr. Manish Gera", "manish.gera@arthritiscenter.in", "Jodhpur", "Rheumatologist"],
    [0, "(vertigo) Paroymsal  Positional Vertigo", "Dr. Aishwarya Rao", "aishwarya.rao@entclinic.in", "Jamshedpur", "ENT Specialist"],
    [2, "Acne", "Dr. Ritu Bhargava", "ritu.bhargava@skincare.in", "Udaipur", "Dermatologist"],
    [38, "Urinary tract infection", "Dr. Deepak Suri", "deepak.suri@urologycenter.in", "Rohtak", "Urologist"],
    [35, "Psoriasis", "Dr. Shruti Malik", "shruti.malik@psoriasisclinic.in", "Panipat", "Dermatologist"],
    [27, "Impetigo", "Dr. Harpal Grewal", "harpal.grewal@dermacenter.in", "Ajmer", "Dermatologist"]
]
columns = ["Disease ID", "Disease", "Doctor", "Email", "Location", "Specialization"]
doctors_df = pd.DataFrame(data, columns=columns)

# Prediction
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')

        if symptoms == "Symptoms" or not symptoms.strip():
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', symptoms="")
        else:
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout, doc_list = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('index.html',
                                   predicted_disease=predicted_disease,
                                   dis_des=dis_des,
                                   my_precautions=my_precautions,
                                   medications=medications,
                                   my_diet=rec_diet,
                                   workout=workout,
                                   doctor_list=doc_list,
                                   symptoms=symptoms)
    return render_template('index.html')

@app.route("/get_symptoms")
def get_symptoms():
    return jsonify(list(symptoms_dict.keys()))

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

if __name__ == '__main__':
    app.run(debug=True)
