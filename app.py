import streamlit as st
import joblib
import pandas as pd
from PIL import Image

# Load the model and image
@st.cache_resource
def load_model_and_image():
    model_info = joblib.load('best.sav')  # Corrected filename to match assumed training output
    header_image = Image.open('mountain_header.jpg')
    return model_info, header_image

# Preprocessing function
def preprocess_input(termination_reason, year, members, hired_staff, citizenship, expedition_role, trekking_agency, peak_succes_rate, expedition_duration, height_metres, peak_name, age_group, oxygen_used, season, sex, solo):
    # Define the exact feature order as trained
    feature_names = [
        'termination_reason', 'year', 'members', 'hired_staff', 'citizenship', 'expedition_role', 
        'trekking_agency', 'peak_succes_rate', 'expedition_duration', 'height_metres', 'peak_name', 
        'age_group', 'oxygen_used', 'season', 'sex', 'solo'
    ]
    
    input_data = pd.DataFrame({
        "termination_reason": [termination_reason],
        "year": [year],
        "members": [members],
        "hired_staff": [hired_staff],
        "citizenship": [citizenship],
        "expedition_role": [expedition_role],
        "trekking_agency": [trekking_agency],
        "peak_succes_rate": [peak_succes_rate],
        "expedition_duration": [expedition_duration],
        "height_metres": [height_metres],
        "peak_name": [peak_name],
        "age_group": [age_group],
        "oxygen_used": [oxygen_used],
        "season": [season],
        "sex": [sex],
        "solo": [solo]
    })
    
    # Mapping categorical variables (adjust based on your training data encoding)
    termination_reason_map = {"Success": 0, "Weather": 1, "Injury": 2, "Other": 3}  # Example; adjust as per training
    citizenship_map = {"Nepal": 0, "USA": 1, "UK": 2, "India": 3, "Other": 4}  # Example; adjust
    expedition_role_map = {"Leader": 0, "Member": 1, "Support": 2}  # Example; adjust
    trekking_agency_map = {"Agency A": 0, "Agency B": 1, "Agency C": 2, "Agency D": 3}  # Example; adjust
    peak_name_map = {
        "Everest": 0, "K2": 1, "Kangchenjunga": 2, "Lhotse": 3, "Makalu": 4,
        "Cho Oyu": 5, "Dhaulagiri": 6, "Manaslu": 7, "Nanga Parbat": 8, "Annapurna": 9,
        "Gasherbrum I": 10, "Broad Peak": 11, "Gasherbrum II": 12, "Shishapangma": 13
    }
    age_group_map = {"Under 20": 0, "20-30": 1, "30-40": 2, "40-50": 3, "50-60": 4, "Over 60": 5}
    oxygen_map = {"Yes": 1, "No": 0}
    season_map = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
    sex_map = {"Male": 0, "Female": 1}
    solo_map = {"Yes": 1, "No": 0}
    
    # Apply mappings
    input_data["termination_reason"] = input_data["termination_reason"].map(termination_reason_map)
    input_data["citizenship"] = input_data["citizenship"].map(citizenship_map)
    input_data["expedition_role"] = input_data["expedition_role"].map(expedition_role_map)
    input_data["trekking_agency"] = input_data["trekking_agency"].map(trekking_agency_map)
    input_data["peak_name"] = input_data["peak_name"].map(peak_name_map)
    input_data["age_group"] = input_data["age_group"].map(age_group_map)
    input_data["oxygen_used"] = input_data["oxygen_used"].map(oxygen_map)
    input_data["season"] = input_data["season"].map(season_map)
    input_data["sex"] = input_data["sex"].map(sex_map)
    input_data["solo"] = input_data["solo"].map(solo_map)
    
    # Ensure no NaN values
    if input_data.isnull().values.any():
        raise ValueError(f"Input data contains NaN values after preprocessing: {input_data.isnull().sum()}")
    
    # Reorder columns to match training feature order
    input_data = input_data[feature_names]
    
    return input_data

# Main app
def main():
    st.set_page_config(page_title="Himalayan Expedition Predictor", page_icon="⛰️", layout="wide")
    
    # Load assets
    model_info, header_image = load_model_and_image()
    model = model_info['model']
    
    # # Debugging: Display expected features
    # st.sidebar.write("Model Feature Names:", model_info.get('feature_names', 'Not stored'))
    
    # Title
    st.title("Himalayan Expedition Success Predictor")
    
    # Layout with two columns
    col1, col2 = st.columns([1, 2])
    
    # Left column: Image
    with col1:
        st.image(header_image, use_container_width=True)
    
    # Right column: Input form
    with col2:
        st.subheader("Enter Expedition Details")
        with st.form(key='expedition_form'):
            termination_reason = st.selectbox("Termination Reason", options=["Success", "Weather", "Injury", "Other"])
            year = st.number_input("Year", min_value=1900, max_value=2025, value=2023)
            members = st.number_input("Team Members", min_value=1, max_value=100, value=5)
            hired_staff = st.number_input("Hired Staff", min_value=0, max_value=50, value=2)
            citizenship = st.selectbox("Citizenship", options=["Nepal", "USA", "UK", "India", "Other"])
            expedition_role = st.selectbox("Expedition Role", options=["Leader", "Member", "Support"])
            trekking_agency = st.selectbox("Trekking Agency", options=["Agency A", "Agency B", "Agency C", "Agency D"])
            peak_succes_rate = st.slider("Peak Success Rate (%)", min_value=0.0, max_value=100.0, value=50.0) / 100.0  # Convert to 0-1 scale
            expedition_duration = st.number_input("Expedition Duration (days)", min_value=1, max_value=365, value=30)
            height_metres = st.number_input("Mountain Height (meters)", min_value=5000, max_value=9000, value=6000)
            peak_name = st.selectbox("Peak Name", options=[
                "Everest", "K2", "Kangchenjunga", "Lhotse", "Makalu",
                "Cho Oyu", "Dhaulagiri", "Manaslu", "Nanga Parbat", "Annapurna",
                "Gasherbrum I", "Broad Peak", "Gasherbrum II", "Shishapangma"
            ])
            age_group = st.selectbox("Age Group", options=["Under 20", "20-30", "30-40", "40-50", "50-60", "Over 60"])
            oxygen_used = st.selectbox("Oxygen Used", options=["Yes", "No"])
            season = st.selectbox("Season", options=["Spring", "Summer", "Autumn", "Winter"])
            sex = st.selectbox("Sex", options=["Male", "Female"])
            solo = st.selectbox("Solo Expedition", options=["Yes", "No"])
            
            submit_button = st.form_submit_button(label="Predict")
        
        # Prediction logic
        if submit_button:
            try:
                processed_input = preprocess_input(
                    termination_reason, year, members, hired_staff, citizenship, expedition_role,
                    trekking_agency, peak_succes_rate, expedition_duration, height_metres, peak_name,
                    age_group, oxygen_used, season, sex, solo
                )
                
                # Debugging: Display processed input
                st.write("Processed Input:", processed_input)
                
                # Make prediction
                prediction = model.predict(processed_input)
                probability = model.predict_proba(processed_input)[0]
                
                # Display results
                st.subheader("Prediction Results")
                outcome = "Success" if prediction[0] == 1 else "Failure"
                st.write(f"Predicted Outcome: **{outcome}**")
                st.write(f"Probability of Success: **{probability[1]:.2%}**")
                st.write(f"Probability of Failure: **{probability[0]:.2%}**")
            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    # Footer
    st.write("---")
    st.write("Built with ❤️ by Pratikshya Karki using Streamlit and CatBoost")

if __name__ == "__main__":
    main()