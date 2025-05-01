import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_processor import DataProcessor

# Set page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

class ChurnPredictor:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
            # Store the feature names from the model
            self.feature_names = self.model.feature_names_in_
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            self.model = None
            self.feature_names = None

    def process_input_data(self, input_data):
        """Process input data using the same logic as DataProcessor."""
        try:
            # Create DataFrame with single row
            df = pd.DataFrame([input_data])
            
            # Create DataProcessor instance (without loading data)
            processor = DataProcessor("")
            processor.data = df
            
            # Process the data
            processed_data = processor.preprocess_data()
            
            if processed_data is None:
                st.error("Error processing input data")
                return None
            
            # Ensure all features from the model are present
            for feature in self.feature_names:
                if feature not in processed_data.columns:
                    processed_data[feature] = 0
            
            # Select only the features used by the model
            processed_data = processed_data[self.feature_names]
                
            return processed_data
        except Exception as e:
            st.error(f"Error processing input data: {str(e)}")
            return None

    def predict(self, input_data):
        """Make prediction using processed input data."""
        if self.model is None:
            return None
            
        try:
            # Process input data
            processed_data = self.process_input_data(input_data)
            if processed_data is None:
                return None
                
            # Make prediction
            return self.model.predict_proba(processed_data)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None

def load_model():
    try:
        model_path = Path("models/random_forest_model.joblib")
        predictor = ChurnPredictor(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_gauge_chart(value, title):
    """Create a gauge chart using matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create gauge
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 1])
    
    # Draw gauge background
    ax.barh(0, 100, height=0.3, color='lightgray')
    
    # Draw colored sections
    ax.barh(0, 30, height=0.3, color='lightgreen')
    ax.barh(0, 40, height=0.3, color='yellow', left=30)
    ax.barh(0, 30, height=0.3, color='red', left=70)
    
    # Draw needle
    ax.plot([value, value], [0, 0.4], 'k-', linewidth=2)
    
    # Add value text
    ax.text(value, 0.5, f'{value:.1f}%', ha='center', va='bottom')
    
    # Customize appearance
    ax.set_title(title)
    ax.axis('off')
    
    return fig

def main():
    # Sidebar
    st.sidebar.title("📱 Telecom Churn Prediction")
    st.sidebar.markdown("---")
    
    # Load model
    predictor = load_model()
    if predictor is None:
        st.error("Please train the model first!")
        return

    # Main content
    st.title("Telecom Customer Churn Prediction Dashboard")
    st.markdown("---")

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Information")
        # Add input fields for customer information
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)
        
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["Yes", "No"]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col2:
        st.subheader("Service Information")
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    # Create a dictionary of input values
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }

    # Prediction button
    if st.button("Predict Churn Probability"):
        # Make prediction
        prediction = predictor.predict(input_data)
        if prediction is not None:
            churn_probability = prediction[0][1] * 100  # Probability of churn as percentage

            # Display prediction results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Create and display gauge chart
            fig = create_gauge_chart(churn_probability, "Churn Probability")
            st.pyplot(fig)
            plt.close()

            # Display recommendation
            st.subheader("Recommendation")
            if churn_probability < 30:
                st.success("Low churn risk! Customer is likely to stay.")
            elif churn_probability < 70:
                st.warning("Medium churn risk! Consider retention strategies.")
            else:
                st.error("High churn risk! Immediate action required.")

            # Display key factors
            st.subheader("Key Factors")
            feature_importance = pd.DataFrame({
                'Feature': predictor.model.feature_names_in_,
                'Importance': predictor.model.feature_importances_
            }).sort_values('Importance', ascending=False)

            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(5), x='Importance', y='Feature')
            plt.title('Top 5 Factors Affecting Churn')
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()

if __name__ == "__main__":
    main() 