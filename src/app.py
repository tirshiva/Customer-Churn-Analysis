import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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

class DataProcessor:
    def __init__(self, data=None):
        self.data = data

    def preprocess_data(self):
        """Preprocess the data."""
        try:
            # Create a copy of the data
            processed_data = self.data.copy()

            # Convert TotalCharges to numeric
            processed_data['TotalCharges'] = pd.to_numeric(processed_data['TotalCharges'], errors='coerce')

            # Fill missing values
            numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
            for col in numeric_columns:
                processed_data[col].fillna(processed_data[col].median(), inplace=True)

            # One-hot encode categorical variables
            categorical_columns = [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod'
            ]

            # One-hot encode each categorical column
            for col in categorical_columns:
                if col in processed_data.columns:
                    dummies = pd.get_dummies(processed_data[col], prefix=col)
                    processed_data = pd.concat([processed_data, dummies], axis=1)
                    processed_data.drop(col, axis=1, inplace=True)

            return processed_data
        except Exception as e:
            st.error(f"Error in preprocessing data: {str(e)}")
            return None

class ChurnPredictor:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
            self.feature_names = self.model.feature_names_in_
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            self.model = None
            self.feature_names = None

    def process_input_data(self, input_data):
        """Process input data using DataProcessor."""
        try:
            # Create DataFrame with single row
            df = pd.DataFrame([input_data])
            
            # Process the data
            processor = DataProcessor(df)
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

    def get_feature_importance(self, input_data):
        """Get feature importance for the prediction."""
        try:
            processed_data = self.process_input_data(input_data)
            if processed_data is None:
                return None

            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create DataFrame with feature names and importances
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            return feature_importance
        except Exception as e:
            st.error(f"Error getting feature importance: {str(e)}")
            return None

def load_model():
    try:
        model_path = Path(__file__).parent.parent / "models" / "random_forest_model.joblib"
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

def plot_feature_importance(feature_importance):
    """Plot feature importance."""
    try:
        plt.figure(figsize=(10, 6))
        # Convert feature names to strings to avoid any encoding issues
        feature_importance['Feature'] = feature_importance['Feature'].astype(str)
        # Get top 10 features
        top_features = feature_importance.head(10)
        # Create the plot
        sns.barplot(data=top_features, x='Importance', y='Feature')
        plt.title('Top 10 Factors Affecting Churn Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        st.error(f"Error creating feature importance plot: {str(e)}")
        return None

def plot_customer_profile(input_data):
    """Plot customer profile visualization."""
    try:
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Contract and Payment Method
        contract_payment = pd.DataFrame({
            'Category': ['Contract', 'Payment Method'],
            'Value': [str(input_data['Contract']), str(input_data['PaymentMethod'])]
        })
        sns.barplot(data=contract_payment, x='Category', y='Value', ax=axes[0,0])
        axes[0,0].set_title('Contract & Payment Method')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Service Features
        services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'TechSupport']
        service_values = [str(input_data[service]) for service in services]
        sns.barplot(x=services, y=service_values, ax=axes[0,1])
        axes[0,1].set_title('Service Features')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Charges Distribution
        charges = ['MonthlyCharges', 'TotalCharges']
        charge_values = [float(input_data['MonthlyCharges']), float(input_data['TotalCharges'])]
        sns.barplot(x=charges, y=charge_values, ax=axes[1,0])
        axes[1,0].set_title('Charges Distribution')
        
        # 4. Customer Demographics
        demographics = ['SeniorCitizen', 'Partner', 'Dependents']
        demo_values = [
            int(input_data['SeniorCitizen']),
            1 if input_data['Partner'] == 'Yes' else 0,
            1 if input_data['Dependents'] == 'Yes' else 0
        ]
        sns.barplot(x=demographics, y=demo_values, ax=axes[1,1])
        axes[1,1].set_title('Customer Demographics')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating customer profile plot: {str(e)}")
        return None

def main():
    # Sidebar
    st.sidebar.title("📱 Telecom Churn Prediction")
    st.sidebar.markdown("---")
    
    # Add feature descriptions
    st.sidebar.subheader("Feature Descriptions")
    with st.sidebar.expander("Customer Information", expanded=True):
        st.markdown("""
        **Tenure**: Number of months the customer has stayed with the company
        - Longer tenure → Lower churn risk
        - Shorter tenure → Higher churn risk

        **Monthly Charges**: Amount charged to the customer monthly
        - Higher charges → Higher churn risk
        - Lower charges → Lower churn risk

        **Total Charges**: Total amount charged to the customer
        - Higher total charges → Lower churn risk (indicates loyalty)
        - Lower total charges → Higher churn risk

        **Contract Type**: Type of contract the customer has
        - Month-to-month → Higher churn risk
        - One year → Medium churn risk
        - Two year → Lower churn risk

        **Payment Method**: How the customer pays their bills
        - Electronic check → Higher churn risk
        - Automatic payments → Lower churn risk
        """)

    with st.sidebar.expander("Service Features", expanded=True):
        st.markdown("""
        **Internet Service**: Type of internet service
        - Fiber optic → Higher churn risk (more competition)
        - DSL → Lower churn risk
        - No internet → Lower churn risk

        **Tech Support**: Availability of technical support
        - No tech support → Higher churn risk
        - Tech support → Lower churn risk

        **Online Security**: Online security service
        - No security → Higher churn risk
        - Security → Lower churn risk

        **Phone Service**: Whether the customer has phone service
        - No phone → Higher churn risk
        - Phone service → Lower churn risk
        """)

    with st.sidebar.expander("Customer Demographics", expanded=True):
        st.markdown("""
        **Senior Citizen**: Whether the customer is a senior citizen
        - Senior citizen → Higher churn risk
        - Non-senior → Lower churn risk

        **Partner**: Whether the customer has a partner
        - No partner → Higher churn risk
        - Has partner → Lower churn risk

        **Dependents**: Whether the customer has dependents
        - No dependents → Higher churn risk
        - Has dependents → Lower churn risk
        """)

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

    # Add a predict button
    if st.button("Predict Churn Probability"):
        try:
            # Make prediction
            prediction = predictor.predict(input_data)
            
            if prediction is not None:
                # Get the churn probability
                churn_prob = prediction[0][1] * 100
                
                # Display the gauge chart
                st.subheader("Churn Probability")
                fig = create_gauge_chart(churn_prob, "Customer Churn Risk")
                st.pyplot(fig)
                plt.close()
                
                # Display risk level and recommendations
                st.subheader("Risk Assessment")
                if churn_prob < 30:
                    st.success("Low Risk of Churn")
                    st.markdown("""
                    **Recommendations:**
                    - Continue providing excellent service
                    - Consider upselling premium services
                    - Collect feedback for further improvements
                    """)
                elif churn_prob < 70:
                    st.warning("Medium Risk of Churn")
                    st.markdown("""
                    **Recommendations:**
                    - Proactively reach out to understand concerns
                    - Offer promotional discounts
                    - Review service usage patterns
                    - Consider service upgrades
                    """)
                else:
                    st.error("High Risk of Churn")
                    st.markdown("""
                    **Recommendations:**
                    - Immediate customer outreach
                    - Offer retention packages
                    - Schedule account review
                    - Consider service plan adjustments
                    - Address any service issues
                    """)

                # Display feature importance
                st.subheader("Key Factors Affecting Prediction")
                feature_importance = predictor.get_feature_importance(input_data)
                if feature_importance is not None:
                    fig = plot_feature_importance(feature_importance)
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.error("Could not create feature importance plot")

                # Display customer profile
                st.subheader("Customer Profile Analysis")
                fig = plot_customer_profile(input_data)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.error("Could not create customer profile plot")

                # Display detailed analysis
                st.subheader("Detailed Analysis")
                try:
                    contract_impact = feature_importance[feature_importance['Feature'].str.contains('Contract')]['Importance'].sum() * 100
                    service_impact = feature_importance[feature_importance['Feature'].str.contains('InternetService|TechSupport')]['Importance'].sum() * 100
                    financial_impact = feature_importance[feature_importance['Feature'].str.contains('Charges')]['Importance'].sum() * 100

                    st.markdown(f"""
                    ### Key Insights:
                    1. **Contract Impact**
                       - Current contract type: {input_data['Contract']}
                       - Contract influence on churn: {contract_impact:.2f}%
                    
                    2. **Service Quality**
                       - Internet service type: {input_data['InternetService']}
                       - Tech support availability: {input_data['TechSupport']}
                       - Service quality impact: {service_impact:.2f}%
                    
                    3. **Financial Factors**
                       - Monthly charges: ${input_data['MonthlyCharges']:.2f}
                       - Total charges: ${input_data['TotalCharges']:.2f}
                       - Financial impact: {financial_impact:.2f}%
                    """)
                except Exception as e:
                    st.error(f"Error creating detailed analysis: {str(e)}")

        except Exception as e:
            st.error(f"Error during prediction process: {str(e)}")

if __name__ == "__main__":
    main() 