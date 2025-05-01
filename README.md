# Telecom Customer Churn Prediction

## 📊 Project Overview
This project implements a machine learning solution to predict customer churn in a telecommunications company. The application uses a Random Forest Classifier to predict the probability of a customer leaving the service, helping telecom companies proactively retain customers and reduce churn rates.

## 🎯 Business Impact
- **Cost Reduction**: Early identification of potential churners helps reduce customer acquisition costs
- **Revenue Protection**: Proactive retention strategies help maintain revenue streams
- **Customer Satisfaction**: Targeted interventions improve customer experience
- **Resource Optimization**: Better allocation of retention resources based on churn risk

## 🛠️ Technical Implementation

### Data Processing Pipeline
1. **Data Cleaning**
   - Handling missing values
   - Converting data types
   - Standardizing formats

2. **Feature Engineering**
   - One-hot encoding of categorical variables
   - Numerical feature scaling
   - Handling of special cases (e.g., "No internet service")

3. **Model Training**
   - Random Forest Classifier implementation
   - Hyperparameter tuning
   - Cross-validation
   - Feature importance analysis

### Model Performance
- **Accuracy**: 79%
- **Precision**: 
  - No Churn: 0.82
  - Churn: 0.63
- **Recall**:
  - No Churn: 0.90
  - Churn: 0.46

### Key Features Identified
1. Contract Type
2. Monthly Charges
3. Internet Service Type
4. Payment Method
5. Tech Support Availability

## 💻 Interactive Web Application

### Features
1. **Real-time Prediction**
   - Instant churn probability calculation
   - Visual risk assessment
   - Detailed recommendations

2. **User Interface**
   - Intuitive input forms
   - Interactive visualizations
   - Responsive design

3. **Risk Assessment**
   - Color-coded risk levels
   - Customized recommendations
   - Actionable insights

### Input Parameters
- Customer Demographics
  - Gender
  - Senior Citizen Status
  - Partner Status
  - Dependents

- Service Details
  - Tenure
  - Contract Type
  - Monthly Charges
  - Total Charges
  - Payment Method

- Service Features
  - Phone Service
  - Internet Service
  - Online Security
  - Online Backup
  - Device Protection
  - Tech Support
  - Streaming Services

## 📁 Project Structure
```
├── data/
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── models/            # Trained model files
├── src/
│   ├── data/         # Data processing modules
│   ├── models/       # Model training modules
│   └── app.py        # Streamlit application
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Shivanshu2407/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Process the data:
```bash
python src/data/data_processor.py
```

2. Train the model:
```bash
python src/models/model_trainer.py
```

3. Run the Streamlit app:
```bash
streamlit run src/app.py
```

## 🔍 Model Insights

### Key Findings
1. **Contract Impact**
   - Month-to-month contracts have higher churn rates
   - Longer contracts show better retention

2. **Service Quality**
   - Tech support availability reduces churn
   - Internet service type significantly affects retention

3. **Financial Factors**
   - Higher monthly charges correlate with increased churn
   - Payment method affects customer satisfaction

### Recommendations
1. **High-Risk Customers**
   - Immediate outreach
   - Retention packages
   - Service plan adjustments

2. **Medium-Risk Customers**
   - Proactive communication
   - Promotional offers
   - Service upgrades

3. **Low-Risk Customers**
   - Regular engagement
   - Premium service upselling
   - Feedback collection

## 🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## 🔄 Future Improvements
1. **Model Enhancement**
   - Deep learning integration
   - Ensemble methods
   - Real-time model updates

2. **Feature Engineering**
   - Additional customer metrics
   - Behavioral patterns
   - Usage trends

3. **Application Features**
   - Batch prediction
   - Historical analysis
   - Custom reporting

## 📊 Live Demo
Access the live application at: [Telecom Churn Prediction App](https://shivanshu2407-customer-churn-analysis-srcapp-b4z2jc.streamlit.app/)

## 👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact
For any queries or suggestions, please reach out through GitHub issues or pull requests. 