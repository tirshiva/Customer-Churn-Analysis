# Telecom Customer Churn Prediction

This project implements a machine learning solution to predict customer churn in a telecommunications company. The application uses a Random Forest Classifier to predict the probability of a customer leaving the service.

## Features

- Data preprocessing and feature engineering
- Random Forest model for churn prediction
- Interactive Streamlit web application
- Real-time churn probability prediction
- Feature importance visualization
- Customer retention recommendations

## Project Structure

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shivanshu2407/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

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

## Model Performance

The Random Forest model achieves:
- Accuracy: ~79%
- Precision: 0.82 (No Churn) / 0.63 (Churn)
- Recall: 0.90 (No Churn) / 0.46 (Churn)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 