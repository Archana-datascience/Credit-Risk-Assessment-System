# Credit Risk Modeling with Machine Learning & Streamlit Frontend

A comprehensive machine learning project for credit risk assessment with real-time predictions and interactive web interface built with Streamlit.

## 🎯 Project Overview

This project implements an end-to-end machine learning solution for credit risk modeling, featuring:

- **Advanced ML Pipeline**: Complete data preprocessing, feature engineering, and model training
- **Hyperparameter Tuning**: Automated optimization using GridSearchCV and RandomizedSearchCV
- **Multiple Algorithms**: Comparison of Logistic Regression, Random Forest, and Gradient Boosting
- **Interactive Web Interface**: Professional Streamlit dashboard for real-time predictions
- **Batch Processing**: Handle multiple credit applications simultaneously
- **Model Performance Monitoring**: Comprehensive metrics and visualizations

## 🔧 Technologies Used

- **Machine Learning**: scikit-learn, XGBoost, LightGBM, imbalanced-learn
- **Data Analysis**: pandas, numpy
- **Visualization**: plotly, seaborn, matplotlib
- **Web Framework**: Streamlit
- **Model Persistence**: joblib
- **Development**: Python 3.8+

## 📊 Dataset Features

The synthetic credit risk dataset includes:

### Financial Features:
- Age (18-80 years)
- Annual Income
- Credit Amount Requested
- Loan Duration (months)
- Debt-to-Income Ratio
- Credit History Length

### Categorical Features:
- Loan Purpose (car, furniture, education, business, vacation, medical)
- Employment Status (employed, self-employed, unemployed, retired)
- Housing Status (own, rent, free)
- Savings Account Level (little, moderate, rich, quite_rich)
- Checking Account Level (little, moderate, rich)

### Engineered Features:
- Credit-to-Income Ratio
- Monthly Payment Amount
- Age Groups (Young, Adult, Middle, Senior)
- Income Groups (Low, Below_Avg, Average, Above_Avg, High)

## 🚀 Project Structure

```
credit-risk-modeling/
│
├── model_training.py      # Complete ML pipeline and model training
├── streamlit_app.py       # Interactive web application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── Generated Files:
├── credit_risk_model.pkl  # Trained model and preprocessing objects
└── credit_risk_dataset.csv # Synthetic credit risk dataset
```

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd credit-risk-modeling
```

### 2. Create Virtual Environment
```bash
python -m venv credit_risk_env
source credit_risk_env/bin/activate  # On Windows: credit_risk_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python model_training.py
```

### 5. Run Streamlit Application
```bash
streamlit run streamlit_app.py
```

## 📈 Model Training Pipeline

### Step 1: Data Generation & Preprocessing
- Creates realistic synthetic credit risk data (5,000 samples)
- Handles missing values using median/mode imputation
- Feature engineering for better predictive power

### Step 2: Feature Engineering
- Credit-to-income ratio calculation
- Monthly payment computation
- Categorical grouping (age groups, income groups)
- Label encoding for categorical variables

### Step 3: Model Training & Hyperparameter Tuning
- **Logistic Regression**: L1/L2 regularization tuning
- **Random Forest**: n_estimators, max_depth, min_samples_split optimization
- **Gradient Boosting**: learning_rate, n_estimators, max_depth tuning

### Step 4: Model Evaluation
- Cross-validation with 5 folds
- ROC-AUC score for model comparison
- Confusion matrix and classification report
- Feature importance analysis

## 💻 Streamlit Application Features

### 🏠 Home Dashboard
- Dataset overview and key statistics
- Model performance summary
- Feature importance visualization
- System introduction and navigation

### 📊 Data Exploration
- Interactive data visualizations
- Feature distribution analysis
- Correlation heatmaps
- Target variable analysis

### 🎯 Individual Risk Prediction
- User-friendly form for applicant data
- Real-time risk assessment
- Risk probability gauge
- Detailed risk interpretation
- Actionable recommendations

### 📈 Model Performance
- Comprehensive performance metrics
- Feature importance rankings
- Model architecture details
- Cross-validation results

### 📋 Batch Processing
- CSV file upload for multiple predictions
- Progress tracking for large datasets
- Downloadable results
- Summary statistics

## 🎯 Key Features

### Machine Learning Pipeline
- **Data Preprocessing**: Automated handling of missing values and feature scaling
- **Feature Engineering**: Creation of derived features for better model performance
- **Model Selection**: Comparison of multiple algorithms with hyperparameter tuning
- **Cross-Validation**: Robust model evaluation with 5-fold cross-validation
- **Performance Metrics**: ROC-AUC, accuracy, precision, recall, and F1-score

### Web Application
- **Responsive Design**: Professional UI with custom CSS styling
- **Real-Time Predictions**: Instant credit risk assessment
- **Interactive Visualizations**: Plotly charts for better data understanding
- **Batch Processing**: Handle multiple applications efficiently
- **Download Capabilities**: Export predictions and analysis results

## 📊 Model Performance

### Expected Performance Metrics:
- **Accuracy**: ~85-90%
- **ROC-AUC**: ~0.85-0.92
- **Precision**: ~80-88%
- **Recall**: ~75-85%
- **F1-Score**: ~78-86%

### Top Important Features:
1. Debt-to-Income Ratio
2. Credit Amount
3. Duration
4. Monthly Payment
5. Credit-to-Income Ratio
6. Employment Status
7. Savings Account Level
8. Age
9. Income Level
10. Loan Purpose

## 🔄 Hyperparameter Tuning Details

### Logistic Regression:
- **C**: [0.1, 1, 10] - Regularization strength
- **penalty**: ['l2'] - Regularization type

### Random Forest:
- **n_estimators**: [50, 100, 200] - Number of trees
- **max_depth**: [5, 10, None] - Maximum tree depth
- **min_samples_split**: [2, 5] - Minimum samples to split
- **class_weight**: ['balanced', None] - Handle class imbalance

### Gradient Boosting:
- **n_estimators**: [50, 100] - Number of boosting stages
- **learning_rate**: [0.1, 0.2] - Learning rate
- **max_depth**: [3, 5] - Maximum tree depth

## 🚀 Usage Examples

### Individual Prediction:
```python
# Example input
applicant_data = {
    'age': 35,
    'income': 50000,
    'credit_amount': 15000,
    'duration': 24,
    'employment_status': 'employed',
    'purpose': 'car'
}

# Get prediction
risk_assessment = predict_credit_risk(applicant_data)
```

### Batch Processing:
```python
# Load batch data
batch_data = pd.read_csv('new_applications.csv')

# Process predictions
results = process_batch_predictions(batch_data)
```

## 📋 Input Requirements

### Required Fields for Prediction:
- **age**: Integer (18-80)
- **income**: Float (> 0)
- **credit_amount**: Float (> 0)
- **duration**: Integer (1-60)
- **debt_to_income_ratio**: Float (0-5)
- **credit_history_length**: Float (≥ 0)
- **purpose**: String (car, furniture, education, business, vacation, medical)
- **employment_status**: String (employed, self-employed, unemployed, retired)
- **housing**: String (own, rent, free)
- **savings_account**: String (little, moderate, rich, quite_rich)
- **checking_account**: String (little, moderate, rich)

## 🔍 Model Interpretability

### Feature Importance:
The model provides feature importance scores to understand which factors most influence credit risk decisions.

### Risk Interpretation:
- **Low Risk (0-30%)**: Excellent credit profile, minimal risk
- **Moderate Risk (30-60%)**: Average risk, standard terms
- **High Risk (60-85%)**: Elevated risk, additional scrutiny needed
- **Very High Risk (85-100%)**: Significant risk, consider rejection

## 🛡️ Risk Assessment Guidelines

### Low Risk Applicants:
- Competitive interest rates
- Higher credit limits
- Faster approval process
- Premium services eligibility

### High Risk Applicants:
- Additional documentation required
- Higher interest rates or fees
- Lower credit limits
- Collateral requirements
- Enhanced monitoring

## 🔮 Future Enhancements

### Advanced Features:
- **Deep Learning Models**: Neural networks for complex pattern recognition
- **Explainable AI**: SHAP values for model interpretability
- **Real-Time Data Integration**: Live credit bureau data feeds
- **A/B Testing Framework**: Model performance comparison
- **API Development**: RESTful API for external integration

### Technical Improvements:
- **Model Monitoring**: Drift detection and automated retraining
- **Feature Store**: Centralized feature management
- **MLOps Pipeline**: Automated deployment and monitoring
- **Security Enhancements**: Data encryption and access controls

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Contribution:
- Additional ML algorithms (XGBoost, LightGBM, Neural Networks)
- Enhanced visualization features
- API development
- Documentation improvements
- Test suite development

## 📞 Support

For questions, issues, or contributions, please:
1. Open an issue on GitHub
2. Submit a pull request
3. Contact the development team

## 🏆 Acknowledgments

- Inspired by real-world credit risk assessment practices
- Built using state-of-the-art machine learning libraries
- Designed with financial industry best practices

---

**⚠️ Important Note**: This project uses synthetic data for demonstration purposes. For production use with real financial data, ensure compliance with relevant regulations (GDPR, CCPA, etc.) and implement appropriate security measures.