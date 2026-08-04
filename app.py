import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        border-left: 5px solid #1976d2;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        data = pd.read_csv('credit_risk_dataset.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please run the model training script first.")
        return None

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        artifacts = joblib.load('credit_risk_model.pkl')
        return artifacts
    except FileNotFoundError:
        st.error("Model file not found. Please run the model training script first.")
        return None

def preprocess_input(input_data, model_artifacts):
    """Preprocess user input for prediction"""
    # Create a copy of input data
    data = input_data.copy()
    
    # Feature engineering (same as in training)
    data['credit_to_income_ratio'] = data['credit_amount'] / data['income']
    data['monthly_payment'] = data['credit_amount'] / data['duration']
    
    # Create age and income groups
    if data['age'] <= 25:
        data['age_group'] = 'Young'
    elif data['age'] <= 35:
        data['age_group'] = 'Adult'
    elif data['age'] <= 50:
        data['age_group'] = 'Middle'
    else:
        data['age_group'] = 'Senior'
    
    # Simplified income grouping
    income_percentiles = [10000, 20000, 30000, 50000]
    if data['income'] <= income_percentiles[0]:
        data['income_group'] = 'Low'
    elif data['income'] <= income_percentiles[1]:
        data['income_group'] = 'Below_Avg'
    elif data['income'] <= income_percentiles[2]:
        data['income_group'] = 'Average'
    elif data['income'] <= income_percentiles[3]:
        data['income_group'] = 'Above_Avg'
    else:
        data['income_group'] = 'High'
    
    # Encode categorical variables
    for col in model_artifacts['categorical_cols']:
        if col in model_artifacts['label_encoders']:
            le = model_artifacts['label_encoders'][col]
            try:
                data[col] = le.transform([str(data[col])])[0]
            except ValueError:
                # Handle unseen categories
                data[col] = 0
    
    # Ensure all required features are present
    feature_values = []
    for feature in model_artifacts['feature_names']:
        if feature in data:
            feature_values.append(data[feature])
        else:
            feature_values.append(0)  # Default value for missing features
    
    return np.array(feature_values).reshape(1, -1)

def predict_credit_risk(input_data, model_artifacts):
    """Make credit risk prediction"""
    processed_data = preprocess_input(input_data, model_artifacts)
    scaled_data = model_artifacts['scaler'].transform(processed_data)
    
    prediction = model_artifacts['model'].predict(scaled_data)[0]
    probability = model_artifacts['model'].predict_proba(scaled_data)[0]
    
    return prediction, probability

def create_risk_visualization(probability):
    """Create a risk meter visualization"""
    risk_score = probability[1]  # Probability of being high risk
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Credit Risk Score (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if risk_score > 0.5 else "darkgreen"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Risk Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data and model
    data = load_data()
    model_artifacts = load_model()
    
    if data is None or model_artifacts is None:
        st.error("Please ensure the dataset and model files are available.")
        st.info("Run the model training script first to generate the required files.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", 
                           ["🏠 Home", "📊 Data Exploration", "🎯 Risk Prediction", "📈 Model Performance", "📋 Batch Prediction"])
    
    if page == "🏠 Home":
        st.header("Welcome to Credit Risk Assessment System")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Good Credit", len(data[data['credit_risk'] == 0]))
        with col3:
            st.metric("Bad Credit", len(data[data['credit_risk'] == 1]))
        with col4:
            st.metric("Default Rate", f"{(len(data[data['credit_risk'] == 1]) / len(data) * 100):.1f}%")
        
        st.subheader("System Overview")
        st.write("""
        This Credit Risk Assessment System uses machine learning to evaluate the creditworthiness of loan applicants.
        
        **Key Features:**
        - 🤖 **AI-Powered Risk Assessment**: Uses advanced machine learning algorithms
        - 📊 **Comprehensive Analysis**: Analyzes multiple financial and demographic factors
        - ⚡ **Real-time Predictions**: Instant risk assessment for new applications
        - 📈 **Performance Monitoring**: Track model accuracy and performance metrics
        - 📋 **Batch Processing**: Process multiple applications simultaneously
        
        **How it works:**
        1. Input applicant information (age, income, credit amount, etc.)
        2. The system processes the data using trained ML models
        3. Receive instant risk assessment with probability scores
        4. Make informed lending decisions based on the predictions
        """)
        
        # Display feature importance if available
        if hasattr(model_artifacts['model'], 'feature_importances_'):
            st.subheader("Most Important Factors in Credit Assessment")
            importance_df = pd.DataFrame({
                'Feature': model_artifacts['feature_names'],
                'Importance': model_artifacts['model'].feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Feature Importance in Credit Risk Model")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📊 Data Exploration":
        st.header("Data Exploration")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            st.dataframe(data.describe())
        
        with col2:
            st.subheader("Target Distribution")
            risk_counts = data['credit_risk'].value_counts()
            fig = px.pie(values=risk_counts.values, names=['Good Credit', 'Bad Credit'],
                        title="Credit Risk Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        # Select feature to visualize
        numerical_features = ['age', 'income', 'credit_amount', 'duration', 'debt_to_income_ratio']
        selected_feature = st.selectbox("Select Feature to Visualize", numerical_features)
        
        fig = px.histogram(data, x=selected_feature, color='credit_risk', 
                          title=f'Distribution of {selected_feature} by Credit Risk',
                          nbins=30)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        corr_matrix = data[numerical_features + ['credit_risk']].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🎯 Risk Prediction":
        st.header("Individual Risk Prediction")
        st.write("Enter applicant information to assess credit risk:")
        
        # Create input form
        with st.form("risk_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=80, value=35)
                income = st.number_input("Annual Income ($)", min_value=1000, value=50000)
                credit_amount = st.number_input("Credit Amount ($)", min_value=100, value=10000)
                duration = st.number_input("Duration (months)", min_value=1, max_value=60, value=24)
                debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=5.0, value=0.3)
                credit_history_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
            
            with col2:
                purpose = st.selectbox("Loan Purpose", 
                                     ['car', 'furniture', 'education', 'business', 'vacation', 'medical'])
                employment_status = st.selectbox("Employment Status",
                                               ['employed', 'self-employed', 'unemployed', 'retired'])
                housing = st.selectbox("Housing", ['own', 'rent', 'free'])
                savings_account = st.selectbox("Savings Account",
                                             ['little', 'moderate', 'rich', 'quite_rich'])
                checking_account = st.selectbox("Checking Account",
                                              ['little', 'moderate', 'rich'])
            
            submit_button = st.form_submit_button("Assess Credit Risk", type="primary")
            
            if submit_button:
                # Prepare input data
                input_data = {
                    'age': age,
                    'income': income,
                    'credit_amount': credit_amount,
                    'duration': duration,
                    'debt_to_income_ratio': debt_to_income_ratio,
                    'credit_history_length': credit_history_length,
                    'purpose': purpose,
                    'employment_status': employment_status,
                    'housing': housing,
                    'savings_account': savings_account,
                    'checking_account': checking_account
                }
                
                # Make prediction
                prediction, probability = predict_credit_risk(input_data, model_artifacts)
                
                # Display results
                st.subheader("Risk Assessment Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"
                    risk_color = "🔴" if prediction == 1 else "🟢"
                    st.markdown(f"### {risk_color} {risk_label}")
                
                with col2:
                    st.metric("Risk Probability", f"{probability[1]*100:.1f}%")
                
                with col3:
                    st.metric("Safe Probability", f"{probability[0]*100:.1f}%")
                
                # Risk visualization
                fig = create_risk_visualization(probability)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk interpretation
                if prediction == 1:
                    st.markdown("""
                    <div class="risk-high">
                    <h4>⚠️ HIGH RISK ASSESSMENT</h4>
                    <p>This applicant shows indicators of higher credit risk. Consider:</p>
                    <ul>
                        <li>Requiring additional documentation</li>
                        <li>Higher interest rates</li>
                        <li>Lower credit limits</li>
                        <li>Additional guarantees or collateral</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="risk-low">
                    <h4>✅ LOW RISK ASSESSMENT</h4>
                    <p>This applicant shows positive credit indicators. Benefits may include:</p>
                    <ul>
                        <li>Faster approval process</li>
                        <li>Competitive interest rates</li>
                        <li>Higher credit limits</li>
                        <li>Premium banking services</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif page == "📈 Model Performance":
        st.header("Model Performance Metrics")
        
        if 'model_performance' in model_artifacts:
            perf = model_artifacts['model_performance']
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{perf['accuracy']:.4f}")
            with col2:
                st.metric("AUC Score", f"{perf['auc_score']:.4f}")
            with col3:
                st.metric("CV AUC Mean", f"{perf['cv_auc_mean']:.4f}")
            with col4:
                st.metric("CV AUC Std", f"{perf['cv_auc_std']:.4f}")
        
        st.subheader("Model Information")
        st.write(f"**Model Type:** {type(model_artifacts['model']).__name__}")
        st.write(f"**Number of Features:** {len(model_artifacts['feature_names'])}")
        st.write(f"**Features Used:** {', '.join(model_artifacts['feature_names'])}")
        
        # Feature importance visualization
        if hasattr(model_artifacts['model'], 'feature_importances_'):
            st.subheader("Feature Importance Analysis")
            importance_df = pd.DataFrame({
                'Feature': model_artifacts['feature_names'],
                'Importance': model_artifacts['model'].feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df.tail(15), x='Importance', y='Feature', 
                        orientation='h', title="Top 15 Most Important Features")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📋 Batch Prediction":
        st.header("Batch Credit Risk Assessment")
        st.write("Upload a CSV file with applicant data for batch processing.")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Data Preview")
                st.dataframe(batch_data.head())
                
                if st.button("Process Batch Predictions", type="primary"):
                    # Process predictions for batch data
                    predictions = []
                    probabilities = []
                    
                    progress_bar = st.progress(0)
                    
                    for idx, (_, row) in enumerate(batch_data.iterrows()):
                        input_data = row.to_dict()
                        pred, prob = predict_credit_risk(input_data, model_artifacts)
                        predictions.append(pred)
                        probabilities.append(prob[1])
                        progress_bar.progress((idx + 1) / len(batch_data))

                    
                    # Add predictions to the dataframe
                    batch_data['predicted_risk'] = predictions
                    batch_data['risk_probability'] = probabilities
                    batch_data['risk_label'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
                    
                    st.subheader("Prediction Results")
                    st.dataframe(batch_data)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Processed", len(batch_data))
                    with col2:
                        high_risk_count = sum(predictions)
                        st.metric("High Risk", high_risk_count)
                    with col3:
                        high_risk_rate = high_risk_count / len(batch_data) * 100
                        st.metric("High Risk Rate", f"{high_risk_rate:.1f}%")
                    
                    # Download results
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="credit_risk_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV file contains the required columns.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Credit Risk Assessment System | Built with Streamlit & Scikit-learn | © 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()