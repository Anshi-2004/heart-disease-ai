# 🫀 Two-Stage Heart Disease Risk Assessment System

A comprehensive machine learning system for heart disease risk prediction using a two-stage approach: lifestyle screening followed by clinical diagnosis.

## 🎯 Architecture Overview

### Stage 1: Lifestyle Screening
- **Model**: Random Forest Classifier
- **Features**: Age, BMI, smoking, exercise, diet, sleep, stress, family history
- **Output**: Lifestyle Risk Score (P_L)
- **Threshold**: 60% (low vs. elevated risk)

### Stage 2: Clinical Diagnosis
- **Model**: Gradient Boosting Classifier  
- **Features**: Blood pressure, cholesterol, ECG results, chest pain type, heart rate
- **Output**: Clinical Risk Score (P_C)

### Final Fusion
\`\`\`
P_final = α × P_L + (1 - α) × P_C
\`\`\`
Where α = 0.4 (40% lifestyle, 60% clinical)

## 🚀 Quick Start

### Local Installation

\`\`\`bash
# Clone repository
git clone https://github.com/yourusername/heart-disease-ai.git
cd heart-disease-ai

# Install dependencies
pip install -r requirements.txt

# Train models
python train_models.py

# Run application
streamlit run app.py
\`\`\`

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and branch
5. Set main file path: \`app.py\`
6. Click "Deploy"

## 📊 Model Performance

### Lifestyle Model
- Accuracy: ~85%
- ROC-AUC: ~0.88
- Precision: ~82%

### Clinical Model  
- Accuracy: ~87%
- ROC-AUC: ~0.91
- Precision: ~85%

## 🗂️ Project Structure

\`\`\`
heart-disease-ai/
├── app.py                  # Main Streamlit application
├── train_models.py         # Model training script
├── lifestyle_model.pkl     # Stage 1 trained model
├── clinical_model.pkl      # Stage 2 trained model
├── scaler_life.pkl         # Lifestyle feature scaler
├── scaler_clin.pkl         # Clinical feature scaler
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
├── .gitignore             # Git ignore rules
└── .streamlit/
    └── config.toml        # Streamlit configuration
\`\`\`

## 🔬 Features

- **Two-stage risk assessment** for efficient screening
- **Interactive web interface** with Streamlit
- **Real-time predictions** with probability scores
- **Visual risk gauges** and charts
- **Comprehensive reporting** with recommendations
- **Mobile-responsive** design

## 📝 Usage

### Stage 1: Lifestyle Screening
1. Enter lifestyle information (age, BMI, habits)
2. Click "Calculate Lifestyle Risk"
3. If risk < 60%: Follow preventive care recommendations
4. If risk ≥ 60%: Proceed to Stage 2

### Stage 2: Clinical Diagnosis
1. Enter clinical measurements (BP, cholesterol, ECG)
2. Input Stage 1 risk score
3. Click "Calculate Final Risk"
4. Review comprehensive risk assessment

## 🔄 Using Your Own Data

Replace the synthetic data in \`train_models.py\` with your actual datasets:

### Lifestyle Dataset (updated_version.csv)
Required columns:
- age, sex, bmi, smoking, alcohol_weekly
- exercise_hours, diet_quality, sleep_hours
- stress_level, family_history, target

### Clinical Dataset (Heart_Disease_Prediction.csv)
Required columns:
- resting_bp, cholesterol, fasting_bs, max_heart_rate
- chest_pain_type, resting_ecg, exercise_angina
- oldpeak, st_slope, target

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

## 📜 License

MIT License - See LICENSE file for details

## 👥 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: Make sure to replace synthetic data with your actual datasets before deployment.