heart-disease-ai/
│
├── app.py                          # Main Streamlit application (Stage 1 & 2)
├── train_models.py                 # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── DEPLOYMENT.md                   # Deployment instructions
├── .gitignore                      # Git ignore rules
│
├── .streamlit/                     # Streamlit configuration
│   └── config.toml                 # Theme and server settings
│
├── models/                         # (Generated after training)
│   ├── lifestyle_model.pkl         # Stage 1: Random Forest model
│   ├── clinical_model.pkl          # Stage 2: Gradient Boosting model
│   ├── scaler_life.pkl             # Lifestyle feature scaler
│   └── scaler_clin.pkl             # Clinical feature scaler
│
├── data/                           # (Optional) Your datasets
│   ├── updated_version.csv         # Lifestyle dataset
│   └── Heart_Disease_Prediction.csv # Clinical dataset
│
└── notebooks/                      # (Optional) Jupyter notebooks
    ├── eda.ipynb                   # Exploratory data analysis
    └── model_evaluation.ipynb      # Model performance analysis