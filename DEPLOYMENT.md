# 🚀 Deployment Guide

## Step-by-Step Instructions

### 1. Setup Local Environment

\`\`\`bash
mkdir heart-disease-ai
cd heart-disease-ai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

### 2. Train Models

\`\`\`bash
python train_models.py
\`\`\`

### 3. Test Locally

\`\`\`bash
streamlit run app.py
\`\`\`

### 4. Push to GitHub

\`\`\`bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-ai.git
git push -u origin main
\`\`\`

### 5. Deploy to Streamlit Cloud

1. Go to share.streamlit.io
2. Click "New app"
3. Select repository
4. Set main file: app.py
5. Click "Deploy!"

## Troubleshooting

- **Models not loading**: Check .pkl files are in repo
- **Memory errors**: Reduce model size in train_models.py
- **Import errors**: Verify requirements.txt versions