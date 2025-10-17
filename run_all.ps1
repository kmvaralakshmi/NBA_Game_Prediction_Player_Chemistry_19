# 🏀 NBA Game Prediction - Automated Pipeline (PowerShell Version)

Write-Host "🏀 Starting NBA Chemistry Prediction Pipeline..." -ForegroundColor Cyan

# Step 1: Activate virtual environment (Assuming 'venv' exists)
Write-Host "`n📦 Activating virtual environment..." -ForegroundColor Yellow
.\\venv\Scripts\Activate.ps1

# Step 2: Preprocessing
Write-Host "`n📊 Step 1: Preprocessing data..." -ForegroundColor Yellow
python src\preprocess.py

# Step 3: Train Logistic Regression Baseline
Write-Host "`n🧠 Step 2: Training Logistic Regression Baseline..." -ForegroundColor Yellow
python src\logistic_baseline.py

# Step 4: Train Custom Chemistry Model (Static)
Write-Host "`n🤝 Step 3: Training Chemistry Model (Static)..." -ForegroundColor Yellow
python src\chemistry_model.py

# Step 5: Train Dynamic Chemistry Model
Write-Host "`n🌟 Step 4: Training Dynamic Chemistry Model..." -ForegroundColor Yellow
python src\dynamic_chemistry_model.py

# Step 6: Train Random Forest Benchmark (NEW ADDITION)
Write-Host "`n🌳 Step 5: Training Random Forest Benchmark..." -ForegroundColor Yellow
python src\random_forest_benchmark.py

# Step 7: Evaluate Results
Write-Host "`n📈 Step 6: Evaluating Model Performance..." -ForegroundColor Yellow
# Note: This step currently evaluates the Logistic Regression output
python src\evaluate.py

# Step 8: Explainability (SHAP)
Write-Host "`n🔍 Step 7: Generating Explainability Visuals..." -ForegroundColor Yellow
python src\explainability.py

# Step 9: Dashboard Info
Write-Host "`n✅ All steps completed successfully! Models are ready." -ForegroundColor Green
Write-Host "To launch the Streamlit dashboard, run:" -ForegroundColor Cyan
Write-Host "python -m streamlit run app.py" -ForegroundColor Magenta
