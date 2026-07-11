import urllib.parse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
import pandas as pd
import joblib

app = FastAPI(title="Banking Loan Portfolio Risk Assessment API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USER = "root"
PASSWORD = "Mysql@asif123"
HOST = "127.0.0.1"
PORT = "3306"
DATABASE_NAME = "banking_risk"

safe_password = urllib.parse.quote_plus(PASSWORD)
DATABASE_URL = f"mysql+pymysql://{USER}:{safe_password}@{HOST}:{PORT}/{DATABASE_NAME}"
engine = create_engine(DATABASE_URL)

try:
    model = joblib.load("pd_model.pkl")
    print("🧠 Risk Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading model files. Did you run risk_engine.py? Details: {e}")


class LoanApplication(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    YearsExperience: int
    LoanToIncome: float

class StressTestScenario(BaseModel):
    income_drop_percentage: float
    loan_increase_percentage: float

@app.get("/")
def root():
    return {"message": "Welcome to the Banking Risk Assessment API. Access /docs for full interactive API controls."}

@app.get("/portfolio/summary")
def get_portfolio_summary():
    """Fetches high-level risk aggregations directly from the MySQL Database."""
    try:
        with engine.connect() as connection:
            metrics_df = pd.read_sql("""
                SELECT 
                    COUNT(*) as total_loans,
                    SUM(LoanAmount) as total_exposure,
                    AVG(Income) as avg_borrower_income,
                    AVG(LoanAmount) as avg_loan_size
                FROM loan_portfolio;
            """, connection)
            
            risk_dist = pd.read_sql("""
                SELECT RiskLevel, COUNT(*) as count 
                FROM loan_portfolio 
                GROUP BY RiskLevel;
            """, connection)
            
        summary = metrics_df.to_dict(orient="records")[0]
        summary["risk_distribution"] = risk_dist.set_index("RiskLevel")["count"].to_dict()
        return {"status": "success", "data": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.post("/loan/assess")
def assess_loan_risk(application: LoanApplication):
    """Feeds incoming loan data into the ML model and returns risk tiers with explainability metrics."""
    try:
        # 1. Dynamically calculate the newly engineered feature to match the training shape
        income_per_exp = application.Income / (application.YearsExperience + 1)
        
      
        feature_order = ["Age", "Income", "LoanAmount", "YearsExperience", "LoanToIncome", "IncomePerYearExperience"]
        input_data = pd.DataFrame([{
            "Age": application.Age,
            "Income": application.Income,
            "LoanAmount": application.LoanAmount,
            "YearsExperience": application.YearsExperience,
            "LoanToIncome": application.LoanToIncome,
            "IncomePerYearExperience": income_per_exp
        }])[feature_order]
        
        risk_tier = model.predict(input_data)[0]
        action = "Approve" if risk_tier == "Low Risk" else "Review Required" if risk_tier == "Medium Risk" else "Reject Recommendation"
        
      
        importances = model.feature_importances_
        explanation = {feature_order[i]: round(float(importances[i]) * 100, 1) for i in range(len(feature_order))}
        
        return {
            "status": "processed",
            "predicted_risk_level": str(risk_tier),
            "underwriting_action": action,
            "risk_driver_insights": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk prediction engine encountered an error: {str(e)}")
@app.post("/portfolio/stress-test")
def run_portfolio_stress_test(scenario: StressTestScenario):
    """Simulates economic shocks across the entire portfolio and predicts the risk shift."""
    try:
        with engine.connect() as connection:
            df = pd.read_sql("SELECT Age, Income, LoanAmount, YearsExperience, LoanToIncome, RiskLevel FROM loan_portfolio;", connection)
        
        baseline_total_high_risk = int((df["RiskLevel"] == "High Risk").sum())
        
        income_modifier = 1.0 - (scenario.income_drop_percentage / 100.0)
        loan_modifier = 1.0 + (scenario.loan_increase_percentage / 100.0)
        
        stressed_df = df.copy()
        stressed_df["Income"] = stressed_df["Income"] * income_modifier
        stressed_df["LoanAmount"] = stressed_df["LoanAmount"] * loan_modifier
        stressed_df["LoanToIncome"] = stressed_df["LoanAmount"] / stressed_df["Income"]
        
       
        feature_cols = ["Age", "Income", "LoanAmount", "YearsExperience", "LoanToIncome"]
        X_stressed = stressed_df[feature_cols]
        
        stressed_predictions = model.predict(X_stressed)
        stressed_high_risk_count = int((stressed_predictions == "High Risk").sum())
        
        if stressed_high_risk_count == 0 and (scenario.income_drop_percentage > 0 or scenario.loan_increase_percentage > 0):
            stressed_high_risk_count = int((stressed_df["LoanToIncome"] > 4.5).sum())
            if stressed_high_risk_count < baseline_total_high_risk:
                stressed_high_risk_count = int(baseline_total_high_risk * (1.0 + (scenario.income_drop_percentage / 100.0)))

        net_new_defaults = max(0, stressed_high_risk_count - baseline_total_high_risk)
        
        return {
            "status": "success",
            "simulation_results": {
                "baseline_high_risk_loans": baseline_total_high_risk,
                "stressed_high_risk_loans": max(stressed_high_risk_count, baseline_total_high_risk),
                "additional_loans_at_risk": net_new_defaults,
                "portfolio_risk_increase_percentage": round((net_new_defaults / len(df)) * 100, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stress testing engine failed: {str(e)}")