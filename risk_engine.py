import urllib.parse
import joblib
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

USER = "root"
PASSWORD = "Mysql@asif123"
HOST = "127.0.0.1"
PORT = "3306"
DATABASE_NAME = "banking_risk"

safe_password = urllib.parse.quote_plus(PASSWORD)
DATABASE_URL = f"mysql+pymysql://{USER}:{safe_password}@{HOST}:{PORT}/{DATABASE_NAME}"
engine = create_engine(DATABASE_URL)

def build_risk_model():
    print("📥 Fetching full dataset from 'loan_portfolio'...")
    with engine.connect() as connection:
        df = pd.read_sql("SELECT * FROM loan_portfolio;", connection)
    
    print(f"✅ Data loaded successfully! Total records: {len(df)}")
   
    df["IncomePerYearExperience"] = df["Income"] / (df["YearsExperience"] + 1)
    
    TARGET_COL = "RiskLevel" 
    FEATURE_COLS = ["Age", "Income", "LoanAmount", "YearsExperience", "LoanToIncome", "IncomePerYearExperience"]
    
  
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🧠 Training the Balanced & Regularized Random Forest Risk Engine...")
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    
  
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model training complete. Test Set Accuracy: {accuracy:.4f}")
    print("\n📋 Detailed Balanced Performance Insight Matrix:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, "pd_model.pkl")
    print("💾 Model artifact successfully saved as 'pd_model.pkl'!")

if __name__ == "__main__":
     build_risk_model()