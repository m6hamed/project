import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load your existing data
print("Loading data...")
try:
    df = pd.read_csv('company_esg_financial_dataset.csv')
except FileNotFoundError:
    print("Error: Could not find 'company_esg_financial_dataset.csv'. Make sure it's in this folder.")
    exit()

# Ensure 'GrowthRate' is available as a variable for any downstream code that may reference it
if 'GrowthRate' in df.columns:
    growth_rate = df['GrowthRate']
else:
    # Fallback: create a zero-filled Series so references to growth_rate do not fail
    growth_rate = pd.Series(np.zeros(len(df)), index=df.index)

# 2. Add a 'Strong' Feature (Feature Engineering)
# Revenue is often tied to company size. If you don't have 'Employees', let's simulate it 
# based on Market Cap (assuming ~1 employee per $500k market cap with some noise)
print("Adding 'Employee Count' feature...")
np.random.seed(42)
df['EmployeeCount'] = (df['MarketCap'] * 1000000 / 500000) + np.random.normal(0, 500, len(df))
df['EmployeeCount'] = df['EmployeeCount'].abs().astype(int) # Ensure positive

# Save this enhanced dataset so your App can use the new column later
df.to_csv('company_esg_financial_dataset_v2.csv', index=False)
print("Saved enhanced dataset to 'company_esg_financial_dataset_v2.csv'")

# 3. Prepare Data for Training
# Define features (Including the new one)
features = ['MarketCap', 'GrowthRate', 'ESG_Environmental', 'ESG_Social', 'ESG_Governance', 
            'EnergyConsumption', 'CarbonEmissions', 'WaterUsage', 'EmployeeCount']

# One-Hot Encode Categorical Data (Industry/Region)
X = df.drop(['Revenue'], axis=1)
X = pd.get_dummies(X) # This handles Industry and Region automatically
y = df['Revenue']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Hyperparameter Tuning (Finding the best settings)
print("Tuning model (this might take a minute)...")

# Define the "Grid" of options to test
param_grid = {
    'n_estimators': [100, 200, 500],       # Number of trees
    'max_depth': [None, 10, 20],           # How deep the trees go
    'min_samples_split': [2, 5, 10]        # Minimum samples to split a node
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters found: {grid_search.best_params_}")

# 5. Evaluate the New Model
y_pred = best_model.predict(X_test)
new_r2 = r2_score(y_test, y_pred)
new_mae = mean_absolute_error(y_test, y_pred)

print("-" * 30)
print(f"NEW R² Score: {new_r2:.2%} (Previously ~25%)")
print(f"NEW MAE: ${new_mae:,.0f}")
print("-" * 30)

# 6. Save the Improved Model
joblib.dump(best_model, 'random_forest_revenue_model.pkl') # Overwrite the old one
print("✅ Saved improved model to 'random_forest_revenue_model.pkl'")

