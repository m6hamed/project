import joblib
import os

# 1. Load your big model
print("Loading model...")
model = joblib.load('random_forest_revenue_model.pkl')

# 2. Save it again with compression (level 3 is usually a good balance)
print("Compressing and saving...")
joblib.dump(model, 'random_forest_revenue_model.pkl', compress=3)

# 3. Check the new size
size_mb = os.path.getsize('random_forest_revenue_model.pkl') / (1024 * 1024)
print(f"New file size: {size_mb:.2f} MB")