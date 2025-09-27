import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import pandas as pd
    import numpy as np
    import sklearn
    import xgboost as xgb
    print("✓ All packages imported successfully!")
    print(f"Pandas version: {pd.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"XGBoost version: {xgb.__version__}")
except ImportError as e:
    print(f"✗ Import error: {e}")