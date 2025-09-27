import pandas as pd

url = "https://www.dropbox.com/scl/fi/huvcvi6vv5w986a0n3znp/Employee_Performance_Retention.csv?rlkey=vhd2bpbjrxzakkdi0q3pyaygz&e=1&st=gvsv4dza&dl=1"
df = pd.read_csv(url)

print("DATASET INSPECTION")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\\nFirst 3 rows:")
print(df.head(3))
print(f"\\nColumn dtypes:")
print(df.dtypes)
print(f"\\nMissing values:")
print(df.isnull().sum())
print(f"\\nAttrition distribution:")
print(df['Attrition'].value_counts())
print(f"\\nPerformance_Rating distribution:")
print(df['Performance_Rating'].value_counts().sort_index())