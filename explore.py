import pandas as pd

df = pd.read_csv(r'C:\Users\lenovo\sales-predictor\train.csv', parse_dates=['date'])

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nSample stats:\n", df.describe())