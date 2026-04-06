import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
def load_data():
    print("── 1. Loading Data ──")
    df = pd.read_csv(r'C:\Users\lenovo\sales-predictor\train.csv', parse_dates=['date'])
    print(f"  Rows loaded       : {len(df):,}")
    print(f"  Columns           : {df.columns.tolist()}")
    daily = df.groupby(['date', 'store_nbr']).agg(
        total_sales=('sales', 'sum'),
        avg_onpromotion=('onpromotion', 'mean')
    ).reset_index()
    print(f"  After aggregation : {len(daily):,} rows")
    return daily

# ─────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────
def clean_data(df):
    print("\n── 2. Cleaning Data ──")
    missing = df.isnull().sum()
    print(f"  Missing values:\n{missing}")
    before = len(df)
    df = df[df['total_sales'] >= 0]
    print(f"  Removed {before - len(df)} negative sales rows")
    return df

# ─────────────────────────────────────────
# 3. OUTLIER DETECTION & REMOVAL
# ─────────────────────────────────────────
def handle_outliers(df):
    print("\n── 3. Outlier Detection ──")
    sales = df['total_sales']

    print(f"  Mean   : {sales.mean():.2f}")
    print(f"  Median : {sales.median():.2f}")
    print(f"  Std    : {sales.std():.2f}")
    print(f"  Min    : {sales.min():.2f}")
    print(f"  Max    : {sales.max():.2f}")

    print("\n  Percentiles:")
    for p in [90, 95, 99, 99.5]:
        print(f"    {p}th : {sales.quantile(p/100):.2f}")

    # IQR
    Q1        = sales.quantile(0.25)
    Q3        = sales.quantile(0.75)
    IQR       = Q3 - Q1
    iqr_upper = Q3 + 1.5 * IQR
    iqr_lower = Q1 - 1.5 * IQR
    iqr_count = len(df[(sales < iqr_lower) | (sales > iqr_upper)])
    print(f"\n  IQR method       : {iqr_count} outliers ({iqr_count/len(df)*100:.1f}%)")
    print(f"  IQR bounds       : {iqr_lower:.2f} to {iqr_upper:.2f}")

    # Z-score
    z_scores = np.abs(stats.zscore(sales))
    z_count  = len(df[z_scores > 3])
    print(f"  Z-score method   : {z_count} outliers ({z_count/len(df)*100:.1f}%)")

    # 99th percentile
    p99       = sales.quantile(0.99)
    p99_count = len(df[sales > p99])
    print(f"  99th pct method  : {p99_count} outliers ({p99_count/len(df)*100:.1f}%)")

    # Apply IQR
    before = len(df)
    df = df[(df['total_sales'] >= iqr_lower) & (df['total_sales'] <= iqr_upper)]
    print(f"\n  Applied IQR — removed {before - len(df)} rows")
    print(f"  Remaining rows   : {len(df):,}")
    return df

# ─────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────
def engineer_features(df):
    print("\n── 4. Feature Engineering ──")
    df = df.copy().sort_values(['store_nbr', 'date'])

    df['day_of_week']  = df['date'].dt.dayofweek
    df['month']        = df['date'].dt.month
    df['year']         = df['date'].dt.year
    df['day']          = df['date'].dt.day
    df['quarter']      = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    df['lag_7']        = df.groupby('store_nbr')['total_sales'].shift(7)
    df['lag_30']       = df.groupby('store_nbr')['total_sales'].shift(30)
    df['rolling_7']    = (
        df.groupby('store_nbr')['total_sales']
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    before = len(df)
    df = df.dropna()
    print(f"  Dropped {before - len(df)} rows due to lag NaNs")
    print(f"  Features: day_of_week, month, year, day, quarter,")
    print(f"            week_of_year, is_weekend, lag_7, lag_30, rolling_7")
    return df

# ─────────────────────────────────────────
# 5. FEATURE SELECTION
# ─────────────────────────────────────────
FEATURES = [
    'store_nbr', 'day_of_week', 'month', 'year', 'day',
    'is_weekend', 'quarter', 'week_of_year',
    'avg_onpromotion', 'lag_7', 'lag_30', 'rolling_7'
]
TARGET = 'total_sales'

# ─────────────────────────────────────────
# 6. SCALING
# ─────────────────────────────────────────
def scale_features(X_train, X_test):
    print("\n── 6. Scaling Features ──")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("  StandardScaler applied (fit on train only)")
    return X_train_scaled, X_test_scaled, scaler

# ─────────────────────────────────────────
# 7. TRAIN + EVALUATE
# ─────────────────────────────────────────
def train_model():
    df = load_data()
    df = clean_data(df)
    df = handle_outliers(df)
    df = engineer_features(df)

    print("\n── 5. Feature Selection ──")
    print(f"  Features : {FEATURES}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train size : {len(X_train):,}")
    print(f"  Test size  : {len(X_test):,}")

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    print("\n── 7. Training Model ──")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    print("  Model trained successfully")

    print("\n── 8. Overfitting Check ──")
    train_r2 = model.score(X_train_s, y_train)
    test_r2  = model.score(X_test_s,  y_test)
    gap      = train_r2 - test_r2
    print(f"  Train R2 : {train_r2:.4f}")
    print(f"  Test  R2 : {test_r2:.4f}")
    print(f"  Gap      : {gap:.4f}")
    if gap < 0.05:
        print("  ✓ No overfitting detected")
    elif gap < 0.15:
        print("  ⚠ Mild overfitting — acceptable")
    else:
        print("  ✗ Overfitting — reduce max_depth or increase min_samples_leaf")

    print("\n── 9. Cross Validation (5-fold) ──")
    cv_scores = cross_val_score(
        model, X_train_s, y_train, cv=5, scoring='r2', n_jobs=-1
    )
    print(f"  CV scores : {[round(s, 4) for s in cv_scores]}")
    print(f"  Mean R2   : {cv_scores.mean():.4f}")
    print(f"  Std R2    : {cv_scores.std():.4f}")
    if cv_scores.std() < 0.02:
        print("  ✓ Low variance — model is stable")
    else:
        print("  ⚠ High variance — model is inconsistent")

    print("\n── 10. Final Metrics ──")
    y_pred = model.predict(X_test_s)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"  MAE : {mae:.2f}")
    print(f"  R2  : {r2:.4f}")

    print("\n── 11. Feature Importance ──")
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    for feat, imp in importances.sort_values(ascending=False).items():
        bar = '█' * int(imp * 50)
        print(f"  {feat:<20} {bar} {imp:.4f}")

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("\n── 12. Saved model.pkl and scaler.pkl ──")

    return model, scaler, mae, r2


if __name__ == '__main__':
    train_model()