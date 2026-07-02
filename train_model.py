import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, root_mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Invoice Payment Prediction", layout="wide")

# ─────────────────────────────────────────────
# DATA LOADING (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_process():
    df_raw = pd.read_csv("dataset.csv")

    # ── Cleaning ──────────────────────────────
    shape_raw        = df_raw.shape
    n_dupes          = df_raw.duplicated().sum()
    df = df_raw.copy()
    df.drop_duplicates(inplace=True)
    shape_after_dedup = df.shape

    df.drop(columns=["area_business", "isOpen"], inplace=True)
    df.drop(columns=["document type", "posting_id", "doc_id", "invoice_id"], inplace=True)

    n_nulls = df.isnull().any(axis=1).sum()
    df.dropna(axis=0, inplace=True)
    shape_after_null = df.shape

    dropped_cols = ["area_business", "isOpen", "document type", "posting_id", "doc_id", "invoice_id"]

    # ── Date processing ───────────────────────
    date_cols = ["clear_date", "posting_date"]
    for i in date_cols:
        df[i] = pd.to_datetime(df[i])

    date_cols_int = ["document_create_date", "document_create_date.1", "due_in_date", "baseline_create_date"]
    for i in date_cols_int:
        df[i] = pd.to_datetime(df[i].astype(int).astype(str), format="%Y%m%d")

    # ── Feature engineering ───────────────────
    df["posting_month"]   = df["posting_date"].dt.month
    df["posting_day"]     = df["posting_date"].dt.day
    df["posting_weekday"] = df["posting_date"].dt.dayofweek
    df["due_month"]       = df["due_in_date"].dt.month
    df["due_day"]         = df["due_in_date"].dt.day
    df["due_weekday"]     = df["due_in_date"].dt.dayofweek
    df["days"]            = (df["clear_date"] - df["posting_date"]).dt.days
    df["due_gap"]         = (df["due_in_date"] - df["posting_date"]).dt.days

    df.drop(columns=["clear_date", "posting_date", "document_create_date",
                     "document_create_date.1", "due_in_date", "baseline_create_date"], inplace=True)

    # ── Encoding ──────────────────────────────
    string_cols = ["cust_number", "name_customer"]
    encoders = {}
    label_samples = {}
    for i in string_cols:
        sample_before = df[i].head(5).tolist()
        le = LabelEncoder()
        df[i] = le.fit_transform(df[i])
        encoders[i] = le
        label_samples[i] = {"before": sample_before, "after": df[i].head(5).tolist()}

    freq = df["cust_payment_terms"].value_counts()
    keep = freq[freq >= 100].index
    df["cust_payment_terms"] = df["cust_payment_terms"].where(df["cust_payment_terms"].isin(keep), "OTHER")
    kept_payment_terms = sorted(keep.tolist())

    cols_before_ohe = list(df.columns)
    df = pd.get_dummies(data=df, columns=["business_code", "invoice_currency", "cust_payment_terms"], drop_first=True)
    ohe_cols = [c for c in df.columns if c not in cols_before_ohe]

    df["buisness_year"] = df["buisness_year"].astype(int)
    shape_final = df.shape

    # ── Train-test split ──────────────────────
    X = df.drop(columns=["days"])
    y = df["days"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _, X_val_seen, _, y_val_seen = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # ── RFE ───────────────────────────────────
    rfe = RFE(
        estimator=RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        n_features_to_select=15, step=3
    )
    rfe.fit(X_train, y_train)
    X_train_rfe    = rfe.transform(X_train)
    X_test_rfe     = rfe.transform(X_test)
    X_val_seen_rfe = rfe.transform(X_val_seen)
    selected       = X.columns[rfe.support_].tolist()

    # ── Models ────────────────────────────────
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.8, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_rfe, y_train)
        val_pred  = model.predict(X_val_seen_rfe)
        test_pred = model.predict(X_test_rfe)
        results[name] = {
            "val_r2":   r2_score(y_val_seen, val_pred),
            "val_rmse": root_mean_squared_error(y_val_seen, val_pred),
            "test_r2":  r2_score(y_test, test_pred),
            "test_rmse":root_mean_squared_error(y_test, test_pred),
        }

    # ── Save artifacts ────────────────────────
    final_model = models["Random Forest"]
    joblib.dump(final_model, "model.pkl")
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")
    joblib.dump(selected, "selected_features.pkl")

    return {
        "df_raw": df_raw,
        "df_final": df,
        "shape_raw": shape_raw,
        "shape_after_dedup": shape_after_dedup,
        "shape_after_null": shape_after_null,
        "shape_final": shape_final,
        "n_dupes": n_dupes,
        "n_nulls": n_nulls,
        "dropped_cols": dropped_cols,
        "label_samples": label_samples,
        "kept_payment_terms": kept_payment_terms,
        "ohe_cols": ohe_cols,
        "selected": selected,
        "X": X,
        "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "results": results,
    }


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Running pipeline... this may take a few minutes on first load."):
    data = load_and_process()

df_raw            = data["df_raw"]
df_final          = data["df_final"]
shape_raw         = data["shape_raw"]
shape_after_dedup = data["shape_after_dedup"]
shape_after_null  = data["shape_after_null"]
shape_final       = data["shape_final"]
n_dupes           = data["n_dupes"]
n_nulls           = data["n_nulls"]
dropped_cols      = data["dropped_cols"]
label_samples     = data["label_samples"]
kept_terms        = data["kept_payment_terms"]
ohe_cols          = data["ohe_cols"]
selected          = data["selected"]
X                 = data["X"]
y                 = data["y"]
results           = data["results"]


# ══════════════════════════════════════════════
# SECTION 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════
st.title("Invoice Payment Prediction")
st.header("Section 1: Project Overview")

st.write("""
**Problem Statement**  
Predict the number of days it takes a customer to clear an invoice after it is posted.
""")

col1, col2 = st.columns(2)
col1.metric("Total Rows", shape_raw[0])
col2.metric("Total Columns", shape_raw[1])

st.write("**Input Features (raw columns)**")
st.write(", ".join(df_raw.columns.tolist()))

with st.expander("Feature Descriptions"):
    st.write("""
| Column | Description |
|---|---|
| business_code | Business unit identifier |
| cust_number | Customer account number |
| name_customer | Customer name |
| clear_date | Date the invoice was cleared/paid |
| buisness_year | Fiscal year |
| posting_date | Date the invoice was posted |
| document_create_date | Document creation date (integer YYYYMMDD) |
| document_create_date.1 | Duplicate document creation date |
| due_in_date | Invoice due date (integer YYYYMMDD) |
| invoice_currency | Currency of the invoice |
| total_open_amount | Outstanding invoice amount |
| baseline_create_date | Baseline date for payment terms |
| cust_payment_terms | Customer's standard payment term code |
| invoice_id | Unique invoice identifier |
| isOpen | Whether invoice is still open (1) or cleared (0) |
""")


# ══════════════════════════════════════════════
# SECTION 2 — RAW DATASET EXPLORATION
# ══════════════════════════════════════════════
st.header("Section 2: Raw Dataset Exploration")

st.subheader("First 5 Rows")
st.dataframe(df_raw.head())

st.subheader("Last 5 Rows")
st.dataframe(df_raw.tail())

col1, col2 = st.columns(2)
with col1:
    st.subheader("Data Types")
    st.dataframe(df_raw.dtypes.rename("dtype").reset_index().rename(columns={"index": "Column"}))
with col2:
    st.subheader("Missing Values")
    mv = df_raw.isnull().sum().rename("Missing").reset_index().rename(columns={"index": "Column"})
    mv["% Missing"] = (mv["Missing"] / len(df_raw) * 100).round(2)
    st.dataframe(mv)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Duplicate Rows")
    st.metric("Count", int(n_dupes))
with col4:
    st.subheader("Unique Value Counts")
    st.dataframe(df_raw.nunique().rename("Unique Values").reset_index().rename(columns={"index": "Column"}))

st.subheader("Descriptive Statistics")
st.dataframe(df_raw.describe())


# ══════════════════════════════════════════════
# SECTION 3 — EDA VISUALIZATIONS
# ══════════════════════════════════════════════
st.header("Section 3: EDA Visualizations")

# ── 3.1 Business Year countplot ──────────────
st.subheader("3.1 Business Year Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df_raw["buisness_year"], ax=ax)
ax.set_xlabel("Business Year")
ax.set_ylabel("Count")
ax.set_title("Invoice Count by Business Year")
st.pyplot(fig, use_container_width=False)
plt.close()

# ── 3.2 Invoice Currency countplot ───────────
st.subheader("3.2 Invoice Currency Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df_raw["invoice_currency"], ax=ax)
ax.set_xlabel("Currency")
ax.set_ylabel("Count")
ax.set_title("Invoice Count by Currency")
st.pyplot(fig, use_container_width=False)
plt.close()

# ── 3.3 Total Open Amount histogram ──────────
st.subheader("3.3 Total Open Amount — Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df_raw["total_open_amount"], bins=50, ax=ax)
ax.set_xlabel("Total Open Amount")
ax.set_title("Histogram of Total Open Amount")
st.pyplot(fig, use_container_width=False)
plt.close()


# ── 3.4 Total Open Amount boxplot ────────────
st.subheader("3.4 Total Open Amount — Outlier View")
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x=df_raw["total_open_amount"], ax=ax)
ax.set_xlabel("Total Open Amount")
ax.set_title("Boxplot of Total Open Amount")
st.pyplot(fig, use_container_width=False)
plt.close()

# ── 3.5 Top 10 Customers bar chart ───────────
st.subheader("3.5 Top 10 Customers by Invoice Volume")
fig, ax = plt.subplots(figsize=(10, 4))
topc = df_raw["name_customer"].value_counts().head(10)
sns.barplot(x=topc.index, y=topc.values, ax=ax)
ax.set_xlabel("Customer")
ax.set_ylabel("Invoice Count")
ax.set_title("Top 10 Customers by Number of Invoices")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig, use_container_width=False)
plt.close()

# ── 3.6 ADDED: Missing values heatmap ────────
st.subheader("3.6 Missing Values Heatmap")
fig, ax = plt.subplots(figsize=(12, 3))
sns.heatmap(df_raw.isnull(), cbar=False, yticklabels=False, ax=ax, cmap="viridis")
ax.set_title("Missing Values Heatmap (yellow = missing)")
st.pyplot(fig, use_container_width=False)
plt.close()
st.write("""
Yellow cells indicate missing values. `area_business` is entirely missing (all 50,000 rows), 
and `clear_date` has ~10,000 missing values (open invoices where payment hasn't been recorded yet).
""")

# ── 3.7 ADDED: isOpen distribution ───────────
st.subheader("3.7 Open vs Cleared Invoices")
fig, ax = plt.subplots(figsize=(6, 4))
df_raw["isOpen"].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "salmon"])
ax.set_xticklabels(["Cleared (0)", "Open (1)"], rotation=0)
ax.set_ylabel("Count")
ax.set_title("Invoice Status: Open vs Cleared")
st.pyplot(fig, use_container_width=False)
plt.close()

# ── 3.8 ADDED: Payment terms distribution ────
st.subheader("3.8 Top 15 Customer Payment Terms")
fig, ax = plt.subplots(figsize=(10, 4))
pt = df_raw["cust_payment_terms"].value_counts().head(15)
sns.barplot(x=pt.index, y=pt.values, ax=ax)
ax.set_xlabel("Payment Term Code")
ax.set_ylabel("Count")
ax.set_title("Top 15 Payment Terms by Frequency")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig, use_container_width=False)
plt.close()
st.write("Payment terms drive when invoices are due. Rare terms are later grouped into 'OTHER' during encoding.")

# ── 3.9 ADDED: Business code distribution ────
st.subheader("3.9 Business Code Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
bc = df_raw["business_code"].value_counts()
ax.pie(bc.values, labels=bc.index, autopct="%1.1f%%", startangle=90)
ax.set_title("Invoice Share by Business Code")
st.pyplot(fig, use_container_width=False)
plt.close()


# ══════════════════════════════════════════════
# SECTION 4 — DATA CLEANING
# ══════════════════════════════════════════════
st.header("Section 4: Data Cleaning")

col1, col2, col3 = st.columns(3)
col1.metric("Original Shape",       f"{shape_raw[0]} × {shape_raw[1]}")
col2.metric("After Dedup",          f"{shape_after_dedup[0]} × {shape_after_dedup[1]}")
col3.metric("After Null Removal",   f"{shape_after_null[0]} × {shape_after_null[1]}")

col4, col5 = st.columns(2)
col4.metric("Duplicates Removed", int(n_dupes))
col5.metric("Null Rows Removed",  int(n_nulls))

st.subheader("Columns Dropped")
st.write("""
| Column | Reason |
|---|---|
| `area_business` | 100% missing — no usable data |
| `isOpen` | Leakage: open invoices have no clear_date; they are dropped anyway |
| `document type` | High cardinality identifier, no predictive value |
| `posting_id` | Has only 1 unique value — zero variance |
| `doc_id` | Unique identifier per row — not a feature |
| `invoice_id` | Unique identifier per row — not a feature |
""")

st.subheader("Before vs After")
cleaning_df = pd.DataFrame({
    "Stage": ["Raw", "After drop_duplicates", "After dropna"],
    "Rows": [shape_raw[0], shape_after_dedup[0], shape_after_null[0]],
    "Columns": [shape_raw[1], shape_after_dedup[1], shape_after_null[1]],
})
st.table(cleaning_df)


# ══════════════════════════════════════════════
# SECTION 5 — DATE PROCESSING
# ══════════════════════════════════════════════
st.header("Section 5: Date Processing")

st.write("""
Two types of date columns exist in the raw data and each needs a different conversion strategy.
""")

st.subheader("String → datetime")
st.write("These columns were already in a readable date-string format (`YYYY-MM-DD HH:MM:SS`):")
st.code("clear_date, posting_date  →  pd.to_datetime(df[col])")

st.subheader("Integer → datetime")
st.write("These columns were stored as 8-digit integers in `YYYYMMDD` format:")
st.code("""document_create_date, document_create_date.1
due_in_date, baseline_create_date
→  pd.to_datetime(df[col].astype(int).astype(str), format='%Y%m%d')""")

st.write("""
**Example:** `20200126` → `2020-01-26`

After feature extraction, all 6 date columns are dropped since their information 
has been captured in derived features.
""")

st.write("**Date columns dropped after feature extraction:**")
st.write("`clear_date`, `posting_date`, `document_create_date`, `document_create_date.1`, `due_in_date`, `baseline_create_date`")


# ══════════════════════════════════════════════
# SECTION 6 — FEATURE ENGINEERING
# ══════════════════════════════════════════════
st.header("Section 6: Feature Engineering")

st.write("8 new features are derived from the date columns:")

feat_df = pd.DataFrame({
    "Feature": ["posting_month", "posting_day", "posting_weekday",
                "due_month", "due_day", "due_weekday", "due_gap", "days"],
    "Source": ["posting_date", "posting_date", "posting_date",
               "due_in_date", "due_in_date", "due_in_date",
               "due_in_date − posting_date", "clear_date − posting_date"],
    "Type": ["Calendar", "Calendar", "Calendar",
             "Calendar", "Calendar", "Calendar",
             "Numeric (days)", "Numeric (days) — TARGET"],
    "Purpose": [
        "Seasonality in payment behaviour",
        "Day-of-month patterns",
        "Weekday patterns (Mon=0, Sun=6)",
        "Month the payment was due",
        "Day of month the payment was due",
        "Weekday the payment was due",
        "How far ahead the due date is from posting",
        "Actual days taken to clear invoice"
    ]
})
st.table(feat_df)

st.subheader("Distribution of Target Variable: days")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(y, bins=50, ax=ax)
ax.set_xlabel("Days to Clear Invoice")
ax.set_title("Target Variable Distribution")
st.pyplot(fig, use_container_width=False)
plt.close()
st.write("Most invoices are cleared within a predictable range of days. Outliers represent very delayed payments.")

st.subheader("6.2 Correlation Heatmap (Numeric Features)")
corr_cols = data["df_final"].select_dtypes(include=["int64", "float64", "bool"])
corr = corr_cols.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", ax=ax, linewidths=0.3, annot=False)
ax.set_title("Correlation Heatmap — Numeric Features after Feature Engineering")
st.pyplot(fig, use_container_width=False)
plt.close()
st.write("""
Showing linear correlations between all numeric features and the target `days`.
""")



# ══════════════════════════════════════════════
# SECTION 7 — ENCODING AND TRANSFORMATION
# ══════════════════════════════════════════════
st.header("Section 7: Encoding and Transformation")

st.subheader("7.1 Label Encoding")
st.write("Applied to high-cardinality string columns where ordinal relationships don't matter but we still want a single numeric column:")
for col, samples in label_samples.items():
    st.write(f"**`{col}`**")
    sample_df = pd.DataFrame({"Before": samples["before"], "After": samples["after"]})
    st.table(sample_df)

st.subheader("7.2 Frequency Grouping — cust_payment_terms")
st.write(f"""
`cust_payment_terms` has 74 unique values. Payment terms with **fewer than 100 occurrences** 
are collapsed into the label `OTHER` to prevent sparse dummy columns after one-hot encoding.

**Kept terms ({len(kept_terms)} values):**
""")
st.write(", ".join(kept_terms))

st.subheader("7.3 One-Hot Encoding")
st.write("Applied to low-to-medium cardinality nominal columns. `drop_first=True` avoids the dummy variable trap.")
st.write("**Columns one-hot encoded:** `business_code`, `invoice_currency`, `cust_payment_terms`")
st.write(f"**New dummy columns created ({len(ohe_cols)}):**")
st.write(ohe_cols)

col1, col2 = st.columns(2)
col1.metric("Columns Before OHE", shape_after_null[1] - 6)
col2.metric("Columns After OHE",  shape_final[1])


# ══════════════════════════════════════════════
# SECTION 8 — FINAL PROCESSED DATASET
# ══════════════════════════════════════════════
st.header("Section 8: Final Processed Dataset")

col1, col2 = st.columns(2)
col1.metric("Final Rows",    shape_final[0])
col2.metric("Final Columns", shape_final[1])

st.subheader("Feature Columns (X)")
feature_cols = [c for c in df_final.columns if c != "days"]
st.write(feature_cols)

st.subheader("Target Column (y): days")
st.write(f"Min: {y.min():.0f} | Max: {y.max():.0f} | Mean: {y.mean():.1f} | Std: {y.std():.1f}")

st.subheader("Sample Rows (Final Dataset)")
st.dataframe(df_final.head(10))


# ══════════════════════════════════════════════
# SECTION 9 — TRAIN-TEST SPLIT
# ══════════════════════════════════════════════
st.header("Section 9: Train-Test Split")

n_total = len(X)
n_test  = data["X_test"].shape[0]
n_train = data["X_train"].shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples",   n_total)
col2.metric("Train Samples",   n_train)
col3.metric("Test Samples",    n_test)

st.write(f"""
**Split ratio:** 80% train / 20% test (`test_size=0.2, random_state=42`)

A secondary **seen validation set** is carved from the training set (also 20%) to evaluate models 
on data they were trained on — this helps diagnose overfitting vs. underfitting.

| Split | Rows | Purpose |
|---|---|---|
| Train | {n_train} | Model fitting |
| Seen Validation | ~{int(n_train * 0.2)} | Overfitting check |
| Test | {n_test} | Final unbiased evaluation |
""")


# ══════════════════════════════════════════════
# SECTION 10 — FEATURE SELECTION (RFE)
# ══════════════════════════════════════════════
st.header("Section 10: Feature Selection (RFE)")

st.write("""
**Recursive Feature Elimination (RFE)** iteratively removes the least important features 
using a Random Forest as the base estimator, until the desired number of features remains.

- Estimator: `RandomForestRegressor(n_estimators=200)`
- Target features: **15**
- Step size: **3** (removes 3 features per iteration)
""")

col1, col2 = st.columns(2)
col1.metric("Original Features", len(X.columns))
col2.metric("Selected Features", len(selected))

st.subheader("Selected Features")
sel_df = pd.DataFrame({"#": range(1, len(selected)+1), "Feature": selected})
st.table(sel_df)

st.subheader("Visual: Selected vs Dropped")
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(["Selected", "Dropped"], [len(selected), len(X.columns) - len(selected)],
        color=["steelblue", "lightcoral"])
ax.set_xlabel("Number of Features")
ax.set_title("RFE Feature Selection Result")
for i, v in enumerate([len(selected), len(X.columns) - len(selected)]):
    ax.text(v + 0.2, i, str(v), va="center")
st.pyplot(fig, use_container_width=False)
plt.close()


# ══════════════════════════════════════════════
# SECTION 11 — MODEL TRAINING
# ══════════════════════════════════════════════
st.header("Section 11: Model Training")

models_info = {
    "Random Forest": {
        "params": {"n_estimators": 300, "max_depth": 12, "min_samples_split": 5, "random_state": 42},
        "purpose": "Ensemble of decision trees using bagging. Robust to outliers and overfitting."
    },
    "Gradient Boosting": {
        "params": {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.8, "random_state": 42},
        "purpose": "Boosting-based ensemble. Builds trees sequentially to correct previous errors."
    },
    "XGBoost": {
        "params": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "random_state": 42},
        "purpose": "Optimized gradient boosting with regularization. Fast and highly accurate."
    }
}

for name, info in models_info.items():
    with st.expander(f"{name}"):
        st.write(f"**Purpose:** {info['purpose']}")
        st.write("**Hyperparameters:**")
        st.json(info["params"])


# ══════════════════════════════════════════════
# SECTION 12 — MODEL RESULTS
# ══════════════════════════════════════════════
st.header("Section 12: Model Results")

results_df = pd.DataFrame([
    {
        "Model": name,
        "Seen Val R²":   round(r["val_r2"],   4),
        "Seen Val RMSE": round(r["val_rmse"],  4),
        "Test R²":       round(r["test_r2"],   4),
        "Test RMSE":     round(r["test_rmse"], 4),
    }
    for name, r in results.items()
])

st.subheader("Results Table")
st.dataframe(results_df, use_container_width=True)

best_model = results_df.loc[results_df["Test R²"].idxmax(), "Model"]
st.success(f"Best Model by Test R²: **{best_model}**")

st.subheader("Test R² Comparison")
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(results_df["Model"], results_df["Test R²"], color=["steelblue", "salmon", "mediumseagreen"])
ax.set_ylim(0, 1)
ax.set_ylabel("Test R²")
ax.set_title("Model Comparison — Test R²")
for i, v in enumerate(results_df["Test R²"]):
    ax.text(i, v + 0.01, f"{v:.4f}", ha="center")
st.pyplot(fig, use_container_width=False)
plt.close()

st.subheader("Test RMSE Comparison")
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(results_df["Model"], results_df["Test RMSE"], color=["steelblue", "salmon", "mediumseagreen"])
ax.set_ylabel("Test RMSE (days)")
ax.set_title("Model Comparison — Test RMSE")
for i, v in enumerate(results_df["Test RMSE"]):
    ax.text(i, v + 0.05, f"{v:.4f}", ha="center")
st.pyplot(fig, use_container_width=False)
plt.close()

st.subheader("Seen Validation vs Test R²")
x_pos = range(len(results_df))
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([p - 0.2 for p in x_pos], results_df["Seen Val R²"], width=0.4, label="Seen Val R²", color="steelblue")
ax.bar([p + 0.2 for p in x_pos], results_df["Test R²"],     width=0.4, label="Test R²",     color="salmon")
ax.set_xticks(list(x_pos))
ax.set_xticklabels(results_df["Model"])
ax.set_ylabel("R²")
ax.set_title("Seen Validation vs Test R² (overfitting check)")
ax.legend()
st.pyplot(fig, use_container_width=False)
plt.close()


# ══════════════════════════════════════════════
# SECTION 13 — FINAL MODEL
# ══════════════════════════════════════════════
st.header("Section 13: Final Model")

st.write("""
The final model chosen is **Random Forest Regressor**, retrained on the full RFE-transformed 
training set for maximum data utilization before deployment.
""")

st.subheader("Final Hyperparameters")
st.json({"n_estimators": 300, "max_depth": 12, "min_samples_split": 5, "random_state": 42, "n_jobs": -1})

st.subheader("Saved Artifacts")
artifacts_df = pd.DataFrame({
    "File": ["model.pkl", "selected_features.pkl", "encoders.pkl", "feature_columns.pkl"],
    "Contents": [
        "Trained RandomForestRegressor",
        "Names of the 15 selected features",
        "LabelEncoders for cust_number and name_customer",
        "List of all feature column names before RFE"
    ],
    "Used for": [
        "Making predictions on new invoices",
        "Transforming new data to 15 selected features",
        "Encoding customer string columns at inference",
        "Aligning column order of new input data"
    ]
})
st.table(artifacts_df)


# ══════════════════════════════════════════════
# SECTION 14 — PROJECT SUMMARY
# ══════════════════════════════════════════════
st.header("Section 14: Project Summary")

best_r = results[best_model]
st.write(f"""
**Goal:** Predict the number of days taken to clear an invoice.

**Data Cleaning:**
- Removed {int(n_dupes)} duplicate rows
- Dropped 6 irrelevant/identifier columns
- Removed {int(n_nulls)} rows with null values (mainly open invoices lacking a `clear_date`)

**Feature Engineering:**
- Extracted 7 calendar features from posting and due dates
- Computed `due_gap` (days between posting and due date)
- Computed target `days` (clear_date − posting_date)

**Encoding:**
- Label encoded: `cust_number`, `name_customer`
- Frequency-grouped: `cust_payment_terms` (terms with < 100 occurrences → OTHER)
- One-hot encoded: `business_code`, `invoice_currency`, `cust_payment_terms`

**Feature Selection:**
- RFE reduced {len(X.columns)} features → **{len(selected)} features**

**Models Evaluated:** Random Forest, Gradient Boosting, XGBoost

**Best Model: {best_model}**
- Test R²:   `{best_r['test_r2']:.4f}`
- Test RMSE: `{best_r['test_rmse']:.4f} days`

**Artifacts saved:** `model.pkl`, `selected_features.pkl`, `encoders.pkl`, `feature_columns.pkl`
""")
