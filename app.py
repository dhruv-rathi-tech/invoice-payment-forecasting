import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Invoice Payment Predictor", layout="centered")

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    selected_features = joblib.load("selected_features.pkl")

return model, encoders, feature_columns, selected_features

try:
    model, encoders, feature_columns, selected_features = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Artifact file not found: {e}\n\n"
        "Please run `streamlit_app.py` first to train the model and generate the .pkl files."
    )
    st.stop()


# ─────────────────────────────────────────────
# DERIVE VALID OPTIONS FROM ENCODERS & FEATURE COLUMNS
# ─────────────────────────────────────────────

# Known customers from label encoders
known_customers    = sorted(encoders["cust_number"].classes_.tolist())
known_cust_names   = sorted(encoders["name_customer"].classes_.tolist())

# Business codes: derive from OHE column names in feature_columns
# OHE pattern: business_code_<value>, drop_first=True so one code is the baseline
business_code_cols = [c for c in feature_columns if c.startswith("business_code_")]
business_codes_ohe = [c.replace("business_code_", "") for c in business_code_cols]
# The dropped (baseline) code is the one NOT in ohe columns.
# We know from the dataset there are 6 codes — expose all 6 to the user.
# The baseline just means all its OHE cols are 0.
all_business_codes = business_codes_ohe  # baseline col is implicit (all zeros)

# Currencies
currency_cols  = [c for c in feature_columns if c.startswith("invoice_currency_")]
currencies_ohe = [c.replace("invoice_currency_", "") for c in currency_cols]
all_currencies = currencies_ohe  # baseline currency is implicit

# Payment terms
payment_term_cols  = [c for c in feature_columns if c.startswith("cust_payment_terms_")]
payment_terms_ohe  = [c.replace("cust_payment_terms_", "") for c in payment_term_cols]
# Add "OTHER" bucket (it maps to all zeros if OTHER is the baseline, or has a col if not)
all_payment_terms  = payment_terms_ohe


# ─────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────
def build_feature_row(
    business_code, cust_number, name_customer,
    buisness_year, posting_date, due_in_date,
    invoice_currency, total_open_amount, cust_payment_terms
):
    # ── 1. Date feature engineering ───────────
    posting_dt = pd.Timestamp(posting_date)
    due_dt     = pd.Timestamp(due_in_date)

    posting_month   = posting_dt.month
    posting_day     = posting_dt.day
    posting_weekday = posting_dt.dayofweek
    due_month       = due_dt.month
    due_day         = due_dt.day
    due_weekday     = due_dt.dayofweek
    due_gap         = (due_dt - posting_dt).days

    # ── 2. Label encode customer fields ───────
    le_cust   = encoders["cust_number"]
    le_name   = encoders["name_customer"]

    if cust_number in le_cust.classes_:
        enc_cust = int(le_cust.transform([cust_number])[0])
    else:
        # unseen customer: use median encoded value as fallback
        enc_cust = int(np.median(le_cust.transform(le_cust.classes_)))

    if name_customer in le_name.classes_:
        enc_name = int(le_name.transform([name_customer])[0])
    else:
        enc_name = int(np.median(le_name.transform(le_name.classes_)))

    # ── 3. Build base row with all feature_columns set to 0 ───
    row = {col: 0 for col in feature_columns}

    # ── 4. Fill scalar features ───────────────
    row["cust_number"]      = enc_cust
    row["name_customer"]    = enc_name
    row["buisness_year"]    = int(buisness_year)
    row["total_open_amount"]= float(total_open_amount)
    row["posting_month"]    = posting_month
    row["posting_day"]      = posting_day
    row["posting_weekday"]  = posting_weekday
    row["due_month"]        = due_month
    row["due_day"]          = due_day
    row["due_weekday"]      = due_weekday
    row["due_gap"]          = due_gap

    # ── 5. OHE: business_code ─────────────────
    bc_col = f"business_code_{business_code}"
    if bc_col in row:
        row[bc_col] = 1
    # else: this is the baseline (dropped) category → all zeros, correct

    # ── 6. OHE: invoice_currency ──────────────
    curr_col = f"invoice_currency_{invoice_currency}"
    if curr_col in row:
        row[curr_col] = 1

    # ── 7. OHE: cust_payment_terms ────────────
    # Apply the same freq grouping: if term isn't in OHE columns, it was rare → OTHER
    pt_col = f"cust_payment_terms_{cust_payment_terms}"
    if pt_col in row:
        row[pt_col] = 1
    # if not found (rare term or OTHER baseline), all zeros

    # ── 8. Return as ordered DataFrame ────────
    return pd.DataFrame([row])[feature_columns]


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("Invoice Payment Date Predictor")
st.write(
    "Enter the invoice details below. The model will predict how many days "
    "it will take for the payment to clear, and calculate the expected payment date."
)

st.divider()

# ── Section A: Customer Info ──────────────────
st.subheader("Customer Information")

col1, col2 = st.columns(2)
with col1:
    cust_input_mode = st.radio(
        "Customer Number",
        ["Select existing", "Enter new"],
        horizontal=True
    )
    if cust_input_mode == "Select existing":
        cust_number = st.selectbox("Customer Number", known_customers)
    else:
        cust_number = st.text_input("Customer Number (new)", placeholder="e.g. 0200123456")

with col2:
    name_input_mode = st.radio(
        "Customer Name",
        ["Select existing", "Enter new"],
        horizontal=True
    )
    if name_input_mode == "Select existing":
        name_customer = st.selectbox("Customer Name", known_cust_names)
    else:
        name_customer = st.text_input("Customer Name (new)", placeholder="e.g. ACME Corp")

st.divider()

# ── Section B: Invoice Details ────────────────
st.subheader("Invoice Details")

col3, col4 = st.columns(2)
with col3:
    bc_options = sorted(all_business_codes) if all_business_codes else ["U001", "CA02"]
    bc_input_mode = st.radio(
        "Business Code",
        ["Select existing", "Enter new"],
        horizontal=True
    )
    if bc_input_mode == "Select existing":
        business_code = st.selectbox("Business Code", bc_options)
    else:
        business_code = st.text_input("Business Code (new)", placeholder="e.g. US01")

    invoice_currency = st.selectbox(
        "Invoice Currency",
        sorted(all_currencies) if all_currencies else ["USD", "CAD"]
    )

    total_open_amount = st.number_input(
        "Total Open Amount (invoice value)",
        min_value=0.01,
        value=10000.00,
        step=100.00,
        format="%.2f"
    )

with col4:
    buisness_year = st.selectbox(
        "Business Year",
        options=[2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        index=5
    )

    # Payment terms — show OHE-derived terms + OTHER
    pt_display = sorted(payment_terms_ohe) if payment_terms_ohe else []
    if "OTHER" not in pt_display:
        pt_display = ["OTHER"] + pt_display
    cust_payment_terms = st.selectbox("Customer Payment Terms", pt_display)

st.divider()

# ── Section C: Dates ──────────────────────────
st.subheader("Invoice Dates")

col5, col6 = st.columns(2)
with col5:
    posting_date = st.date_input(
        "Posting Date (when invoice was posted)",
        value=date.today(),
        max_value=date.today()
    )
with col6:
    due_in_date = st.date_input(
        "Due Date (when payment is expected)",
        value=date.today() + timedelta(days=30),
        min_value=posting_date
    )

due_gap_preview = (due_in_date - posting_date).days
st.caption(f"Due gap (due date − posting date): **{due_gap_preview} days**")

st.divider()

# ── Validate & Predict ────────────────────────
if not cust_number or not name_customer or not business_code:
    st.warning("Please fill in customer number, customer name, and business code.")
    st.stop()

if st.button("Predict Payment Date", type="primary", use_container_width=True):
    with st.spinner("Running prediction..."):
        try:
            X_input = build_feature_row(
                business_code    = business_code,
                cust_number      = str(cust_number),
                name_customer    = str(name_customer),
                buisness_year    = buisness_year,
                posting_date     = posting_date,
                due_in_date      = due_in_date,
                invoice_currency = invoice_currency,
                total_open_amount= total_open_amount,
                cust_payment_terms= cust_payment_terms
            )

            X_input = X_input[selected_features]
           
            # Predict
            predicted_days = float(model.predict(X_input)[0])
            predicted_days_rounded = max(0, round(predicted_days))

            # Calculate expected payment date
            expected_payment_date = pd.Timestamp(posting_date) + pd.Timedelta(days=predicted_days_rounded)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # ── Results ───────────────────────────────
    st.divider()
    st.subheader("Prediction Result")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted Days to Clear", f"{predicted_days_rounded} days")
    col_b.metric("Posting Date",            posting_date.strftime("%d %b %Y"))
    col_c.metric("Expected Payment Date",   expected_payment_date.strftime("%d %b %Y"))

    # Payment status relative to due date
    days_vs_due = predicted_days_rounded - due_gap_preview
    if days_vs_due <= 0:
        st.success(
            f"Payment is expected **{abs(days_vs_due)} day(s) before** the due date "
            f"({due_in_date.strftime('%d %b %Y')}). On-time payment likely."
        )
    elif days_vs_due <= 7:
        st.warning(
            f"Payment is expected **{days_vs_due} day(s) after** the due date "
            f"({due_in_date.strftime('%d %b %Y')}). Slight delay possible."
        )
    else:
        st.error(
            f"Payment is expected **{days_vs_due} day(s) after** the due date "
            f"({due_in_date.strftime('%d %b %Y')}). Significant delay predicted."
        )

    
    st.caption(
        "Note: If a new customer number or name is entered the model uses the median encoded value as a fallback."
        "Predictions for new customers may be less accurate."
    )
