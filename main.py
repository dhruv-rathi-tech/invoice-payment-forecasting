import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error, f1_score

df = pd.read_csv("dataset.csv")

df.drop_duplicates(inplace=True)
df.drop(columns=["area_business", "isOpen"], inplace=True)
df.drop(columns=["document type", "posting_id", "doc_id", "invoice_id"], inplace=True)
df.dropna(axis=0, inplace=True)
#print(df.info)
#print(df.head)
#print(df.tail)
#print(df.columns)
#print(df.duplicated().sum())
#print(df.isnull().sum())
#print(df.describe())
#print(df.dtypes)
#print(df.nunique())
#print(df.shape)


sns.countplot(x=df["buisness_year"])
plt.show()

sns.countplot(x=df["invoice_currency"])
plt.show()

sns.histplot(df['total_open_amount'], bins=50, kde=True)
plt.show()

sns.boxplot(x=df["total_open_amount"])
plt.show()

topc = df["name_customer"].value_counts().head(10)
sns.barplot(x=topc.index, y=topc.values)
plt.show()



date_cols = ["clear_date", "posting_date"]
for i in date_cols:
    df[i] = pd.to_datetime(df[i])

df["days"] = (df["clear_date"] - df["posting_date"]).dt.days
#print(df["days"])

y = df["days"]
print(df.dtypes)