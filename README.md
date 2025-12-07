# Applied-Technical-Project
The datasets generated and used during this applied project—along with the complete source code for the Intelligent Audit Assistant proof-of-concept—have been archived in a publicly accessible repository for review, reproduction, and further development.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")


path = "/Users/wangruobing/Desktop/Sample auditing data.csv"
df = pd.read_csv(path)
print("Columns:", list(df.columns))
display(df.head())
if df["Date"].dtype == "object":
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.columns = df.columns.str.strip()

    # 2 Basic descriptive statistics
print("\n=== Overall Amount Summary ===")
display(df["Amount"].describe())

print("\n=== Amount by Category ===")
display(df.groupby("Category")["Amount"].describe())

print("\n=== Amount by Department ===")
display(df.groupby("Department")["Amount"].describe())

#3 Anomaly detection using the IQR method
Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_mask = (df["Amount"] < lower_bound) | (df["Amount"] > upper_bound)
df_outliers = df[outlier_mask].copy()
df_normal = df[~outlier_mask].copy()

print(f"\nIQR lower bound: {lower_bound:.2f}, upper bound: {upper_bound:.2f}")
print(f"Detected {df_outliers.shape[0]} outlier transactions based on Amount.")

display(df_outliers.sort_values("Amount", ascending=False).head(20))

#4 Visualization 1: Overall Box plot + swarm
%matplotlib inline
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Amount"])
sns.swarmplot(x=df["Amount"], color="red", size=4, alpha=0.6)
plt.title("Outlier Detection – Amount (Overall)")
plt.xlabel("Amount")
plt.show()

#5 Visualization 2: Box plots by Category
%matplotlib inline
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Category", y="Amount")
plt.title("Amount Distribution by Category")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Amount")
plt.show()

#6 Visualization 3: Time Series + Marking Anomalies
%matplotlib inline
plt.figure(figsize=(10, 5))

# Normal points
plt.plot(df_normal["Date"], df_normal["Amount"], "o", label="Normal", alpha=0.6)

#  Abnormal points are marked in red
plt.plot(df_outliers["Date"], df_outliers["Amount"], "ro", label="Outlier", alpha=0.9)

plt.title("Amount over Time (Outliers Highlighted)")
plt.xlabel("Date")
plt.ylabel("Amount")
plt.legend()
plt.tight_layout()
plt.show()

#7 Automatically generate Top N outlier vendors by Vendor/Department
df.columns = df.columns.str.strip()
def detect_group_outliers(df, group_col, value_col="Amount", top_n=5):
    """
    Detect outliers per group (Vendor or Department) using IQR.
    Returns: summary dataframe with outlier counts and mean outlier amount.
    """  
    results = []
    groups = df.groupby(group_col)
    for name, g in groups:
        q1 = g[value_col].quantile(0.25)
        q3 = g[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr  
        outliers = g[(g[value_col] < lower) | (g[value_col] > upper)]
        results.append({
            group_col: name,
            "transaction_count": len(g),
            "outlier_count": len(outliers),
            "outlier_ratio": len(outliers) / max(len(g), 1),
            "avg_outlier_amount": outliers[value_col].mean() if len(outliers) > 0 else 0
        })
    summary = pd.DataFrame(results)    
    # Rank outlier-heavy groups
    summary = summary.sort_values("outlier_count", ascending=False).head(top_n)
    return summary
    top_vendors = detect_group_outliers(df, group_col="Vendor", value_col="Amount", top_n=5)
print("Top Outlier Vendors:")
display(top_vendors)
top_departments = detect_group_outliers(df, group_col="Department", value_col="Amount", top_n=5)
print("Top Outlier Departments:")
display(top_departments)

import pandas as pd
from openai import OpenAI
client = OpenAI(api_key='')
df = pd.read_csv("/Users/wangruobing/Desktop/Sample auditing data.csv")

print("===== Raw Data =====")
print(df)
print("\n")
def build_prompt(transaction_table):
    return f"""
You are an Audit Assistant helping with analytical review.
Your task is to analyze the transaction table and identify:

1. Unusual or high-risk transactions
2. Any anomalies (negative numbers, sudden spikes, inconsistent vendors)
3. Possible explanations or risks
4. Actionable auditor follow-up steps

Return results in this structure:
- Key anomalies
- Explanation for each
- Recommended next steps
- Summary audit memo (3–5 sentences)

Here is the transaction table:
{transaction_table}
"""

prompt = build_prompt(df.to_string())

import pandas as pd
from openai import OpenAI

client = OpenAI(api_key="")

df = pd.read_csv("/Users/wangruobing/Desktop/Sample auditing data.csv")

def build_prompt(table):
    return f"""
You are an Audit Assistant helping with analytical review.

Tasks:
1. Identify unusual or high-risk transactions  
2. Explain each anomaly  
3. Recommend follow-up audit procedures  
4. Provide a short 3–5 sentence audit memo summary  

Transaction table:
{table}
"""

prompt = build_prompt(df.to_string())

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a professional audit assistant."},
        {"role": "user", "content": prompt}
    ]
)

audit_output = response.choices[0].message.content
    print("===== AI Audit Analysis =====")
print(audit_output)

