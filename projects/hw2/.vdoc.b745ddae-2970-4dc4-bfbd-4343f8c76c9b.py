# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("blueprinty.csv")  
df.head()
#
#
#
#
# Map numeric to label
df["customer_label"] = df["iscustomer"].map({0: "Non-Customer", 1: "Customer"})

# Create histogram
ax = sns.histplot(data=df, x="patents", hue="customer_label", multiple="dodge", 
                  bins=30, palette={"Non-Customer": "orange", "Customer": "skyblue"})

# Add title and axis labels
plt.title("Histogram of Patents by Customer Status")
plt.xlabel("Number of Patents")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Summary statistics by customer status
df.groupby("customer_label")["patents"].agg(["mean", "std", "count"]).reset_index()

#
#
#
#
#
#
#
#
#
#
#
#
#
# Region distribution
pd.crosstab(df["region"], df["iscustomer"])

# Age distribution
sns.kdeplot(data=df, x="age", hue="iscustomer", fill=True, common_norm=False, alpha=0.5)
plt.title("Density of Firm Age by Customer Status")
plt.xlabel("Age")
plt.legend(title="Customer Status", labels=["Non-Customer", "Customer"])
plt.show()
#
#
#
region_ct = pd.crosstab(df["region"], df["customer_label"], normalize="columns") * 100
region_ct.columns.name = None
region_ct.index.name = "Region"
region_ct = region_ct.reset_index()
print(region_ct.to_string(index=False))
#
#
#
#
# Crosstab of region vs customer status
region_ct = pd.crosstab(df["region"], df["iscustomer"], normalize="columns") * 100
region_ct.columns = ["Non-Customer (%)", "Customer (%)"]
region_ct

# Age summary by customer status
df.groupby("iscustomer")["age"].agg(['mean','std','min','max'])

# KDE for age by customer status
ages = np.linspace(df["age"].min(), df["age"].max(), 200)
kde_cust = gaussian_kde(df.loc[df.iscustomer==1,"age"])(ages)
kde_non = gaussian_kde(df.loc[df.iscustomer==0,"age"])(ages)

plt.plot(ages, kde_non, label="Non-Customer", color="grey")
plt.plot(ages, kde_cust, label="Customer", color="teal")
plt.fill_between(ages, kde_non, alpha=0.2, color="grey")
plt.fill_between(ages, kde_cust, alpha=0.2, color="teal")
plt.title("Age Density by Customer Status")
plt.xlabel("Firm Age")
plt.ylabel("Density")
plt.legend()
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
