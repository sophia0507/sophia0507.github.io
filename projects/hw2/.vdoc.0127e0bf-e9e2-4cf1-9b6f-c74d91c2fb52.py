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
# Define a mapping from iscustomer values to readable labels
df["customer_label"] = df["iscustomer"].map({0: "Non-Customer", 1: "Customer"})

# Create histogram with labeled hue
sns.histplot(data=df, x="patents", hue="customer_label", multiple="dodge", bins=30)

# Add titles and axis labels
plt.title("Histogram of Patents by Customer Status")
plt.xlabel("Number of Patents")
plt.ylabel("Count")
plt.legend(title="Customer Status")
plt.show()
#
#
#
# Summary statistics by customer status
df.groupby("customer_label")["patents"].agg(["mean", "std", "count"])

```
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
