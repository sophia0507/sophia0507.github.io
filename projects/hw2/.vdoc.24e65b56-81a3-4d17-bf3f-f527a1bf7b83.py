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

df = pd.read_csv("blueprinty.csv", index_col=0)  
df.head()
#
#
#
#
# Histogram
sns.histplot(data=df, x="patents", hue="iscustomer", multiple="dodge", bins=30)
plt.title("Histogram of Patents by Customer Status")
plt.xlabel("Number of Patents")
plt.ylabel("Count")
plt.show()

# Mean and std
df.groupby("iscustomer")["patents"].agg(["mean", "std", "count"])
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
