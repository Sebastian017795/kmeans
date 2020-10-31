# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
#%%
df = pd.read_csv("ulabox.csv")
baby = df[df["Baby%"] > 0]

kmeans = KMeans(n_clusters=3).fit(baby[["weekday", "Baby%"]])
sns.scatterplot(data=baby, x="weekday", y= "Baby%", hue=kmeans.labels_)
plt.show()

# %%
df = pd.read_csv("ulabox.csv")
fresh = df[df["Fresh%"] > 0]

kmeans = KMeans(n_clusters=3).fit(fresh[["weekday", "Fresh%"]])
sns.scatterplot(data=fresh, x="weekday", y= "Fresh%", hue=kmeans.labels_)
plt.show()

# %%
df = pd.read_csv("ulabox.csv")
total = df[df["total_items"] > 0]

kmeans = KMeans(n_clusters=3).fit(total[["weekday", "total_items"]])
sns.scatterplot(data=total, x="weekday", y= "total_items", hue=kmeans.labels_)
plt.show()


# %%
