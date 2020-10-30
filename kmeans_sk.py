# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
#%%
df = pd.read_csv("ulabox.csv")
baby = df[df["Baby%"] > 0]
#dfp = df[["Annual_Income_(k$)", "Spending_Score"]]
kmeans = KMeans(n_clusters=3).fit(baby[["weekday", "Baby%"]])
sns.scatterplot(data=baby, x="weekday", y= "Baby%", hue=kmeans.labels_)
plt.show()

# %%
df = pd.read_csv("ulabox.csv")
pet = df[df["Fresh%"] > 0]
#dfp = df[["Annual_Income_(k$)", "Spending_Score"]]
kmeans = KMeans(n_clusters=3).fit(pet[["weekday", "Fresh%"]])
sns.scatterplot(data=pet, x="weekday", y= "Fresh%", hue=kmeans.labels_)
plt.show()

# %%
