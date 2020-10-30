# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("ulabox.csv")

#dfp = df[["Annual_Income_(k$)", "Spending_Score"]]
kmeans = KMeans(n_clusters=3).fit(df[["weekday", "Baby%"]])
sns.scatterplot(data=df, x="weekday", y= "Baby%", hue=kmeans.labels_)
plt.show()

# %%
