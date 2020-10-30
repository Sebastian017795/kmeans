import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("ulabox.csv")
baby = df[df["Baby%"] > 0]
dfp = baby[["weekday", "Baby%"]]

ssd = []
ks = range(1,11)
for k in range(1,11):
    km = KMeans(n_clusters=k)
    km = km.fit(dfp)
    ssd.append(km.inertia_)

kneedle = KneeLocator(ks, ssd, S=1.0, curve="convex", direction="decreasing")
kneedle.plot_knee()
plt.show()

k = round(kneedle.knee)

print(f"Number of clusters suggested by knee method: {k}")

kmeans = KMeans(n_clusters=k).fit(baby[["weekday", "Baby%"]])
sns.scatterplot(data=baby, x="weekday", y=" Baby%",  hue=kmeans.labels_)
plt.show()