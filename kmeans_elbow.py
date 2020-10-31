# %%
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
# %%
df = pd.read_csv("ulabox.csv")
#baby = df[df["Baby%"] > 0]
df.drop(columns = ["order","customer"])
dft = df
v=["Food%","Drinks%","Home%","Baby%","Pets%","Beauty%","Health%", "discount%"]



 
# %%
ssd = []
ks = range(1,14)
for k in range(1,14):
    km = KMeans(n_clusters=k)
    km = km.fit(dft)
    ssd.append(km.inertia_)
 
kneedle = KneeLocator(ks, ssd, S=1.0, curve="convex", direction="decreasing")
kneedle.plot_knee()
plt.show()
 
k = round(kneedle.knee)
 
print(f"Number of clusters suggested by knee method: {k}")
 
kmeans = KMeans(n_clusters=k).fit(df)
sns.scatterplot(data=df, x="weekday",  hue=kmeans.labels_)
plt.show()
 
 
# %%
ssd = []
ks = range(1,20)
for k in range(1,20):
    km = KMeans(n_clusters=k)
    km = km.fit(dft)
    ssd.append(km.inertia_)
 
kneedle = KneeLocator(ks, ssd, S=1.0, curve="convex", direction="decreasing")
kneedle.plot_knee()
plt.show()
 
k = round(kneedle.knee)
 
print(f"Number of clusters suggested by knee method: {k}")
 
kmeans = KMeans(n_clusters=k).fit(df[["weekday", "total_items"]])
sns.scatterplot(data=df, x="weekday", y=" total_items",  hue=kmeans.labels_)
plt.show()
 
 
# %%
cluster0 = df[kmeans.labels_==0]
cluster0[v].sum().plot.bar()
plt.show()


# %%
cluster1 = df[kmeans.labels_==1]
cluster1[v].sum().plot.bar()
plt.show()


# %%

cluster2 = df[kmeans.labels_==2]
cluster2[v].sum().plot.bar()
plt.show()

# %%
cluster3 = df[kmeans.labels_==3]
cluster3[v].sum().plot.bar()
plt.show()



# %%
sns.displot(data=cluster0, x="hour")
sns.displot(data=cluster1, x="hour")
sns.displot(data=cluster2, x="hour")


# %%
