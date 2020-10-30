import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

df = pd.read_csv("Mall_Customers.csv")

#dfp = df[["Annual_Income_(k$)", "Spending_Score"]]
kmeans = KMeans(n_clusters=3).fit(df[["Annual_Income_(k$)", "Spending_Score"]])
sns.scatterplot(data=df, x="Annual_Income_(k$)", y="Spending_Score", hue=kmeans.labels_)
plt.show()