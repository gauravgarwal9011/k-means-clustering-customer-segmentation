import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the customer data
data = pd.read_csv('Data\Mall_Customers.csv')

# Preprocessing: Remove any missing values and scale the data
data = data.dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Age', 'Spending Score']])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Age'], y=data['Spending Score'], hue=data['Cluster'], palette='viridis', s=100)
plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.show()

# Save the clustered data
data.to_csv('data/segmented_customer_data.csv', index=False)