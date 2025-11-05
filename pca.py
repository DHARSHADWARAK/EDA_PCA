
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1️⃣ Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['target'] = iris.target

# Save dataset (optional)
df.to_csv("iris_dataset.csv", index=False)
print("✅ Dataset saved as 'iris_dataset.csv'")
print(df.head())

# 2️⃣ Separate features and standardize
X = df.drop(columns=['target'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 4️⃣ Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

# 5️⃣ Scree Plot (Explained Variance Ratio)
plt.figure(figsize=(8,4))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.title('Scree Plot: Explained Variance Ratio by Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6️⃣ Cumulative Explained Variance Plot
plt.figure(figsize=(8,4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='orange')
plt.axhline(y=0.9, color='red', linestyle='--', label='90% threshold')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7️⃣ Determine number of components for ≥90% variance
num_components_90 = (cumulative_variance >= 0.9).argmax() + 1
print(f"✅ Minimum number of principal components to retain ≥90% variance: {num_components_90}")
