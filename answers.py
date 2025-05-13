# Data Analysis and Visualization with the Iris Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set styling for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Task 1: Load and Explore the Dataset
print("Task 1: Loading and Exploring the Dataset")
print("-"*50)

# Load the Iris dataset from sklearn
try:
    iris = load_iris()
    # Create a pandas DataFrame
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Add the target column
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    
# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(iris_df.head())

# Explore the structure
print("\nDataset Information:")
print(f"Shape: {iris_df.shape} (rows, columns)")
print("\nData Types:")
print(iris_df.dtypes)

# Check for missing values
print("\nMissing Values:")
missing_values = iris_df.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    print("\nCleaning the dataset by dropping rows with missing values...")
    iris_df = iris_df.dropna()
    print(f"New shape after cleaning: {iris_df.shape}")
else:
    print("\nNo missing values found. The dataset is clean!")

# Task 2: Basic Data Analysis
print("\n\nTask 2: Basic Data Analysis")
print("-"*50)

# Compute basic statistics
print("Basic Statistics:")
print(iris_df.describe())

# Perform grouping by species
print("\nAverage measurements by species:")
species_means = iris_df.groupby('species').mean()
print(species_means)

# Look for interesting patterns
print("\nInteresting findings:")
print("1. The average sepal length varies significantly across species")
print("2. Iris-setosa has the smallest petal length and width")
print("3. Iris-virginica has the largest measurements overall")

# Task 3: Data Visualization
print("\n\nTask 3: Data Visualization")
print("-"*50)

# Create a figure with subplots
plt.figure(figsize=(20, 15))

# 1. Line Chart - Average measurements by species
plt.subplot(2, 2, 1)
species_means.T.plot(marker='o', linestyle='-', ax=plt.gca())
plt.title('Average Measurements by Species', fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Measurement (cm)', fontsize=12)
plt.grid(True)
plt.legend(title='Species')

# 2. Bar Chart - Comparing average petal length by species
plt.subplot(2, 2, 2)
species_means['petal length (cm)'].plot(kind='bar', color='skyblue', ax=plt.gca())
plt.title('Average Petal Length by Species', fontsize=14)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Petal Length (cm)', fontsize=12)
plt.grid(axis='y')

# 3. Histogram - Distribution of Sepal Width
plt.subplot(2, 2, 3)
sns.histplot(data=iris_df, x='sepal width (cm)', kde=True, bins=20, ax=plt.gca())
plt.title('Distribution of Sepal Width', fontsize=14)
plt.xlabel('Sepal Width (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.subplot(2, 2, 4)
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='petal length (cm)', 
                hue='species', palette='Set2', s=70, ax=plt.gca())
plt.title('Sepal Length vs Petal Length by Species', fontsize=14)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Petal Length (cm)', fontsize=12)
plt.legend(title='Species')

plt.tight_layout()
plt.show()

# Additional Analysis: Pairplot to explore relationships between all features
plt.figure(figsize=(10, 8))
pairplot = sns.pairplot(iris_df, hue='species', height=2.5, markers=["o", "s", "D"])
pairplot.fig.suptitle('Pairwise Relationships in Iris Dataset', y=1.02, fontsize=16)
plt.show()

# Correlation Analysis
plt.figure(figsize=(10, 8))
numeric_iris = iris_df.drop('species', axis=1)
correlation = numeric_iris.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Iris Features', fontsize=16)
plt.tight_layout()
plt.show()

print("\nAnalysis Complete!")
print("Key Insights:")
print("1. Clear separation between Iris-setosa and the other two species")
print("2. Iris-versicolor and Iris-virginica have some overlap but are generally distinguishable")
print("3. Strong correlation between petal length and petal width (0.96)")
print("4. The three species form distinct clusters in the scatter plots")
