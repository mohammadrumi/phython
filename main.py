import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('F:\phython\insurance.csv')

# 1. Dataset Overview
print("Dataset Overview:\n", data.head())
print("\nData Types:\n", data.dtypes)

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# 2. Balance Check
for column in ['sex', 'smoker', 'region']:
    print(f"\nValue counts for {column}:")
    print(data[column].value_counts(normalize=True) * 100)

# 3. Statistical Analysis
stats = {}
for column in data.columns:
    if data[column].dtype in ['int64', 'float64']:
        stats[column] = {
            "Mean": data[column].mean(),
            "Median": data[column].median(),
            "Variance": data[column].var(),
            "Standard Deviation": data[column].std()
        }
    else:
        frequency = data[column].value_counts()
        percentage = data[column].value_counts(normalize=True) * 100
        stats[column] = pd.DataFrame({'Frequency': frequency, 'Percentage': percentage})

# Print statistical results
for column, result in stats.items():
    print(f"\nColumn: {column}")
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"{key}: {value:.2f}")
    else:
        print(result)

# 4. Data Visualization
# Bar Chart for Categorical Variables
categorical_columns = ['sex', 'smoker', 'region']
for column in categorical_columns:
    data[column].value_counts().plot(kind='bar', title=column)
    plt.show()

# Pie Chart for Smoker vs Non-Smoker
data['smoker'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='Smoker Distribution')
plt.show()

# Correlation Heatmap
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = data[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 5. Categorical to Numerical Conversion
# One-Hot Encoding for region
encoded_data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Label Encoding for smoker and sex
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['sex'] = data['sex'].map({'male': 1, 'female': 0})

# 6. Missing Values Handling
# Since there are no missing values in this dataset, no handling is required.

# 7. Conclusion
print("\nConclusion:\n")
print("- The dataset contains both numerical and categorical data.")
print("- Some columns like 'smoker' are imbalanced, while others are relatively balanced.")
print("- Key statistics for numerical columns and frequency distributions for categorical columns have been calculated.")
print("- Data visualization includes bar charts, pie charts, and a correlation heatmap.")
print("- Categorical columns have been converted into numerical values for machine learning tasks.")
