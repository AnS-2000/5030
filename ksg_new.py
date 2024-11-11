import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from npeet import entropy_estimators as ent

# Load the adult dataset
file_path = r'C:\Users\Eric\Desktop\iris\adult.data'
data = pd.read_csv(file_path, header=None)


column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race",
    "sex", "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]
data.columns = column_names

for col in data.columns:
    if data[col].dtype == 'object':
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])



def ksg_mutual_information(X, Y, k=3):
    """Calculate mutual information using the KSG estimator with k-nearest neighbors."""
    return ent.mi(X, Y, k=k)


for k in range(3, 11):
    mutual_info_matrix = pd.DataFrame(np.zeros((len(column_names), len(column_names))), columns=column_names,
                                      index=column_names)

    for i in range(len(column_names)):
        for j in range(i, len(column_names)):
            if i == j:
                mutual_info_matrix.iat[i, j] = np.nan
            else:
                X = data.iloc[:, i].values
                Y = data.iloc[:, j].values
                mutual_info = ksg_mutual_information(X, Y, k=k)

                mutual_info_matrix.iat[i, j] = mutual_info
                mutual_info_matrix.iat[j, i] = mutual_info

    plt.figure(figsize=(12, 10))
    sns.heatmap(mutual_info_matrix, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mutual Information'})
    plt.title(f'Mutual Information Heatmap with KSG Estimator (k={k})')
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
