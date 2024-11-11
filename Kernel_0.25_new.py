import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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

num_columns = len(column_names)
mutual_info_matrix = pd.DataFrame(np.zeros((num_columns, num_columns)), columns=column_names, index=column_names)

def kde_mutual_information(X, Y, bandwidth=0.25):
    XY = np.vstack([X, Y]).T
    kde_X = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X[:, np.newaxis])
    kde_Y = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(Y[:, np.newaxis])
    kde_XY = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(XY)

    log_p_X = kde_X.score_samples(X[:, np.newaxis])
    log_p_Y = kde_Y.score_samples(Y[:, np.newaxis])
    log_p_XY = kde_XY.score_samples(XY)

    mutual_info = np.mean(log_p_XY - log_p_X - log_p_Y)
    return mutual_info

for i in range(num_columns):
    for j in range(i, num_columns):
        X = data.iloc[:, i].values
        Y = data.iloc[:, j].values
        mutual_info = kde_mutual_information(X, Y, bandwidth=0.25)

        mutual_info_matrix.iat[i, j] = mutual_info
        mutual_info_matrix.iat[j, i] = mutual_info



# Plot the heatmap of the mutual information matrix
plt.figure(figsize=(12, 10))
sns.heatmap(mutual_info_matrix, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mutual Information'})
plt.title('Mutual Information Heatmap with Gaussian KDE (Bandwidth=1)')
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
