import pandas as pd
import prince
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import data_preparation

df = data_preparation.final_df

'''Results for PCA, MCA and FAMD'''

# Separate data in features and target.
X = df[['age', 'avg_glucose_level', 'bmi']]
y = df['stroke']

# Execute Principal Component Analysis (PCA).
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=1)
pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=2))
pca_pipeline.fit_transform(X_train, y_train)
pca = pca_pipeline.named_steps['pca']
pca_result = pd.DataFrame({'explanation_ratio': pca.explained_variance_ratio_,
                           'PCs': ['PC1', 'PC2']})
ax = sns.barplot(x='PCs', y="explanation_ratio", data=pca_result, color="green")
ax.grid(False)
ax.text(x=0, y=pca_result['explanation_ratio'][0] + 0.001, s=round(pca_result['explanation_ratio'][0], 2),
        color='black', ha="center")
ax.text(x=1, y=pca_result['explanation_ratio'][1] + 0.001, s=round(pca_result['explanation_ratio'][1], 2),
        color='black', ha="center")
plt.title('PCA Cluster')
plt.show()
plt.close()

# Multiple correspondence analysis (MCA) for non-continuous variables.
mca = prince.MCA(
    n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='auto',
    random_state=1
)

X_categorial = df.select_dtypes('object')
print("X_categorial")
print(X_categorial)
mca.fit_transform(X_categorial)
mca_result = pd.DataFrame({'explanation_ratio': mca.explained_inertia_,
                           'MCs': ['MC1', 'MC2']})
ax = sns.barplot(x='MCs', y="explanation_ratio", data=mca_result, color="red")
ax.grid(False)
ax.text(x=0, y=mca_result['explanation_ratio'][0] + 0.001, s=round(mca_result['explanation_ratio'][0], 2),
        color='black', ha="center")
ax.text(x=1, y=mca_result['explanation_ratio'][1] + 0.001, s=round(mca_result['explanation_ratio'][1], 2),
        color='black', ha="center")
plt.title('MCA Cluster')
plt.show()
plt.close()

# Starting FAMD, a combination of PCA and MCA.
famd = prince.FAMD(
    n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='auto',
    random_state=1
)
attr = df.drop('stroke', 1)
print("attr")
print(attr)
famd.fit_transform(attr)
famd_result = pd.DataFrame({'explanation_ratio': famd.explained_inertia_,
                           'FAMDCs': ['FAMDC1', 'FAMDC2']})
ax = sns.barplot(x='FAMDCs', y="explanation_ratio", data=famd_result, color="blue")
ax.grid(False)
ax.text(x=0, y=famd_result['explanation_ratio'][0] + 0.001, s=round(famd_result['explanation_ratio'][0], 2),
        color='black', ha="center")
ax.text(x=1, y=famd_result['explanation_ratio'][1] + 0.001, s=round(famd_result['explanation_ratio'][1], 2),
        color='black', ha="center")
plt.title('FAMD Cluster')
plt.show()
plt.close()
