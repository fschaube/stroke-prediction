import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from yellowbrick.classifier import ClassificationReport

import data_preparation
import component_analysis
from custom_plots import resultStroke

df = data_preparation.final_df

'''Make a first categorisation with kNN'''

# Distribution of the class attribute.
print('Number of non strokes {}'.format(resultStroke['smoking_status'][0]))
print('Number of strokes {}'.format(resultStroke['smoking_status'][1]))
print('Non stroke rate {}'.format(
    round(resultStroke['smoking_status'][0] / (resultStroke['smoking_status'][0] + resultStroke['smoking_status'][1]),
          2)))
print('Stroke rate {}'.format(
    round(resultStroke['smoking_status'][1] / (resultStroke['smoking_status'][0] + resultStroke['smoking_status'][1]),
          2)))

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(component_analysis.X, component_analysis.y, train_size=0.3, random_state=1)

# Not encoded example with oversampling.
print("Before OverSampling, counts of label '1': {}".format(sum(y_train_2 == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train_2 == 0)))

# Use SMOTE to oversample the under represented target variable.
X_train_result, y_train_result = SMOTE(random_state=1).fit_resample(X_train_2, y_train_2.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(y_train_result == 1)))
print("After OverSampling, counts of label '0': {} \n".format(sum(y_train_result == 0)))

colors = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

# Function to search the optimal k with the smallest error.
def searchOptimalK(X_train, X_test, y_train, y_test, scaling, info, fileName):
    if scaling:
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
    error = []
    for i in range(1, 40):
        kNN = KNeighborsClassifier(n_neighbors=i)
        kNN.fit(X_train, y_train)
        y_pred = kNN.predict(X_test)
        error.append(np.mean(y_pred != y_test))
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 40), error, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Find optimal k for kNN ' + info)
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
    plt.close()

# Function to use the optimal k created with GridSearch.
def optimalK(X_train, X_test, y_train, y_test, scaling, info, fileName, optimalK):
    if scaling:
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
    kNN = KNeighborsClassifier(n_neighbors=optimalK)
    kNN.fit(X_train, y_train)
    visualizer = ClassificationReport(kNN, cmap=colors.pop(0),
                                        classes=['no stroke', 'stroke'],
                                        support=True,
                                        title= info + " k=" + str(optimalK))
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath="report_" + fileName)
    plt.gcf().clear()

# Use the specific optimalK from GridSearch (see below) in case of not specific k-Graph.
searchOptimalK(X_train_result, component_analysis.X_test, y_train_result, component_analysis.y_test, False,
             '(with oversampling & no standardization)',
            'kNN-oversampling-no-standardization.png')
optimalK(X_train_result, component_analysis.X_test, y_train_result, component_analysis.y_test, False,
             '(with oversampling & no standardization)',
            'kNN-oversampling-no-standardization.png', 2)

searchOptimalK(X_train_result, component_analysis.X_test, y_train_result, component_analysis.y_test, True,
             '(with oversampling & standardization)',
             'kNN-oversampling-standardization.png')
optimalK(X_train_result, component_analysis.X_test, y_train_result, component_analysis.y_test, True,
             '(with oversampling & standardization)',
             'kNN-oversampling-standardization.png', 2)

searchOptimalK(component_analysis.X_train, component_analysis.X_test, component_analysis.y_train, component_analysis.y_test, False, '(no oversampling & no standardization)',
             'kNN-no-oversampling-no-standardization.png')
optimalK(component_analysis.X_train, component_analysis.X_test, component_analysis.y_train, component_analysis.y_test, False, '(no oversampling & no standardization)',
             'kNN-no-oversampling-no-standardization.png', 39)

searchOptimalK(component_analysis.X_train, component_analysis.X_test, component_analysis.y_train, component_analysis.y_test, True, '(no oversampling & standardization)',
             'kNN-no-oversampling-standardization.png')
optimalK(component_analysis.X_train, component_analysis.X_test, component_analysis.y_train, component_analysis.y_test, True, '(no oversampling & standardization)',
             'kNN-no-oversampling-standardization.png', 31)

# use GridSearch to get the best k alternatively.
gridsearch1 = GridSearchCV(estimator=KNeighborsRegressor(),
                          param_grid={'n_neighbors': range(1, 40),
                                      'weights': ['distance'],
                                      'metric': ['euclidean']})
gridsearch2 = GridSearchCV(estimator=KNeighborsRegressor(),
                          param_grid={'n_neighbors': range(1, 40),
                                      'weights': ['distance'],
                                      'metric': ['euclidean']})
gridsearch3 = GridSearchCV(estimator=KNeighborsRegressor(),
                          param_grid={'n_neighbors': range(1, 40),
                                      'weights': ['distance'],
                                      'metric': ['euclidean']})
gridsearch4 = GridSearchCV(estimator=KNeighborsRegressor(),
                          param_grid={'n_neighbors': range(1, 40),
                                      'weights': ['distance'],
                                      'metric': ['euclidean']})

result1 = gridsearch1.fit(X_train_result, y_train_result)

s1 = StandardScaler()
s2 = StandardScaler()
s3 = StandardScaler()

X_train_result = s1.fit_transform(X_train_result)
X_test = s1.transform(component_analysis.X_test)
result2 = gridsearch2.fit(X_train_result, y_train_result)
result3 = gridsearch3.fit(component_analysis.X_train, component_analysis.y_train)
X_train_3 = s3.fit_transform(component_analysis.X_train)
X_test_3 = s3.transform(X_test)

result4 = gridsearch4.fit(X_train_3, component_analysis.y_train)
print("GridSearchCV best params:")
print(result1.best_params_)
print("Scaled best params:")
print(result2.best_params_)
print("Regular best params:")
print(result3.best_params_)
print(result4.best_params_)
