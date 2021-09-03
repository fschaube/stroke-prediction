from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from yellowbrick.classifier import ClassificationReport
import component_analysis
from component_analysis import X_train, y_train, X_test, y_test

'''Create different classifications with different classifiers.'''

ss = StandardScaler()
X_train_scale = ss.fit_transform(component_analysis.X_train)
X_test_scale = ss.transform(X_test)

clf = tree.DecisionTreeClassifier(max_leaf_nodes=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
visualizer = ClassificationReport(clf, cmap='Oranges',
title="DecisionTreeClassifier without standardization", classes=['no stroke', 'stroke'], support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show(outpath="report_DecisionTreeClassifier-no_standardization")
plt.gcf().clear()

# Decision tree (with standardization).
clf_standard = tree.DecisionTreeClassifier(max_leaf_nodes=5)
clf_standard.fit(X_train_scale, y_train)
y_pred = clf.predict(X_test_scale)
visualizer = ClassificationReport(clf_standard, cmap='Oranges', classes=['no stroke', 'stroke'],
                                  support=True, title="DecisionTreeClassifier with standardization")
visualizer.fit(X_train_scale, y_train)
visualizer.score(X_test_scale, y_test)
visualizer.show(outpath="report_DecisionTreeClassifier-standardization")
plt.gcf().clear()

# RandomForest.
rfc = RandomForestClassifier(random_state=12)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
visualizer = ClassificationReport(rfc, cmap='YlGnBu',
                                  classes=['no stroke', 'stroke'],
                                  support=True, title="RandomForestClassifier without standardization")
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show(outpath="report_RandomForestClassifier-no_standardization")
plt.gcf().clear()

# RandomForest (with standardization).
rfc = RandomForestClassifier(random_state=12)
rfc.fit(X_train_scale, y_train)
y_pred = rfc.predict(X_test_scale)
visualizer = ClassificationReport(rfc, cmap='YlGnBu', classes=['no stroke', 'stroke'],
                                  support=True, title="RandomForestClassifier with standardization")
visualizer.fit(X_train_scale, y_train)
visualizer.score(X_test_scale, y_test)
visualizer.show(outpath="report_RandomForestClassifier-standardization")
plt.gcf().clear()

# SVC classifier.
clf2 = SVC(gamma='auto')
clf2.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
visualizer = ClassificationReport(clf2, cmap='PuRd', title="SVC without standardization", classes=['no stroke', 'stroke'],
                                  support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show(outpath="report-SVC-no_standardization")
plt.gcf().clear()

# SVC classifier (with standardization).
clf2 = SVC(gamma='auto')
clf2.fit(X_train_scale, y_train)
y_pred = clf.predict(X_test_scale)
visualizer = ClassificationReport(clf2, cmap='PuRd', classes=['no stroke', 'stroke'],
                                  support=True, title="SVC with standardization")
visualizer.fit(X_train_scale, y_train)
visualizer.score(X_test_scale, y_test)
visualizer.show(outpath="report-SVC-standardization")
plt.gcf().clear()
