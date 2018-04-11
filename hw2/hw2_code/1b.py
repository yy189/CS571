import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
import graphviz

X = np.array([
    [.75, .10],
    [.85, .80],
    [.85, .95],
    [.15, .10],
    [.05, .25],
    [.05, .50],
    [.85, .25],
])

y = np.array([-1, -1, 1, -1, 1, 1, -1])

target_names = ['two points', 'three points']
feature_names = ['X1', 'X2']

clf = tree.DecisionTreeClassifier(splitter='random')
clf = clf.fit(X, y)

y_pred = clf.predict(X)
print(classification_report(y, y_pred, target_names = target_names))
print('\nAcuracy: {0:.4f}'.format(accuracy_score(y, y_pred)))

dot = tree.export_graphviz(clf, out_file=None, feature_names=feature_names, class_names=target_names, filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot)
graph.format = 'png'
graph.render('1b', view=True)





