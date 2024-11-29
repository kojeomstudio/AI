
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
from graphviz import Source

iris = load_iris(as_frame=True)

# sklearn에서 제공하는 iris 데이터에 대한 처리.
# x -> traits / y -> target
x_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(x_iris, y_iris)

export_graphviz(tree_clf, out_file='iris_tree.dot', feature_names=["꽃잎 길이 (cm)", "꽃잎 너비 (cm)"],
                class_names=iris.target_names, rounded=True, filled=True)

Source.from_file("iris_tree.dot")