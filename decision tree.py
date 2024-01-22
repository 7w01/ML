import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv

# input data
Dtree = open(r'classification.csv', 'r')
reader = csv.reader(Dtree)

# extract tags
head = reader.__next__()

feature_list = []
label_list = []

# generate dictionary
for row in reader:
    label_list.append(row[-1])
    row_dic = {}
    for i in range(1, len(row) - 1):
        row_dic[head[i]] = row[i]
    feature_list.append(row_dic)

# convert dictionaries to matrix and str to bool
x_label = DictVectorizer()
x_data = x_label.fit_transform(feature_list).toarray()
# toarray() He can convert the cross-linked list form of a sparse matrix to a matrix form

# convert str to bool
y_label = preprocessing.LabelBinarizer()
y_data = y_label.fit_transform(label_list)

# modeling
model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(x_data, y_data)

# graph
import graphviz
dot_data = tree.export_graphviz(model,
                                out_file=None,
                                feature_names=x_label.get_feature_names_out(),
                                class_names=y_label.classes_,
                                filled=True,
                                rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('computer')

