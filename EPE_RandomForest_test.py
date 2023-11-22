# Data Processing
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn import tree
import csv

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')





"""
## below tutorial info comes from https://data36.com/random-forest-in-python/
# read and check data
df = pd.read_csv(r"T:\MIP\Katie_Merriman\Project2Data\tutorialData\possum.csv")
df.sample(5, random_state=44)
# drop any rows with missing data
df = df.dropna()
# remove unnecessary columns, store features and label data in separate variables
X = df.drop(["case", "site", "Pop", "sex"], axis=1)
y = df["sex"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
"""

dfTrain = pd.read_csv(r"T:\MIP\Katie_Merriman\Project2bData\SortedPatientData_thresh2_train_binary.csv")

#X_train = dfTrain[["dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY", "asymmetry",
#                 "flipDist_outsideChange", "flipDist_dist3DChange", "flipDist_distXYChange",
#                 "flipDist_var", "flipDist_outside", "flipDist_inside", "flipDist_dist3D", "flipDist_distXY"]]

#X_train = dfTrain[["area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY",
#                  "dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY",
#                  "flipArea_outsideChange", "flipArea_dist3DChange", "flipArea_distXYChange",
#                  "flipArea_var", "flipArea_outside", "flipArea_inside", "flipArea_dist3D", "flipArea_distXY",
#                  "flipDist_outsideChange", "flipDist_dist3DChange", "flipDist_distXYChange",
#                  "flipDist_var", "flipDist_outside", "flipDist_inside", "flipDist_dist3D", "flipDist_distXY"]]

#X_train = dfTrain.drop(["patient", "area_lesion", "dist_lesion", "flipArea_lesion", "flipDist_lesion", "GroundTruth_EPE", "Set"], axis=1)
X_train = dfTrain[["area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY",
                 "dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY"]]
#X_train = dfTrain[["dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY"]]

X_train.sample(5, random_state=44)
y_train = dfTrain["GroundTruth_EPE"]

dfVal = pd.read_csv(r"T:\MIP\Katie_Merriman\Project2bData\SortedPatientData_thresh2_test_binary.csv")

#X_test = dfVal[["dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY", "asymmetry",
#                 "flipDist_outsideChange", "flipDist_dist3DChange", "flipDist_distXYChange",
#                 "flipDist_var", "flipDist_outside", "flipDist_inside", "flipDist_dist3D", "flipDist_distXY"]]

#X_test = dfVal[["asymmetry", "area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY",
#                 "dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY",
#                 "flipArea_outsideChange", "flipArea_dist3DChange", "flipArea_distXYChange",
#                 "flipArea_var", "flipArea_outside", "flipArea_inside", "flipArea_dist3D", "flipArea_distXY",
#                 "flipDist_outsideChange", "flipDist_dist3DChange", "flipDist_distXYChange",
#                 "flipDist_var", "flipDist_outside", "flipDist_inside", "flipDist_dist3D", "flipDist_distXY"]]]]
#X_test = dfVal.drop(["patient", "area_lesion", "dist_lesion", "flipArea_lesion", "flipDist_lesion", "GroundTruth_EPE", "Set"], axis=1)
#X_test = dfVal[["dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY"]]
X_test = dfVal[["area_var", "area_outside", "area_inside", "area_dist3D", "area_distXY",
                 "dist_var", "dist_outside", "dist_inside", "dist_dist3D", "dist_distXY"]]
y_test = dfVal["GroundTruth_EPE"]

trial = 'EPE grade, threshold 2, dist no flip, unweighted'
from sklearn.ensemble import RandomForestClassifier
#rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", class_weight={0: 1, 1: 5}, random_state=44)
#rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", class_weight={0: 1, 1: 10, 2: 8, 3: 5}, random_state=44)
#sample_weight = np.array([5 if i == 0 else 1 for i in y])  , class_weight={0: 1, 1: 4}, class_weight={0: 2, 1: 10, 2: 8, 3: 5}
rf_model.fit(X_train, y_train)

# find out predictions generated for test set by model
predictions = rf_model.predict(X_test)
print(predictions)

file = open(r"T:\MIP\Katie_Merriman\Project2bData\binaryPredictions.csv", 'a+', newline='')
# writing the data into the file
with file:
    write = csv.writer(file)
    write.writerows([[trial], predictions])

file.close()



# look at the probabilities for each class generated by model
pred_probs = rf_model.predict_proba(X_test)
pred_classes = rf_model.classes_
#print(pred_classes) # tells which class each column refers to
#print(pred_probs)

# find out how important each feature was to prediction
importances = rf_model.feature_importances_
columns = X_train.columns
i=0
while i < len(columns):
    print(f" The importance of feature '{columns[i]}' is {round(importances[i] * 100, 2)}%.")
    i += 1


# Export the first three decision trees from the forest
'''
for i in range(3):
    export_graphviz(rf_model.estimators_[i], out_file='tree.dot',
                    feature_names=X_train.columns,
                    filled=True,
                    #max_depth=6,
                    impurity=False,
                    proportion=True)

    import pydot

    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')

    # Display in jupyter notebook
    from IPython.display import Image

    Image(filename='tree.png')

    from matplotlib import image as mpimg
    tree_image = mpimg.imread("tree.png")
    plt.imshow(tree_image)
    plt.show()

'''


## USE FOR BINARY

fpr, tpr, thresholds = roc_curve(y_test, pred_probs[:,1], pos_label=1)
roc_auc = roc_auc_score(y_test, pred_probs[:,1])
roc_auc
# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# roc curve for tpr = fpr
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
'''

## USE FOR MULTI-CLASS

y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

colors = ['blue', 'green', 'darkorange', 'red']
aucs = []

for i in range(n_classes):
  fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], pred_probs[:, i])
  plt.plot(fpr[i], tpr[i], color=colors[i], lw=2)
  print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
  aucs.append(auc(fpr[i], tpr[i]))


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curves')
plt.legend(['EPE 0 - AUC: ' + str(round(aucs[0], 3)), 'EPE 1 - AUC: ' + str(round(aucs[1], 3)), 'EPE 2 - AUC: ' +
            str(round(aucs[2], 3)), 'EPE 3 - AUC: ' + str(round(aucs[3], 3))], loc="lower right")
plt.show()

import scikitplot as skplt
skplt.metrics.plot_roc_curve(y_test, pred_probs)
plt.show()

'''
plt.pause(1)







#from sklearn.metrics import roc_curve, roc_auc_score
#roc_auc = roc_auc_score(y_test, pred_probs)

'''
# try out a new prediction
new_possum = [['age', 'hdlngth', 'skullw', 'totlngth', 'taill', 'footlgth', 'earconch', 'eye', 'chest', 'belly'],
[7.0, 83.2, 54.3, 81.0, 37.0, 70.0, 46.3, 14.7, 25.0, 32.0]]
new_pred = rf_model.predict(new_possum)
print(new_pred)
'''









'''
## below tutorial info comes from https://www.datacamp.com/tutorial/random-forests-classifier-python
bank_data['default'] = bank_data['default'].map({'no':0,'yes':1,'unknown':0})
bank_data['y'] = bank_data['y'].map({'no':0,'yes':1})

# Split the data into features (X) and target (y)
X = bank_data.drop('y', axis=1)
y = bank_data['y']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Export the first three decision trees from the forest

for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)
'''