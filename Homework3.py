import math
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing


def randomNumberList(number):
    """

    :param number: number of random variables
    :return:  list of num random variables in the range [0,1]
    """
    list = []
    for i in range(number):
        list.append(random.uniform(0, 1))
    return list


def function(x):
    """

    :param x: x coordinate
    :return: value of function of x = y
    """
    return (math.sin(6 * x) / 6) + 0.6


def classificators(X_train, X_test, y_train, y_test, max_iter):
    # Feed Forward NN
    if max_iter != 0:
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1, max_iter=max_iter)
    else:
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    # print("Confusion matrix FFN(5,2):", confusion_matrix(y_test, y_pred))
    print("Accuracy FFN(5,2):", accuracy_score(y_test, y_pred))
    if max_iter != 0:
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(6, 3, 2), random_state=1, max_iter=max_iter)
    else:
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(6, 3, 2), random_state=1)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    # print("Confusion matrix FFN(6,3,2):", confusion_matrix(y_test, y_pred))
    print("Accuracy FFN(6,3,2):", accuracy_score(y_test, y_pred))
    if max_iter != 0:
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 4, 2), random_state=1, max_iter=max_iter)
    else:
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(5, 4, 2), random_state=1)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    # print("Confusion matrix FFN(6,3,2):", confusion_matrix(y_test, y_pred))
    print("Accuracy FFN(5,4,2):", accuracy_score(y_test, y_pred))
    # SVM linear
    svc_linear = SVC(kernel='linear', C=1)
    svc_linear.fit(X_train, y_train)
    y_pred = svc_linear.predict(X_test)
    # print("Confusion matrix C=1:", confusion_matrix(y_test, y_pred))
    print("Accuracy C=1:", accuracy_score(y_test, y_pred))
    svc_linear = SVC(kernel='linear', C=10)
    svc_linear.fit(X_train, y_train)
    y_pred = svc_linear.predict(X_test)
    # print("Confusion matrix C=10:", confusion_matrix(y_test, y_pred))
    print("Accuracy C=10:", accuracy_score(y_test, y_pred))
    svc_linear = SVC(kernel='linear', C=50)
    svc_linear.fit(X_train, y_train)
    y_pred = svc_linear.predict(X_test)
    # print("Confusion matrix C=50:", confusion_matrix(y_test, y_pred))
    print("Accuracy C=50:", accuracy_score(y_test, y_pred))
    # SVC gaussian
    # gamma hight- closer trainning examples influence the decision boundiry
    svc_gaussian = SVC(kernel='rbf', gamma=.01, C=1)
    svc_gaussian.fit(X_train, y_train)
    y_pred = svc_gaussian.predict(X_test)
    # print("Confusion matrix SVM gaussian gamma=0.01 C=1:", confusion_matrix(y_test, y_pred))
    print("Accuracy SVM gaussian gamma=0.01 C=1", accuracy_score(y_test, y_pred))
    svc_gaussian = SVC(kernel='rbf', gamma=.01, C=10)
    svc_gaussian.fit(X_train, y_train)
    y_pred = svc_gaussian.predict(X_test)
    # print("Confusion matrix SVM gaussian gamma=0.01 C=10:", confusion_matrix(y_test, y_pred))
    print("Accuracy SVM gaussian gamma=0.01 C=10", accuracy_score(y_test, y_pred))
    svc_gaussian = SVC(kernel='rbf', gamma=1, C=1)
    svc_gaussian.fit(X_train, y_train)
    y_pred = svc_gaussian.predict(X_test)
    # print("Confusion matrix SVM gaussian gamma=1 C=1:", confusion_matrix(y_test, y_pred))
    print("Accuracy SVM gaussian gamma=1 C=1", accuracy_score(y_test, y_pred))
    svc_gaussian = SVC(kernel='rbf', gamma=1, C=10)
    svc_gaussian.fit(X_train, y_train)
    y_pred = svc_gaussian.predict(X_test)
    # print("Confusion matrix SVM gaussian gamma=1 C=10:", confusion_matrix(y_test, y_pred))
    print("Accuracy SVM gaussian gamma=1 C=10", accuracy_score(y_test, y_pred))
    svc_gaussian = SVC(kernel='rbf', gamma=10, C=1)
    svc_gaussian.fit(X_train, y_train)
    y_pred = svc_gaussian.predict(X_test)
    # print("Confusion matrix SVM gaussian gamma=10 C=1:", confusion_matrix(y_test, y_pred))
    print("Accuracy SVM gaussian gamma=10 C=1", accuracy_score(y_test, y_pred))
    svc_gaussian = SVC(kernel='rbf', gamma=10, C=10)
    svc_gaussian.fit(X_train, y_train)
    y_pred = svc_gaussian.predict(X_test)
    # print("Confusion matrix SVM gaussian gamma=10 C=10:", confusion_matrix(y_test, y_pred))
    print("Accuracy SVM gaussian gamma=10 C=10", accuracy_score(y_test, y_pred))


# broj na primeroci za prvo i vtoro baranje
num = 1000

# PRVO BARANJE

x_list = randomNumberList(num)
y_list = randomNumberList(num)

class_list = []
# class_node =0 ako tockata e pod krivata inaku 1
for x, y in zip(x_list, y_list):
    if y < function(x):
        class_list.append(0)
    else:
        class_list.append(1)

tmp = {
    'x_value': x_list,
    'y_value': y_list,
    'class_node': class_list
}
data_set_node = pd.DataFrame(tmp, columns=['x_value', 'y_value', 'class_node'])
node_features_set = np.array(data_set_node.drop(['class_node'], axis=1))
node_class_set = np.array(data_set_node['class_node'])
X_train, X_test, y_train, y_test = train_test_split(node_features_set, node_class_set, test_size=0.5, random_state=1)
print("PRVO BARANJE")
classificators(X_train, X_test, y_train, y_test, 0)

# VTORO BARANJE

x_list = randomNumberList(num)
y_list = randomNumberList(num)
class_list = []
# 1-crno pole 0-belo pole
for x, y in zip(x_list, y_list):
    if ((0 < x <= 0.25 or 0.5 < x <= 0.75) and (0 < y <= 0.25 or 0.5 < y <= 0.75)) or (
            (0.25 < x <= 0.5 or 0.75 < x <= 1) and (0.25 < y <= 0.5 or 0.75 < y <= 1)):
        class_list.append(1)
    else:
        class_list.append(0)

tmp = {
    'x_value': x_list,
    'y_value': y_list,
    'class_chess': class_list
}
data_set_chess = pd.DataFrame(tmp, columns=['x_value', 'y_value', 'class_chess'])
chess_features_set = np.array(data_set_chess.drop(['class_chess'], axis=1))
chess_class_set = np.array(data_set_chess['class_chess'])
# podelba na mnozestvoto na training i test data
X_train, X_test, y_train, y_test = train_test_split(chess_features_set, chess_class_set, test_size=0.5, random_state=1)
print("VTORO BARANJE")
classificators(X_train, X_test, y_train, y_test, 0)

# TRETO BARANJE (iris)

column_names = ['sepalL', 'sepalW', 'petalL', 'petalW', 'class_iris']
data_set_iris = pd.read_csv("Data\iris.data", header=None, names=column_names)
iris_features_set = np.array(data_set_iris.drop(['class_iris'], axis=1))
iris_class_set = np.array(data_set_iris['class_iris'])
# podelba na mnozestvoto na princip 10-fold cross validation
kFold = KFold(10)
for train, test in kFold.split(iris_features_set):
    X_train, X_test = iris_features_set[train], iris_features_set[test]
    y_train, y_test = iris_class_set[train], iris_class_set[test]
# svc_linear
print("TRETO BARANJE")
classificators(X_train, X_test, y_train, y_test, 500)

# CETVRTO BARANJE

column_names = ['seismic', 'seismoacoustic', 'shift', 'genergy', 'gpuls', 'gdenergy', 'gdpuls', 'ghazard', 'nbumps',
                'nbumps2', 'nbumps3', 'nbumps4', 'nbumps5', 'nbumps6', 'nbumps7', 'nbumps89', 'energy', 'maxenergy',
                'class_bump']
data_set_bumps = pd.read_csv("Data\seismic-bumps", header=None, names=column_names)
# mora da se napravi pretvaranje na string vrednostite vo numericki
data_set_bumps['seismic'] = pd.to_numeric(data_set_bumps['seismic'], errors='coerce')
data_set_bumps['seismoacoustic'] = pd.to_numeric(data_set_bumps['seismoacoustic'], errors='coerce')
# data_set_bumps.apply(preprocessing.LabelEncoder().fit_transform())
data_set_bumps['shift'] = pd.to_numeric(data_set_bumps['shift'], errors='coerce')
data_set_bumps['ghazard'] = pd.to_numeric(data_set_bumps['ghazard'], errors='coerce')
data_set_bumps = data_set_bumps.replace(np.nan, 0, regex=True)

bumps_features_set = np.array(data_set_bumps.drop(['class_bump'], axis=1))
bumps_class_set = np.array(data_set_bumps['class_bump'])
# podelba na mnozestvoto na princip 10-fold cross validation
kFold = KFold(10)
for train, test in kFold.split(bumps_features_set):
    X_train, X_test = bumps_features_set[train], bumps_features_set[test]
    y_train, y_test = bumps_class_set[train], bumps_class_set[test]
# #svc_linear
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
print("CETVRTO BARANJE")
classificators(X_scaled, X_test, y_train, y_test, 500)

# PETTO BARANJE (letter recognition)

column_names = ['lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar',
                'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
data_set_letter = pd.read_csv("Data\letter-recognition", header=None, names=column_names)
letter_features_set = np.array(data_set_letter.drop(['lettr'], axis=1))
letter_class_set = np.array(data_set_letter['lettr'])
# podelba na mnozestvoto na training i test set (1/3 e test a se zdava so test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(letter_features_set, letter_class_set, test_size=0.3,
                                                    random_state=1)
print("PETTO BARANJE")
classificators(X_train, X_test, y_train, y_test, 1800)
