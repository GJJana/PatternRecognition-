from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing
import math
import numpy as np
import random
import  pandas as pd


def randomNumberList(number):
    """

    :param number: number of random variables
    :return:  list of num random variables in the range [0,1]
    """
    list=[]
    for i in range(number):
        list.append(random.uniform(0,1))
    return list


def function(x):
    """

    :param x: x coordinate
    :return: value of function of x = y
    """
    return (math.sin(6*x)/6)+0.6

#broj na primeroci za prvo i vtoro baranje
num=1000

#PRVO BARANJE

x_list=randomNumberList(num)
y_list=randomNumberList(num)

class_list=[]
#class_node =0 ako tockata e pod krivata inaku 1
for x,y in zip(x_list,y_list):
    if y<function(x):
        class_list.append(0)
    else:
        class_list.append(1)

tmp={
    'x_value' : x_list,
    'y_value' : y_list,
    'class_node' : class_list
}
data_set_node=pd.DataFrame(tmp,columns=['x_value','y_value','class_node'])
node_features_set=np.array(data_set_node.drop(['class_node'],axis=1))
node_class_set=np.array(data_set_node['class_node'])
X_train, X_test, y_train, y_test = train_test_split(node_features_set,node_class_set, test_size=0.5, random_state=1)
#clasifikatori
#DecisionTree
classifier_tree = DecisionTreeClassifier()
#train
classifier_tree= classifier_tree.fit(X_train,y_train)
#predisction on test set
y_pred = classifier_tree.predict(X_test)
print("Accuracy of dots under/over curve decision tree:",classifier_tree.score(X_test,y_test))
#KNN
#3 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=3)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy of dots under/over curve KNN with 3 neighbors :",classifier_KNN.score(X_test,y_test))
#5 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy of dots under/over curve KNN with 5 neighbors :",classifier_KNN.score(X_test,y_test))
#7 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=7)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy of dots under/over curve KNN with 7 neighbors :",classifier_KNN.score(X_test,y_test))
print()

#VTORO BARANJE (chess board)

x_list=randomNumberList(num)
y_list=randomNumberList(num)
class_list=[]
#1-crno pole 0-belo pole
for x,y in zip(x_list,y_list):
    if ((0<x<=0.25 or 0.5<x<=0.75) and (0<y<=0.25 or 0.5<y<=0.75)) or ((0.25<x<=0.5 or 0.75<x<=1) and (0.25<y<=0.5 or 0.75<y<=1)):
        class_list.append(1)
    else:
        class_list.append(0)


tmp={
    'x_value' : x_list,
    'y_value' : y_list,
    'class_chess' : class_list
}
data_set_chess=pd.DataFrame(tmp,columns=['x_value','y_value','class_chess'])
chess_features_set=np.array(data_set_chess.drop(['class_chess'],axis=1))
chess_class_set=np.array(data_set_chess['class_chess'])
#podelba na mnozestvoto na training i test data
X_train, X_test, y_train, y_test = train_test_split(chess_features_set,chess_class_set, test_size=0.5, random_state=1)
#klasifikatori
#DecisionTree
classifier = DecisionTreeClassifier()
#train
classifier= classifier.fit(X_train,y_train)
#predisction on test set
y_pred = classifier.predict(X_test)
print("Accuracy of dots in chess board color field decision tree:",classifier.score(X_test,y_test))
#KNN
#3 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=3)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy of dots in chess board color field KNN with 3 neighbors :",classifier_KNN.score(X_test,y_test))
#5 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy of dots in chess board color field  KNN with 5 neighbors :",classifier_KNN.score(X_test,y_test))
#7 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=7)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy of dots in chess board color field KNN with 7 neighbors :",classifier_KNN.score(X_test,y_test))
print()





#TRETO BARANJE (iris)

column_names = ['sepalL','sepalW','petalL','petalW','class_iris']
data_set_iris = pd.read_csv("Data\iris.data", header=None, names=column_names)
iris_features_set=np.array(data_set_iris.drop(['class_iris'], axis=1))
iris_class_set=np.array(data_set_iris['class_iris'])
#podelba na mnozestvoto na princip 10-fold cross validation
kFold=KFold(10)
for train,test in kFold.split(iris_features_set):
    X_train,X_test=iris_features_set[train],iris_features_set[test]
    y_train,y_test=iris_class_set[train],iris_class_set[test]
#klasifikatori
#DecisionTree
classifier = DecisionTreeClassifier()
#train
classifier= classifier.fit(X_train,y_train)
#predisction on test set
y_pred = classifier.predict(X_test)
print("Accuracy iris decision tree :",classifier.score(X_test,y_test))
#DecisionTree
classifier = DecisionTreeClassifier(min_impurity_decrease=0.00008)
#train
classifier= classifier.fit(X_train,y_train)
#predisction on test set
y_pred = classifier.predict(X_test)
print("Accuracy iris decision tree 0.00008:",classifier.score(X_test,y_test))
#DecisionTree
classifier = DecisionTreeClassifier(min_impurity_decrease=0.0007)
#train
classifier= classifier.fit(X_train,y_train)
#predisction on test set
y_pred = classifier.predict(X_test)
print("Accuracy iris decision tree 0.0007:",classifier.score(X_test,y_test))

#KNN
#3 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=3)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy iris KNN with 3 neighbors :",classifier_KNN.score(X_test,y_test))
#5 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy iris  KNN with 5 neighbors :",classifier_KNN.score(X_test,y_test))
#7 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=7)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy iris KNN with 7 neighbors :",classifier_KNN.score(X_test,y_test))
print()


#CETVRTO BARANJE(
column_names = ['seismic','seismoacoustic','shift','genergy','gpuls','gdenergy','gdpuls','ghazard','nbumps','nbumps2','nbumps3','nbumps4','nbumps5','nbumps6','nbumps7','nbumps89','energy','maxenergy','class_bump']
data_set_bumps = pd.read_csv("Data\seismic-bumps", header=None, names=column_names)
#mora da se napravi pretvaranje na string vrednostite vo numericki
data_set_bumps['seismic']=pd.to_numeric(data_set_bumps['seismic'],errors='coerce')
data_set_bumps['seismoacoustic']=pd.to_numeric(data_set_bumps['seismoacoustic'],errors='coerce')
#data_set_bumps.apply(preprocessing.LabelEncoder().fit_transform())
data_set_bumps['shift']=pd.to_numeric(data_set_bumps['shift'],errors='coerce')
data_set_bumps['ghazard']=pd.to_numeric(data_set_bumps['ghazard'],errors='coerce')
data_set_bumps = data_set_bumps.replace(np.nan, 0, regex=True)

bumps_features_set=np.array(data_set_bumps.drop(['class_bump'], axis=1))
bumps_class_set=np.array(data_set_bumps['class_bump'])
#podelba na mnozestvoto na princip 10-fold cross validation
kFold=KFold(10)
for train,test in kFold.split(bumps_features_set):
    X_train,X_test=bumps_features_set[train],bumps_features_set[test]
    y_train,y_test=bumps_class_set[train],bumps_class_set[test]
#klasifikatori
#DecisionTree
classifier = DecisionTreeClassifier()
#train
classifier= classifier.fit(X_train,y_train)
#predisction on test set
y_pred = classifier.predict(X_test)
print("Accuracy seismic bump decision tree:",classifier.score(X_test,y_test))
classifier = DecisionTreeClassifier(min_impurity_decrease=0.000008)
#train
classifier= classifier.fit(X_train,y_train)
#predisction on test set
y_pred = classifier.predict(X_test)
print("Accuracy seismic bump decision tree 0.00008:",classifier.score(X_test,y_test))

#KNN
#3 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=3)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy seismic bump KNN with 3 neighbors :",classifier_KNN.score(X_test,y_test))
#5 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy seismic bump KNN with 5 neighbors :",classifier_KNN.score(X_test,y_test))
#7 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=7)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy seismic bump KNN with 7 neighbors :",classifier_KNN.score(X_test,y_test))
print()


#PETTO BARANJE (letter recognition)

column_names= ['lettr','x-box','y-box','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx']
data_set_letter= pd.read_csv("Data\letter-recognition", header=None, names=column_names)
letter_features_set=np.array(data_set_letter.drop(['lettr'],axis=1))
letter_class_set=np.array(data_set_letter['lettr'])
#podelba na mnozestvoto na training i test set (1/3 e test a se zdava so test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(letter_features_set,letter_class_set, test_size=0.3, random_state=1)
#klasifikatori
#DecisionTree
classifier = DecisionTreeClassifier()
#train
classifier= classifier.fit(X_train,y_train)
#predisction on test set
y_pred = classifier.predict(X_test)
print("Accuracy letter recognition decision tree:",classifier.score(X_test,y_test))
#DecisionTree
#eksperiment kade e pomestena granicata za pruning,
classifier = DecisionTreeClassifier(min_impurity_decrease=0.000009)
#train
classifier= classifier.fit(X_train,y_train)
#predisction on test set
y_pred = classifier.predict(X_test)
print("Accuracy letter recognition decision tree 0.000009:",classifier.score(X_test,y_test))
#DecisionTree
#eksperiment kade e pomestena granicata za pruning,
classifier = DecisionTreeClassifier(min_impurity_decrease=0.00000006)
#train
classifier= classifier.fit(X_train,y_train)
#predisction on test set
y_pred = classifier.predict(X_test)
print("Accuracy letter recognition decision tree 0.00000006:",classifier.score(X_test,y_test))

#KNN
#3 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=3)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy letter recognition KNN with 3 neighbors :",classifier_KNN.score(X_test,y_test))
#5 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy letter recognition KNN with 5 neighbors :",classifier_KNN.score(X_test,y_test))
#7 neighbors
classifier_KNN=KNeighborsClassifier(n_neighbors=7)
classifier_KNN.fit(X_train,y_train)
y_pred=classifier_KNN.predict(X_test)
print("Accuracy letter recognition KNN with 7 neighbors :",classifier_KNN.score(X_test,y_test))
print()


