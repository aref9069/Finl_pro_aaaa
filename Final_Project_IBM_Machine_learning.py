import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline


#Getting data
!wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv

#Load Data From CSV File
df = pd.read_csv('loan_train.csv')
df.head()


#Convert to date time object
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()

#Data visualization and pre-processing
df['loan_status'].value_counts()

# notice: installing seaborn might takes a few minutes
!conda install -c anaconda seaborn -y





import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


#Pre-processing: Feature selection/extraction
#Lets look at the day of the week people get the loan

df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


#Convert Categorical features to numerical values
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()

#One Hot Encoding
df.groupby(['education'])['loan_status'].value_counts(normalize=True)

#Feature befor One Hot Encoding
df[['Principal','terms','age','Gender','education']].head()

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

#Feature selection
#Lets defind feature sets, X:
X = Feature
X[0:5]

y = pd.get_dummies(df['loan_status'])['PAIDOFF'].values
y[0:5]

#Normalize Data

X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]





###Classification
##K Nearest Neighbor(KNN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)
X_train_normalised = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
X_test_normalised = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))



from sklearn import metrics

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train_normalised,y_train)
    yhat=neigh.predict(X_test_normalised)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc



from sklearn import metrics
k = 11
accuracy = {}
knn = None
for i in range(1, k, 2):
    
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    res = knn.predict(X_test)
    accuracy.update({i:metrics.accuracy_score(y_test, res)})

accuracy




import operator
sorted_dict = sorted(accuracy.items(), key = operator.itemgetter(1), reverse=True )
print("With K = {}, we will get highest accuracy of {}".format(sorted_dict[0][0], sorted_dict[0][1]))




##Decision Tree
#Using Gini method to find nodes

from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)
X_train_normalised = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
X_test_normalised = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))

max_depth = 7
depth_acc = {}
for i in range(1, max_depth):
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=i)
    dtc = dtc.fit(X_train_normalised, y_train)
    y_pred = dtc.predict(X_test_normalised)
    depth_acc.update({i:metrics.accuracy_score(y_test, y_pred)})
    
    
    
depth_acc = sorted(depth_acc.items(), key = operator.itemgetter(1), reverse = True)
print("Accuracy with criterion as gini: {}".format(depth_acc[0][1]))


#Using Entropy to find Nodes
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)
X_train_normalised = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
X_test_normalised = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))

max_depth = 7
depth_acc = {}
for i in range(1, max_depth):
    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=i)
    dtc = dtc.fit(X_train_normalised, y_train)
    y_pred = dtc.predict(X_test_normalised)
    depth_acc.update({i:metrics.accuracy_score(y_test, y_pred)})
    
    
 
depth_acc = sorted(depth_acc.items(), key = operator.itemgetter(1), reverse = True)
print("Accuracy with criterion as entropy: {}".format(depth_acc[0][1]))
    
    
    
    
    

##Support Vector Machine


from sklearn import svm
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 4)
X_train_normalised = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
X_test_normalised = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
li = ['linear', 'poly', 'rbf', 'sigmoid']
kernals_accuracy = {}
for v in li:
    svm_model = svm.SVC(kernel=v)
    svm_model = svm_model.fit(X_train_normalised, y_train)
    y_pred = svm_model.predict(X_test_normalised)
    kernals_accuracy.update({v:metrics.accuracy_score(y_test, y_pred)})

kernals_accuracy = sorted(kernals_accuracy.items(), key = lambda x: x[1], reverse = True)
print("With '{}' as kernal, we will get highest accuracy of {}".format(kernals_accuracy[0][0], kernals_accuracy[0][1]))





##Logistic Regression

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)
X_train_normalised = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
X_test_normalised = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
li = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
solvers_accuracy = {}
for v in li:
    lr = LogisticRegression(C = 0.01, solver=v)
    lr = lr.fit(X_train_normalised, y_train)
    pred = lr.predict(X_test_normalised)
    
    solvers_accuracy.update({v:metrics.accuracy_score(y_test, pred)})

solvers_accuracy




solvers_accuracy = sorted(solvers_accuracy.items(), key = lambda x: x[1], reverse = True)
print("With '{}' as solver, we will get highest accuracy of {}".format(solvers_accuracy[0][0], solvers_accuracy[0][1]))




##Model Evaluation using Test set

from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss




#First, download and load the test se
!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv

    
#Load Test set for evaluation
test_df = pd.read_csv('loan_test.csv')
test_df.head()



test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek

test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3) else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1], inplace=True)

Feature = test_df[['Principal','terms','age','Gender','weekend']]

Feature = pd.concat([Feature,pd.get_dummies(test_df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1, inplace=True)
Feature.head()

X = Feature
X = preprocessing.StandardScaler().fit(X).transform(X)
y = test_df['loan_status'].values
type(y)



test_df['loan_status'].replace(to_replace=['PAIDOFF', 'COLLECTION'], value=[0,1], inplace =True)




y = test_df['loan_status'].values
y[:10]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)
X_train_normalised = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
X_test_normalised = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))





knn = KNeighborsClassifier(n_neighbors=3).fit(X_train_normalised, y_train)
dtc = DecisionTreeClassifier(criterion='gini', max_depth=6).fit(X_train_normalised, y_train)
svm_model = svm.SVC(kernel='rbf').fit(X_train_normalised, y_train)
lr = LogisticRegression(C = 0.01, solver='newton-cg').fit(X_train_normalised, y_train)

evaluation = [metrics.jaccard_score(y_test, knn.predict(X_test_normalised), pos_label=0),metrics.jaccard_score(y_test, dtc.predict(X_test_normalised), pos_label=0),metrics.jaccard_score(y_test, svm_model.predict(X_test_normalised), pos_label=0),metrics.jaccard_score(y_test, lr.predict(X_test_normalised), pos_label=0)]

dict = {'Jaccard':list(map(lambda x: round(x, 2), evaluation))}
evaluation = [ metrics.f1_score(y_test, knn.predict(X_test_normalised), average='weighted'), metrics.f1_score(y_test, dtc.predict(X_test_normalised), average='weighted'), metrics.f1_score(y_test, svm_model.predict(X_test_normalised), average='weighted'), metrics.f1_score(y_test, lr.predict(X_test_normalised), average='weighted')]
dict.update({'F1-score': list(map(lambda x: round(x, 2), evaluation))})
evaluation = ['','','',round(metrics.log_loss(y_test, lr.predict(X_test_normalised)), 2)]
dict.update({'LogLoss': evaluation})
repo = pd.DataFrame(dict)
repo['LogLoss'].replace('', np.nan, inplace =True)
repo.set_index([pd.Index(['KNN', 'Decision Tree', 'SVM', 'LogisticRegression'])], inplace =True)
repo.columns.name = 'Algorithm'
repo




