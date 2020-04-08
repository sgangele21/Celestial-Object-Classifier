#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Skyserver_SQL2_27_2018 6_51_39 PM.csv')

df.drop(['objid', 'specobjid'], axis=1,inplace=True)
df.head()


# In[2]:


sns.countplot('class', data = df)
plt.title("Class Distribution")
plt.ylabel('Frequency')
plt.show()
df['class'].value_counts()


# In[22]:


cols=df[['ra','dec','u','g','r','i','z','run','rerun','camcol','field','redshift','plate','mjd']]

fig, ax = plt.subplots(len(cols.columns), 1, figsize=(10,30))
counter = 0
for i in cols.columns:
    ax[counter].boxplot(data=df, x=i)
    ax[counter].set_title(i+" Dist.")
    ax[counter].set_ylabel(i+' Dist.')
    counter += 1
plt.tight_layout()
plt.show()


# In[4]:


# Now let's find out the importances for each feature
model = RandomForestClassifier()
model.fit(df.drop('class', axis=1) , df['class'])
imp = model.feature_importances_
f = df.columns.drop('class')
f_sorted = f[np.argsort(imp)[::-1]]
sns.barplot(x=f,y = imp, order = f_sorted)
plt.title("Attribute Scaling")
plt.ylabel("Scale")
plt.show()


# In[5]:


f_selected = f_sorted[:7].values
df_features = df.loc[:,f_selected]
df_labels = df['class']
df_features.head()


# In[6]:


# Let's test Random Forest Tree
train_X, test_X, train_y, test_y = train_test_split(df_features, df_labels, stratify=df_labels, test_size=0.20)
forest = RandomForestClassifier(random_state=42)
tuned_parameters={'n_estimators': range(10,100,10)[1:]} # Got help for this part
clf = GridSearchCV(forest, tuned_parameters,cv=5)
train = clf.fit(train_X, train_y)
pred = clf.predict(test_X)


# In[7]:


print(metrics.classification_report(pred, test_y))
print(metrics.confusion_matrix(pred, test_y))

rfF1 = metrics.f1_score(pred, test_y, average='weighted')
print(rfF1)
sns.barplot(["Random Forest Classifier"], rfF1)
plt.title("F1 score by labels")
plt.show()


# In[8]:


from sklearn.neighbors import KNeighborsClassifier
# Let's test KNN
knn = KNeighborsClassifier()
knn.fit(train_X, train_y)
knnpred = knn.predict(test_X)

print(metrics.classification_report(knnpred, test_y))
print(metrics.confusion_matrix(knnpred, test_y))

knnF1 = metrics.f1_score(knnpred, test_y, average='weighted')
print(knnF1)
sns.barplot(["KNN Classifier"], knnF1)
plt.title("F1 score by labels")
plt.show()


# In[9]:


from sklearn.naive_bayes import GaussianNB
# Lets test Naive Bayes
gnb=GaussianNB()
gnb.fit(train_X,train_y)
gnbpred=gnb.predict(test_X)

print(metrics.classification_report(gnbpred, test_y))
print(metrics.confusion_matrix(gnbpred, test_y))

gnbF1 = metrics.f1_score(gnbpred, test_y, average='weighted')
print(gnbF1)
sns.barplot(["Naive Bayes Classifier"], gnbF1)
plt.title("F1 score by labels")
plt.show()


# In[10]:


from sklearn.svm import LinearSVC
# Lets test Linear SVC
svc = LinearSVC(penalty='l2',C=10.0,max_iter = 11000)
svc.fit(train_X,train_y)
svcpred = svc.predict(test_X)

print(metrics.classification_report(svcpred, test_y))
print(metrics.confusion_matrix(svcpred, test_y))

svcF1 = metrics.f1_score(svcpred, test_y, average='weighted')
print(svcF1)
sns.barplot(["SVC Classifier"], svcF1)
plt.title("F1 score by labels")
plt.show()


# In[11]:


from sklearn.tree import DecisionTreeClassifier
# Let's test Decision Tree
dtree = DecisionTreeClassifier(max_depth=5)
dtree.fit(train_X,train_y)
dtreepred = dtree.predict(test_X)

print(metrics.classification_report(dtreepred, test_y))
print(metrics.confusion_matrix(dtreepred, test_y))

dtreeF1 = metrics.f1_score(dtreepred, test_y, average='weighted')
print(dtreeF1)
sns.barplot(["Decision Tree Classifier"], dtreeF1)
plt.title("F1 score by labels")
plt.show()


# In[12]:


from sklearn.neural_network import MLPClassifier
# Lets test Neural Network
nn = MLPClassifier(hidden_layer_sizes = (1000,1000),max_iter = 10000)
nn.fit(train_X,train_y)
nnpred = nn.predict(test_X)

print(metrics.classification_report(nnpred, test_y))
print(metrics.confusion_matrix(nnpred, test_y))

nnF1 = metrics.f1_score(nnpred, test_y, average='weighted')
print(nnF1)
sns.barplot(["Neural Network Classifier"], nnF1)
plt.title("F1 score by labels")
plt.show()


# In[13]:


# Let's show all the results of each classifier
classifiers = ["KNN", "NN MPL", "Naive Bayes", "Linear SVC", "Decision Tree", "Rand. Forest"]
f1Results = [knnF1, nnF1, gnbF1, svcF1, dtreeF1, rfF1]

fig,ax = plt.subplots()
fig.set_size_inches(15,8)
bottom, top = ax.set_ylim(0.75, 1)
plt.title("F1 score by labels with out Sampling")
plt.bar(classifiers,f1Results)


# In[14]:


from imblearn.over_sampling import SMOTE
# Now let's oversample our minoriy class.... BUT ONLY DO THIS ON TRAINING DATA!!!!
sm = SMOTE(random_state=21)
train_X_res, train_y_res = sm.fit_sample(train_X, train_y)


# In[15]:


# Let's test random forest
forest = RandomForestClassifier(random_state=42)
tuned_parameters={'n_estimators': range(10,100,10)[1:]}
clf = GridSearchCV(forest, tuned_parameters,cv=5)

train = clf.fit(train_X_res, train_y_res)
rfpredRes = clf.predict(test_X)

print(metrics.classification_report(pred, test_y))
print(metrics.confusion_matrix(pred, test_y))

rfF1Res = metrics.f1_score(rfpredRes, test_y, average='weighted')
print(rfF1Res)
sns.barplot(["Random Forest Classifier"], rfF1Res)
plt.title("F1 score by labels with Sampling")
plt.show()


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
# Let's test KNN
knn = KNeighborsClassifier()
knn.fit(train_X_res, train_y_res)
knnpredRes = knn.predict(test_X)


print(metrics.classification_report(knnpredRes, test_y))
print(metrics.confusion_matrix(knnpredRes, test_y))

knnF1Res = metrics.f1_score(knnpredRes, test_y, average='weighted')
print(knnF1Res)
sns.barplot(["KNN Classifier"], knnF1Res)
plt.title("F1 score by labels with Sampling")
plt.show()


# In[17]:


from sklearn.naive_bayes import GaussianNB
# Let's test Naive Bayes
gnb=GaussianNB()
gnb.fit(train_X_res,train_y_res)
gnbpredRes = gnb.predict(test_X)

print(metrics.classification_report(gnbpredRes, test_y))
print(metrics.confusion_matrix(gnbpredRes, test_y))

gnbF1Res = metrics.f1_score(gnbpredRes, test_y, average='weighted')
print(gnbF1Res)
sns.barplot(["Naive Bayes Classifier"], gnbF1Res)
plt.title("F1 score by labels with Samples")
plt.show()


# In[18]:


from sklearn.svm import LinearSVC
# Let's test Linear SVC
svc = LinearSVC(penalty='l2',C=10.0,max_iter = 50000)
svc.fit(train_X_res,train_y_res)
svcpredRes = svc.predict(test_X)

print(metrics.classification_report(svcpredRes, test_y))
print(metrics.confusion_matrix(svcpredRes, test_y))

svcF1Res = metrics.f1_score(svcpredRes, test_y, average='weighted')
print(svcF1Res)
sns.barplot(["SVC Classifier"], svcF1Res)
plt.title("F1 score by labels with Sampling")
plt.show()


# In[19]:


from sklearn.tree import DecisionTreeClassifier
# Let's test Decision Trees
dtree = DecisionTreeClassifier(max_depth=5)
dtree.fit(train_X_res,train_y_res)
dtreepredRes = dtree.predict(test_X)


print(metrics.classification_report(dtreepredRes, test_y))
print(metrics.confusion_matrix(dtreepredRes, test_y))

dtreeF1Res = metrics.f1_score(dtreepredRes, test_y, average='weighted')
print(dtreeF1Res)
sns.barplot(["Decision Tree Classifier"], dtreeF1Res)
plt.title("F1 score by labels with Sampling")
plt.show()


# In[20]:


from sklearn.neural_network import MLPClassifier
# Lets test Neural Network
nn = MLPClassifier(hidden_layer_sizes = (1000,1000),max_iter = 1000)
nn.fit(train_X_res,train_y_res)
nnpredRes = nn.predict(test_X)

print(metrics.classification_report(nnpredRes, test_y))
print(metrics.confusion_matrix(nnpredRes, test_y))

nnF1Res = metrics.f1_score(nnpredRes, test_y, average='weighted')
print(nnF1Res)
sns.barplot(["Neural Network Classifier"], nnF1Res)
plt.title("F1 score by labels with Sampling")
plt.show()


# In[21]:


# Let's show all the results of each classifier
classifiers = ["KNN", "NN MPL", "Naive Bayes", "Linear SVC", "Decision Tree", "Rand. Forest"]
f1Results = [knnF1Res, nnF1Res, gnbF1Res, svcF1Res, dtreeF1Res, rfF1Res]

fig,ax = plt.subplots()
fig.set_size_inches(15,8)
bottom, top = ax.set_ylim(0.4, 1)
plt.title("F1 score by labels with Sampling")
plt.bar(classifiers,f1Results)


