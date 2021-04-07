import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score


def categorizer(df, type='train'):
    global category_0
    global category_2
    global category_4
    global category_10
    global category_11
    global category_12
    quantile_count = 5
    uniqueness_no = .001

    if (type == 'train'):
        category_0 = []
        category_2 = []
        category_4 = []
        category_10 = []
        category_11 = []
        category_12 = []
        for i in range(quantile_count):
            if i == 0:
                category_0.append(df.iloc[:, 0].quantile(i / (quantile_count - 1)) - 1 + i * uniqueness_no)
            else:
                category_0.append(df.iloc[:, 0].quantile(i / (quantile_count - 1)) + i * uniqueness_no)
        for i in range(quantile_count):
            if i == 0:
                category_2.append(df.iloc[:, 2].quantile(i / (quantile_count - 1)) - 1 + i * uniqueness_no)
            else:
                category_2.append(df.iloc[:, 2].quantile(i / (quantile_count - 1)) + i * uniqueness_no)
        for i in range(quantile_count):
            if i == 0:
                category_4.append(df.iloc[:, 4].quantile(i / (quantile_count - 1)) - 1 + i * uniqueness_no)
            else:
                category_4.append(df.iloc[:, 4].quantile(i / (quantile_count - 1)) + i * uniqueness_no)
        for i in range(quantile_count):
            if i == 0:
                category_10.append(df.iloc[:, 10].quantile(i / (quantile_count - 1)) - 1 + i * uniqueness_no)
            else:
                category_10.append(df.iloc[:, 10].quantile(i / (quantile_count - 1)) + i * uniqueness_no)
        for i in range(quantile_count):
            if i == 0:
                category_11.append(df.iloc[:, 11].quantile(i / (quantile_count - 1)) - 1 + i * uniqueness_no)
            else:
                category_11.append(df.iloc[:, 11].quantile(i / (quantile_count - 1)) + i * uniqueness_no)
        for i in range(quantile_count):
            if i == 0:
                category_12.append(df.iloc[:, 12].quantile(i / (quantile_count - 1)) - 1 + i * uniqueness_no)
            else:
                category_12.append(df.iloc[:, 12].quantile(i / (quantile_count - 1)) + i * uniqueness_no)
        df.iloc[:, 0] = pd.cut(df.iloc[:, 0], bins=category_0, labels=range(quantile_count - 1))
        df.iloc[:, 2] = pd.cut(df.iloc[:, 2], bins=category_2, labels=range(quantile_count - 1))
        df.iloc[:, 4] = pd.cut(df.iloc[:, 4], bins=category_4, labels=range(quantile_count - 1))
        df.iloc[:, 10] = pd.cut(df.iloc[:, 10], bins=category_10, labels=range(quantile_count - 1))
        df.iloc[:, 11] = pd.cut(df.iloc[:, 11], bins=category_11, labels=range(quantile_count - 1))
        df.iloc[:, 12] = pd.cut(df.iloc[:, 12], bins=category_12, labels=range(quantile_count - 1))
    elif (type == 'test'):
        max_0 = df.iloc[:, 0].max()
        max_2 = df.iloc[:, 2].max()
        max_4 = df.iloc[:, 4].max()
        max_10 = df.iloc[:, 10].max()
        max_11 = df.iloc[:, 11].max()
        max_12 = df.iloc[:, 12].max()
        min_0 = df.iloc[:, 0].min()
        min_2 = df.iloc[:, 2].min()
        min_4 = df.iloc[:, 4].min()
        min_10 = df.iloc[:, 10].min()
        min_11 = df.iloc[:, 11].min()
        min_12 = df.iloc[:, 12].min()
        # check maximums
        if max_0 > category_0[quantile_count - 1]:
            category_0[quantile_count - 1] = max_0
        if max_2 > category_2[quantile_count - 1]:
            category_2[quantile_count - 1] = max_2
        if max_4 > category_4[quantile_count - 1]:
            category_4[quantile_count - 1] = max_4
        if max_10 > category_10[quantile_count - 1]:
            category_10[quantile_count - 1] = max_10
        if max_11 > category_11[quantile_count - 1]:
            category_11[quantile_count - 1] = max_11
        if max_12 > category_12[quantile_count - 1]:
            category_12[quantile_count - 1] = max_12
        # check mins
        if min_0 < category_0[0]:
            category_0[0] = min_0
        if min_2 < category_2[0]:
            category_2[0] = min_2
        if min_4 < category_4[0]:
            category_4[0] = min_4
        if min_10 < category_10[0]:
            category_10[0] = min_10
        if min_11 < category_11[0]:
            category_11[0] = min_11
        if min_12 < category_12[0]:
            category_12[0] = min_12

        df.iloc[:, 0] = pd.cut(df.iloc[:, 0], bins=category_0, labels=range(quantile_count - 1))
        df.iloc[:, 2] = pd.cut(df.iloc[:, 2], bins=category_2, labels=range(quantile_count - 1))
        df.iloc[:, 4] = pd.cut(df.iloc[:, 4], bins=category_4, labels=range(quantile_count - 1))
        df.iloc[:, 10] = pd.cut(df.iloc[:, 10], bins=category_10, labels=range(quantile_count - 1))
        df.iloc[:, 11] = pd.cut(df.iloc[:, 11], bins=category_11, labels=range(quantile_count - 1))
        df.iloc[:, 12] = pd.cut(df.iloc[:, 12], bins=category_12, labels=range(quantile_count - 1))
    return df


train = pd.read_csv("adult.data", header=None)
# print(train.describe())
x_train = train.drop(len(train.columns) - 1, axis=1)
y_train = train.iloc[:, len(train.columns) - 1]
# y_train = train[len(train.columns)-1]
y_train = y_train.values.reshape(len(y_train), 1)
#
x_train = categorizer(x_train, type='train')
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(x_train)
# print(enc.n_values_)
x_train = enc.transform(x_train).toarray()
# print(enc.categories_)
# y_final = enc.transform(y_train).toarray()
# print(y_final)
####test data####
test = pd.read_csv("adult.test", header=None)
x_test = test.drop(len(test.columns) - 1, axis=1)
y_test = test.iloc[:, len(test.columns) - 1]
# y_test = y_test.values.reshape(len(y_test), 1)
y_test = y_test.values
y_test = [sub[: -1] for sub in y_test]
y_test = np.array(y_test)
#
x_test = categorizer(x_test, type='test')
# enc = OneHotEncoder(handle_unknown='ignore')
# enc.fit(x_test)
x_test = enc.transform(x_test).toarray()
###########train decision tree############
dt_clf = DecisionTreeClassifier()
parameters_dt = {'max_depth':[30,40,50],'min_samples_split':[10,20,30],'min_samples_leaf':[5,10,20]}

grid_dt = GridSearchCV( dt_clf , parameters_dt, scoring=make_scorer(accuracy_score) , cv=3)
grid_dt.fit(x_train,y_train.ravel())
dt_clf = grid_dt.best_estimator_
dt_clf.fit(x_train,y_train.ravel())
############test accuracy of decision tree classifier#############
pred_dt = dt_clf.predict(x_test)
acc_dt = accuracy_score(y_test, pred_dt)

print("The Score for Decision Tree is: " + str(acc_dt))
###########train random forest############
rf_clf = RandomForestClassifier()
parameters_rf = {'n_estimators': [5, 10, 15], 'max_features': ["sqrt", "log2"], 'max_depth': [10, 20, 30]}

grid_rf = GridSearchCV(rf_clf, parameters_rf, scoring=make_scorer(accuracy_score), cv=3)
grid_rf.fit(x_train, y_train.ravel())
rf_clf = grid_rf.best_estimator_
rf_clf.fit(x_train, y_train.ravel())
#############test accuracy of random forest classifier#############
pred_rf = rf_clf.predict(x_test)
acc_rf = accuracy_score(y_test, pred_rf)

print("The Score for Random Forest is: " + str(acc_rf))
###########train random forest############
ab_clf = AdaBoostClassifier()
parameters_ab = {'n_estimators':[10,15,30] , 'learning_rate':[0.3,0.5,1]}

grid_ab = GridSearchCV(ab_clf, parameters_ab, scoring=make_scorer(accuracy_score), cv=3)
grid_ab.fit(x_train, y_train.ravel())
ab_clf = grid_ab.best_estimator_
ab_clf.fit(x_train, y_train.ravel())
#############test accuracy of random forest classifier#############
pred_ab = ab_clf.predict(x_test)
acc_ab = accuracy_score(y_test, pred_ab)

print("The Score for Adaboost is: " + str(acc_ab))

