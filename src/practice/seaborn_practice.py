import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# pclass and sex(female) main features
from MyLogisticRegression import MyLogisticRegression


def input_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        if Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


def input_fare(cols):
    Fare = cols[0]
    Pclass = cols[1]
    if pd.isnull(Fare):
        if Pclass == 1:
            return 150
        if Pclass == 2:
            return 50
        else:
            return 30
    else:
        return Fare


train = pd.read_csv(
    "D:\\study\\all\\code\\clothing-classifier\\data\\train.csv")
sns.set_theme(style="whitegrid")
test = pd.read_csv(
    "D:\\study\\all\\code\\clothing-classifier\\data\\titanic_test.csv")
# barplot survived(passenger class) divided into sex
# sns.barplot(x=train['Pclass'], y=train['Survived'], hue=train['Sex'])
# sns.violinplot(data=train, x='Survived', y='Age', hue='Sex', split=True) violin plot график плотности распределения

# sns.scatterplot(data=train, x='Fare', y='Survived', hue='Pclass')
# sns.regplot(data=train, x='Fare', y='Survived', scatter=False, color='Red') график зависимости стоимости билета от выживаемости с линией тренда
# sns.histplot(train, x='Pclass', y='Fare')

train['Age'] = train[['Age', "Pclass"]].apply(input_age, axis=1)
test['Age'] = test[['Age', "Pclass"]].apply(input_age, axis=1)
mean_fare = test["Fare"].mean()
test["Fare"] = test[["Fare", "Pclass"]].apply(input_fare, axis=1)
test = test[["Age", "Pclass", "Sex", "Fare"]]


model = LogisticRegression()
my_model = MyLogisticRegression(lr=0.007, n_iters=2000, lambda_l1=0.01)
X = train[["Age", "Pclass", "Sex", "Fare"]]
Y = train["Survived"]
X["Sex"] = X["Sex"].map({'male': 0, 'female': 1})
test["Sex"] = test["Sex"].map({'male': 0, 'female': 1})
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=50)
model.fit(X_train, Y_train)
my_model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
y_pred_my = my_model.predict(X_test)
print(classification_report(Y_test, y_pred))
print(classification_report(Y_test, y_pred_my))
print(accuracy_score(Y_test, y_pred_my))
