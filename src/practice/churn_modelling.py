import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv(
    "D:\\study\\all\\code\\clothing-classifier\\data\\Churn_Modelling.csv")

data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
data["Geography"] = data["Geography"].map(
    {"Spain": 0, "France": 1, "Germany": 2})

data = data[["Gender", "Tenure", "CreditScore", "Balance",
             "IsActiveMember", "EstimatedSalary", "NumOfProducts", "Exited"]].dropna()
data["BalancePerProduct"] = data["Balance"]/data["NumOfProducts"]

X = data[["Gender", "Tenure", "CreditScore", "Balance", "IsActiveMember",
          "EstimatedSalary", "NumOfProducts", "BalancePerProduct"]]
Y = data[["Exited"]]


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.15, random_state=43)

model = LogisticRegression()
model_rf = RandomForestClassifier(n_estimators=100)
model_grad = GradientBoostingClassifier(n_estimators=150)

model_rf.fit(X_train, y_train)
model.fit(X_train, y_train)
model_grad.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_grad = model_grad.predict(X_test)


print("Linear Accuracy score: ", accuracy_score(y_test, y_pred))
print("RF Accuracy score: ", accuracy_score(y_test, y_pred_rf))
print("Gradient Accuracy score: ", accuracy_score(y_test, y_pred_grad))
