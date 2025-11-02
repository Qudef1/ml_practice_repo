import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

data["Sex"] = data['Sex'].map({'male':0,"female":1})
data = data[["Pclass",'Sex',"Age",'Survived',"Fare","Parents/Children Aboard"]].dropna()

X = data[['Pclass',"Sex","Age","Fare","Parents/Children Aboard"]]
Y = data["Survived"]

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=43)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))


