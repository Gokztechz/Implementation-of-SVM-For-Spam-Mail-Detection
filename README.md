# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the necessary packages using import statement.
2. Read the given csv file and print the number of contents to be displayed. 
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy. 5.Display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: R GOKUL SHARAN
RegisterNumber:  212223040052
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
## Data.head():
![image](https://github.com/Gokztechz/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117667038/afdc14d2-1cd5-4368-911c-16118ba3da1f)
## Data.info():
![image](https://github.com/Gokztechz/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117667038/2d3976bf-023c-4f43-98db-155dd6d641ec)
## Data.isnull().sum():
![image](https://github.com/Gokztechz/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117667038/1736a221-9f21-45e8-a2da-cdbe7773fc0e)
## Y_Pred:
![image](https://github.com/Gokztechz/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117667038/bfe9fed2-4868-4bf6-8fa9-6cd390ed24ab)
## Accuracy:
![image](https://github.com/Gokztechz/Implementation-of-SVM-For-Spam-Mail-Detection/assets/117667038/b93254e9-4342-4b96-ad79-5221865423fd)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
