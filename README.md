# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result. 

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: AJAYRAJA RATHINAM T
RegisterNumber: 212224240006
```
```
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:
<img width="656" height="388" alt="image" src="https://github.com/user-attachments/assets/a76e6297-5fa6-4c7b-a13c-2621ba9f40fc" /><br><br>
<img width="592" height="31" alt="image" src="https://github.com/user-attachments/assets/6365c6f8-8a09-4e89-abdd-e335cc003068" /><br><br>
<img width="1183" height="142" alt="image" src="https://github.com/user-attachments/assets/1406fc0f-889a-4f51-a4f6-2323085b97bd" /><br><br>
<img width="230" height="36" alt="image" src="https://github.com/user-attachments/assets/80f8910f-2a7e-49e8-8b00-e90df333c305" /><br><br>
<img width="123" height="55" alt="image" src="https://github.com/user-attachments/assets/3cedf24c-aaf5-455e-be3d-1fd09d48408d" /><br><br>
<img width="497" height="149" alt="image" src="https://github.com/user-attachments/assets/6b7fadc6-5453-4efd-bce1-05393024cc9a" />






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
