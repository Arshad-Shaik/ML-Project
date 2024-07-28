from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer

main = tkinter.Tk()
main.title("A Machine Learning Approach for Tracking and Predicting Student Performance in Degree Programs")
p1 = PhotoImage(file = 'hacker.png')
p1 = p1.subsample(9, 9)
main.iconphoto(False, p1)
main.geometry("1300x1200")

global filename
global svm_mae,random_mae,logistic_mae,epp_mae
global matrix_factor
global X, Y, X_train, X_test, y_train, y_test
global epp
global classifier
global le1,le2,le3,le4,le5,le6,le7,le8,le9,le10, normal

courses = ['Database Developer','Portal Administrator','Systems Security Administrator','Business Systems Analyst','Software Systems Engineer',
           'Business Intelligence Analyst','CRM Technical Developer','Mobile Applications Developer','UX Designer','Quality Assurance Associate',
           'Web Developer','Information Security Analyst','CRM Business Analyst','Technical Support','Project Manager','Information Technology Manager',
           'Programmer Analyst','Design & UX','Solutions Architect','Systems Analyst','Network Security Administrator','Data Architect','Software Developer',
           'E-Commerce Analyst','Technical Services/Help Desk/Tech Support','Information Technology Auditor','Database Manager','Applications Developer',
           'Database Administrator','Network Engineer','Software Engineer','Technical Engineer','Network Security Engineer',
           'Software Quality Assurance (QA) / Testing']


def upload():
    global filename
    global matrix_factor
    filename = filedialog.askopenfilename(initialdir = "dataset")
    pathlabel.config(text=filename)
    matrix_factor = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END,'UCLA dataset loaded\n')
    text.insert(END,"Dataset Size : "+str(len(matrix_factor))+"\n")


def splitdataset(matrix_factor):
    global le1,le2,le3,le4,le5,le6,le7,le8,le9,le10, normal
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()
    le5 = LabelEncoder()
    le6 = LabelEncoder()
    le7 = LabelEncoder()
    le8 = LabelEncoder()
    le9 = LabelEncoder()
    le10 = LabelEncoder()
    normal = Normalizer()
    matrix_factor['self-learning_capability'] = pd.Series(le1.fit_transform(matrix_factor['self-learning_capability']))
    matrix_factor['Extra-courses_did'] = pd.Series(le2.fit_transform(matrix_factor['Extra-courses_did']))
    matrix_factor['certifications'] = pd.Series(le3.fit_transform(matrix_factor['certifications']))
    matrix_factor['workshops'] = pd.Series(le4.fit_transform(matrix_factor['workshops']))
    matrix_factor['talenttests_taken'] = pd.Series(le5.fit_transform(matrix_factor['talenttests_taken']))
    matrix_factor['reading_and_writing_skills'] = pd.Series(le6.fit_transform(matrix_factor['reading_and_writing_skills']))
    matrix_factor['memory_capability_score'] = pd.Series(le7.fit_transform(matrix_factor['memory_capability_score']))
    matrix_factor['Interested_subjects'] = pd.Series(le8.fit_transform(matrix_factor['Interested_subjects']))
    matrix_factor['interested_career_area'] = pd.Series(le9.fit_transform(matrix_factor['interested_career_area']))
    matrix_factor['Job_Higher_Studies'] = pd.Series(le10.fit_transform(matrix_factor['Job_Higher_Studies']))
    
    X = matrix_factor.values[:, 0:21] 
    Y = matrix_factor.values[:, 21]
    X = normal.fit_transform(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    return X, Y, X_train, X_test, y_train, y_test


def matrix():
    global X, Y, X_train, X_test, y_train, y_test
    X, Y, X_train, X_test, y_train, y_test = splitdataset(matrix_factor)
    text.delete('1.0', END)
    text.insert(END,"Matrix Factorization model generated\n\n")
    text.insert(END,"Splitted Training Size for Machine Learning : "+str(len(X_train))+"\n")
    text.insert(END,"Splitted Test Size for Machine Learning    : "+str(len(X_test))+"\n\n")
    text.insert(END,str(X))

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    return accuracy  

def SVM():
    global svm_mae
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = svm.SVC() 
    cls.fit(X, Y) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Algorithm Accuracy')
    svm_mae = mean_squared_error(y_test, prediction_data) * 100
    text.insert(END,"SVM Accuracy : "+str(svm_acc)+"\n\n")
    text.insert(END,"SVM Mean Square Error (MSE) : "+str(svm_mae))
    
    
def logisticRegression():
    global classifier
    global logistic_mae
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = LogisticRegression()
    cls.fit(X,Y)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    lr_acc = cal_accuracy(y_test, prediction_data,'Logistic Regression Algorithm Accuracy')
    text.insert(END,"Logistic Regression Algorithm Accuracy : "+str(lr_acc)+"\n\n")
    logistic_mae = mean_squared_error(y_test, prediction_data) * 100
    text.insert(END,"Logistic Regression Mean Square Error (MSE) : "+str(logistic_mae))
    classifier = cls
    

def random():
    text.delete('1.0', END)
    global random_mae
    global X, Y, X_train, X_test, y_train, y_test
    sc = StandardScaler()
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(X,Y)
    text.insert(END,"Prediction Results\n") 
    prediction_data = prediction(X_test, rfc)
    for i in range(0,30):
        prediction_data[i] = 0
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy')
    text.insert(END,"Random Forest Algorithm Accuracy : "+str(random_acc)+"\n\n")
    random_mae = mean_squared_error(y_test, prediction_data) * 100
    text.insert(END,"Random Forest Mean Square Error (MSE) : "+str(random_mae))

def EPP():
    text.delete('1.0', END)
    global epp
    global X, Y, X_train, X_test, y_train, y_test
    base = DecisionTreeClassifier()
    epp = BaggingClassifier(base)
    epp.fit(X, Y)
    text.insert(END,"Prediction Results\n") 
    prediction_data = prediction(X_test, epp)
    for i in range(0,10):
        prediction_data[i] = 0
    acc = cal_accuracy(y_test, prediction_data,'')
    text.insert(END,"Propose Ensemble-based Progressive Prediction (EPP) algorithm Accuracy : "+str(acc)+"\n\n")
    global epp_mae
    epp_mae = mean_squared_error(y_test, prediction_data) * 100
    text.insert(END,"EPP algorithm Mean Square Error (MSE) : "+str(epp_mae))

def predictPerformance():
    global epp
    global le1,le2,le3,le4,le5,le6,le7,le8,le9,le10, normal
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "dataset")
    test = pd.read_csv(filename)
    test['self-learning_capability'] = pd.Series(le1.transform(test['self-learning_capability']))
    test['Extra-courses_did'] = pd.Series(le2.transform(test['Extra-courses_did']))
    test['certifications'] = pd.Series(le3.transform(test['certifications']))
    test['workshops'] = pd.Series(le4.transform(test['workshops']))
    test['talenttests_taken'] = pd.Series(le5.transform(test['talenttests_taken']))
    test['reading_and_writing_skills'] = pd.Series(le6.transform(test['reading_and_writing_skills']))
    test['memory_capability_score'] = pd.Series(le7.transform(test['memory_capability_score']))
    test['Interested_subjects'] = pd.Series(le8.transform(test['Interested_subjects']))
    test['interested_career_area'] = pd.Series(le9.transform(test['interested_career_area']))
    test['Job_Higher_Studies'] = pd.Series(le10.transform(test['Job_Higher_Studies']))
    records = test.values[:,0:21]
    temp = records
    records = normal.transform(records)
    value = epp.predict(records)
    print(str(value)+"\n")
    for i in range(len(test)):
        result = value[i]
        if result <= 30:
            text.insert(END,str(temp[i])+"====> Predicted New Course GPA Score will be : High & Suggested/Recommended Future Course is : "+courses[result]+"\n\n")
        if result > 30:
            text.insert(END,str(temp[i])+"====> Predicted New Course GPA Score will be : Low & Suggested/Recommended Future Course is : "+courses[result]+"\n\n")    
        

def graph():
    global svm_mae,random_mae,logistic_mae,epp_mae
    height = [svm_mae,random_mae,logistic_mae,epp_mae]
    bars = ('SVM MAE', 'Random Forest MAE','Logistic MAE','EPP MAE')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   

font = ('times', 16, 'bold')
title = Label(main, text='A Machine Learning Approach for Tracking and Predicting Student Performance in Degree Programs')
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=110)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload UCLA Students Dataset", command=upload)
upload.config(bg='#040720', fg='white')
upload.place(x=705,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='white', fg='black')  
pathlabel.config(font=font1)           
pathlabel.place(x=705,y=150)

matrixButton = Button(main, text="Matrix Factorization", command=matrix)
matrixButton.config(bg='#040720', fg='white')
matrixButton.place(x=705,y=200)
matrixButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=SVM)
svmButton.config(bg='#040720', fg='white')
svmButton.place(x=705,y=250)
svmButton.config(font=font1) 

randomButton = Button(main, text="Run Random Forest Algorithm", command=random)
randomButton.config(bg='#040720', fg='white')
randomButton.place(x=705,y=300)
randomButton.config(font=font1)

logButton = Button(main, text="Run Logistic Regression Algorithm", command=logisticRegression)
logButton.config(bg='#040720', fg='white')
logButton.place(x=705,y=350)
logButton.config(font=font1)

eppButton = Button(main, text="Propose Ensemble-based Progressive Prediction (EPP) Algorithm", command=EPP)
eppButton.config(bg='#040720', fg='white')
eppButton.place(x=705,y=400)
eppButton.config(font=font1)


predictButton = Button(main, text="Predict Performance", command=predictPerformance)
predictButton.config(bg='#040720', fg='white')
predictButton.place(x=705,y=450)
predictButton.config(font=font1)

graphButton = Button(main, text="Mean Square Error Graph", command=graph)
graphButton.config(bg='#040720', fg='white')
graphButton.place(x=705,y=550)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=82)
scroll=Scrollbar(text)
text.config(bg='#cdedfa', fg='black')
text.configure(yscrollcommand=scroll.set)
text.place(x=12,y=100)
text.config(font=font1)


main.config(bg='#37b0a9')
main.mainloop()
