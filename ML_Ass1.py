
# coding: utf-8

# In[1]:


"""
Created By:
RIYA KURIAKOSE
"""

#Importing the required modules

import pandas as pd
import numpy as np
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import  classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Importing the dataset

data = pd.read_csv(r"C:\Users\Admin\Desktop\IIT SEM1\ML\dataset - Sheet1.csv")
#print(data)
predicted_value = list(data['Predicted Class'])
actual_value = list(data['Actual Class'])
print(len(predicted_value))

#Creating the Confusion Matrix

def create_cm(actual,predicted):
    classes = len(np.unique(actual))
    cm = np.zeros((classes,classes))

    for i in range(len(actual)):
      cm[actual[i]] [predicted[i]] += 1
    return cm

conf_matrix = create_cm(np.array(actual_value),np.array(predicted_value))
print("CONFUSION MATRIX")
print(conf_matrix)


# In[2]:


#Calculating Accuracy

accuracy = conf_matrix.trace()/conf_matrix.sum()
print("Accuracy",accuracy)
print("Accuracy using sklearn",accuracy_score(actual_value,predicted_value))


# In[3]:


#Calculating the values of TP,TN,FP,FN

TP0 = conf_matrix[0][0]
TP1 = conf_matrix[1][1]
TP2 = conf_matrix[2][2]
TP3 = conf_matrix[3][3]

TN0 =  np.sum(conf_matrix[1: ,1:])
TN1 =  np.sum(np.delete((np.delete(conf_matrix,1,0)),1,1))
TN2 =  np.sum(np.delete((np.delete(conf_matrix,2,0)),2,1))
TN3 =  np.sum(np.delete((np.delete(conf_matrix,3,0)),3,1))

FN0 = np.sum(conf_matrix[:1,1:])
FN1 = np.sum(conf_matrix[1:2,:])-conf_matrix[1,1]
FN2 = np.sum(conf_matrix[2:3,:])-conf_matrix[2,2]
FN3 = np.sum(conf_matrix[3:4,:])-conf_matrix[3,3]

FP0 = np.sum(conf_matrix[1:,:1])
FP1 = np.sum(conf_matrix[:,1:2])-conf_matrix[1,1]
FP2 = np.sum(conf_matrix[:,2:3])-conf_matrix[2,2]
FP3 = np.sum(conf_matrix[:,3:4])-conf_matrix[3,3]


# In[4]:



#Calculating Classwise Accuracy

print("CLASSWISE ACCURACY")
print("Class 0 Accuracy = ", (TP0+TN0)/(TP0+TN0+FP0+FN0))
print("Class 1 Accuracy = ", (TP1+TN1)/(TP1+TN1+FP1+FN1))
print("Class 2 Accuracy = ", (TP2+TN2)/(TP2+TN2+FP2+FN2))
print("Class 3 Accuracy = ", (TP3+TN3)/(TP3+TN3+FP3+FN3))
print("----------------------------------------------------")


# In[5]:


#Calculating Classwise Precision

precision_0 = TP0/(TP0+FP0)
precision_1 = TP1/(TP1+FP1)
precision_2 = TP2/(TP2+FP2)
precision_3 = TP3/(TP3+FP3)
print("PRECISION")
print("Class 0 Precision = ", precision_0)
print("Class 1 Precision = ", precision_1)
print("Class 2 Precision = ", precision_2)
print("Class 3 Precision = ", precision_3)
print("----------------------------------------------------")


# In[6]:


#Calculating Classwise Recall

recall_0 = TP0/(TP0+FN0)
recall_1 = TP1/(TP1+FN1)
recall_2 = TP2/(TP2+FN2)
recall_3 = TP3/(TP3+FN3)
print("RECALL")
print("Class 0 Recall = ", recall_0)
print("Class 1 Recall = ", recall_1)
print("Class 2 Recall = ", recall_2)
print("Class 3 Recall = ", recall_3)
print("----------------------------------------------------")


# In[7]:


#Calculating Classwise F1 Score

print("F1 SCORE")
print("Class 0 F1 SCORE = ", (2*precision_0*recall_0)/(precision_0+recall_0))
print("Class 1 F1 SCORE = ", (2*precision_1*recall_1)/(precision_1+recall_1))
print("Class 2 F1 SCORE = ", (2*precision_2*recall_2)/(precision_2+recall_2))
print("Class 3 F1 SCORE = ", (2*precision_3*recall_3)/(precision_3+recall_3))
print("----------------------------------------------------")


# In[8]:


#Calculating Macro Average Precision and Recall

print("MACR0 AVERAGE PRECISION")
print("MACRO Average of all classes = ",(precision_0+precision_1+precision_2+precision_3)/4)
print("----------------------------------------------------")

print("MACR0 AVERAGE RECALL")
print("MACRO Average of all classes = ",(recall_0+recall_1+recall_2+recall_3)/4)
print("----------------------------------------------------")

#Calculating Weighted Average Precision
print("WEIGHTED AVERAGE PRECISION")
print("WEIGHTED Average of all classes = ",(TP0+TP1+TP2+TP3)/160)
print("----------------------------------------------------")


# In[9]:


#Calculating Type 1 and Type 2 Errors
print("TYPE1 ERROR")
print("Type 1 error for class0",FN0)
print("Type 1 error for class1",FN1)
print("Type 1 error for class2",FN2)
print("Type 1 error for class3",FN3)
print("Type 1 error for data",FN0+FN1+FN2+FN3)
print("----------------------------------------------------")

print("TYPE2 ERROR")
print("Type 2 error for class0",FP0)
print("Type 2 error for class1",FP1)
print("Type 2 error for class2",FP2)
print("Type 2 error for class3",FP3)
print("Type 2 error for data",FP0+FP1+FP2+FP3)
print("----------------------------------------------------")
  


# In[10]:


#Printing Classification Report using sklearn
print("sklearn classification report",classification_report(actual_value,predicted_value))
print("-----------------------------------------------------")


# In[11]:


#Calculating mean absolute ans mean squared error
print("Mean Absolute Error:",mean_absolute_error(actual_value,predicted_value))
print("Mean Squared Error",mean_squared_error(actual_value,predicted_value))

