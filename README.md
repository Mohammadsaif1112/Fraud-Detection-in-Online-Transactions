# Fraud-Detection-in-Online-Transactions
Fraud is a significant problem in online transactions, costing businesses billions of dollars each year.
In this project, I am going to do explorations to try and understand the fraud transaction patterns and then implement some models of machine learning. 
Due to the class imbalance ratio of this kind of data, I will measure the accuracy using the Area Under the Precision-Recall Curve (AUPRC). 
The reason is because confusion matrix accuracy is not meaningful for unbalanced classification.

#Importing the libraries
import pandas as pd #To hand with data 
import numpy as np #To math 
import seaborn as sns #to visualization
import matplotlib.pyplot as plt # to plot the graphs
import matplotlib.gridspec as gridspec # to do the grid of plots

#loading the data
df_credit = pd.read_csv("../input/creditcard.csv")

# Looking how the data looks like
df_credit.head()
Output
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/7e994888-65a1-4d15-b7c5-1271e40236aa)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/2097922f-c062-42c9-9ecc-386cab7246e4)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/006735d3-c673-46df-83f6-1afbf1e21415)

# Determining the type of data and checking whether there are null values
df_credit.info()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/379d7094-5491-4258-9c01-14306d82a2ea)

# The data is stardarized
#For now I will look the "normal" columns
df_credit[["Time","Amount","Class"]].describe()
Output
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/f75841a5-dc48-4513-b0e3-7ab60d89d924)

The data will be explored through 3 cplumns: Time, Amount, and Class
# I will start by looking at the difference by Normal and Fraud transactions
print("Distribuition of Normal(0) and Frauds(1): ")
print(df_credit["Class"].value_counts())

plt.figure(figsize=(7,5))
sns.countplot(df_credit['Class'])
plt.title("Class Count", fontsize=18)
plt.xlabel("Is fraud?", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/a4d39540-df0f-4d0c-8eed-b47b671836f5)
It is clear that the data is highly imbalanced. However, this is very common when dealing with fraud 

I will first explotre the data through the Time and Amount, and then explore the V's Features, that are PCA's
Time Features
Since the feature is in seconds I will transform it to minutes and hours to get a better understanding of the patterns
timedelta = pd.to_timedelta(df_credit['Time'], unit='s')
df_credit['Time_min'] = (timedelta.dt.components.minutes).astype(int)
df_credit['Time_hour'] = (timedelta.dt.components.hours).astype(int)

#Exploring the distribution by Class types through hours
plt.figure(figsize=(12,5))
sns.distplot(df_credit[df_credit['Class'] == 0]["Time_hour"], 
             color='g')
sns.distplot(df_credit[df_credit['Class'] == 1]["Time_hour"], 
             color='r')
plt.title('Fraud x Normal Transactions by Hours', fontsize=17)
plt.xlim([-1,25])
plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/0660d5d5-4003-4020-a14a-ee98976186fc)

#Exploring the distribution by Class types through minutes
plt.figure(figsize=(12,5))
sns.distplot(df_credit[df_credit['Class'] == 0]["Time_min"], 
             color='g')
sns.distplot(df_credit[df_credit['Class'] == 1]["Time_min"], 
             color='r')
plt.title('Fraud x Normal Transactions by minutes', fontsize=17)
plt.xlim([-1,61])
plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/860365aa-bdcb-4d01-b71d-2a52c9e6b845)

I will now look at the statistics of the Amount class frauds and normal transactions
# Show the data of frauds and no frauds
df_fraud = df_credit[df_credit['Class'] == 1]
df_normal = df_credit[df_credit['Class'] == 0]

print("Fraud transaction statistics")
print(df_fraud["Amount"].describe())
print("\nNormal transaction statistics")
print(df_normal["Amount"].describe())
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/bdc9ff6a-7e73-4ccf-bcbe-f2cdf6601c69)

I will then use this informations to ilter the values to look for Amount by Class
I will filter the "normal" amounts by 3.000

# Better visualization of the values through Feature engineering
df_credit['Amount_log'] = np.log(df_credit.Amount + 0.01)
plt.figure(figsize=(14,6))
#I will explore the Amount by Class and see the distribuition of Amount transactions
plt.subplot(121)
ax = sns.boxplot(x ="Class",y="Amount",
                 data=df_credit)
ax.set_title("Class x Amount", fontsize=20)
ax.set_xlabel("Is Fraud?", fontsize=16)
ax.set_ylabel("Amount(US)", fontsize = 16)

plt.subplot(122)
ax1 = sns.boxplot(x ="Class",y="Amount_log", data=df_credit)
ax1.set_title("Class x Amount", fontsize=20)
ax1.set_xlabel("Is Fraud?", fontsize=16)
ax1.set_ylabel("Amount(Log)", fontsize = 16)

plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/ea184955-d72b-4593-8d70-4351e2ce17c3)
As seen in the graph, there is a slight difference in the log amount of the two Classes.
The IQR of fraudulent transactions is higher than normal transactions, but normal transactions have highest values

I will now generate a scatter plot of the Time_min distribuition by Amount

#Looking the Amount and time distribuition of Fraud transactions
ax = sns.lmplot(y="Amount", x="Time_min", fit_reg=False,aspect=1.8,
                data=df_credit, hue='Class')
plt.title("Amounts by Minutes of Frauds and Normal Transactions",fontsize=16)
plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/41a5ea67-bfc7-4ad5-89e2-c0a3e5176e3c)

Scatter plot of the Time_hour distribuition by Amount
ax = sns.lmplot(y="Amount", x="Time_hour", fit_reg=False,aspect=1.8,
                data=df_credit, hue='Class')
plt.title("Amounts by Hour of Frauds and Normal Transactions", fontsize=16)

plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/7014c205-f0ea-4572-b728-238666f814ea)

Using boxplot to search for different distribuitions:
I will search for features that diverges from normal distribuition

#Looking the V's features
columns = df_credit.iloc[:,1:29].columns

frauds = df_credit.Class == 1
normals = df_credit.Class == 0

grid = gridspec.GridSpec(14, 2)
plt.figure(figsize=(15,20*4))

for n, col in enumerate(df_credit[columns]):
    ax = plt.subplot(grid[n])
    sns.distplot(df_credit[col][frauds], bins = 50, color='g') #Will receive the "semi-salmon" violin
    sns.distplot(df_credit[col][normals], bins = 50, color='r') #Will receive the "ocean" color
    ax.set_ylabel('Density')
    ax.set_title(str(col))
    ax.set_xlabel('')
plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/09da053c-bf26-4dc8-8937-869d41e9e952)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/27a22ca8-56e0-41e5-86a0-5a8ef5c096a4)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/dde3411e-5f4d-40a5-847d-5611dcd026e6)
As seen above, there is an interesting different distribuition in some of the features like V4, V9, V16, V17 and so forth.
Now let's take a look on time distribuition

Diference in time
Feature selections
#I will select the variables where fraud class have an interesting behavior and can help in predicting

df_credit = df_credit[["Time_hour","Time_min","V2","V3","V4","V9","V10","V11","V12","V14","V16","V17","V18","V19","V27","Amount","Class"]]

Feature Engineering
df_credit.Amount = np.log(df_credit.Amount + 0.001)
#Looking the final df
df_credit.head()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/122a0243-4c56-446b-af35-85b27887faef)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/64ec2e84-9404-471d-acb8-a7fd2d59d05d)

colormap = plt.cm.Greens

plt.figure(figsize=(14,12))

sns.heatmap(df_credit.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap = colormap, linecolor='white', annot=True)
plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/12f5c637-04fe-4f0f-a6b6-3bbab3b00e00)

Preprocessing
from imblearn.pipeline import make_pipeline as make_pipeline_imb # To do our transformation in a unique time
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

from sklearn.model_selection import train_test_split
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve, accuracy_score
X = df_credit.drop(["Class"], axis=1).values #Setting the X to do the split
y = df_credit["Class"].values # transforming the values in array

# The function to be used to better evaluate the model
def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f2: {}".format(fbeta_score(true_value, pred, beta=2)))

# splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.20)

classifier = RandomForestClassifier

# build model with SMOTE imblearn
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), \
                                   classifier(random_state=42))

smote_model = smote_pipeline.fit(X_train, y_train)
smote_prediction = smote_model.predict(X_test)

#Showing the diference before and after the transformation used
print("normal data distribution: {}".format(Counter(y)))
X_smote, y_smote = SMOTE().fit_sample(X, y)
print("SMOTE data distribution: {}".format(Counter(y_smote)))
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/eec8c3e0-d2b4-448d-8c1c-8aea20f95089)

Evaluate the model SMOTE + Random Forest
print("Confusion Matrix: ")
print(confusion_matrix(y_test, smote_prediction))

print('\nSMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))

print_results("\nSMOTE + RandomForest classification", y_test, smote_prediction)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/0f65ac40-c343-46ca-a8bd-f724180481a7)
# Compute predicted probabilities: y_pred_prob
y_pred_prob = smote_pipeline.predict_proba(X_test)[:,1]

# Generate precision recall curve values: precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/99c732fc-9cbc-407f-bba1-efe7f88eaaea)
From the graph, it is clear that it is overfitted.
I am now going to correct this overfitting. 
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV

#params of the model
param_grid = {"max_depth": [3,5, None],
              "n_estimators":[3,5,10],
              "max_features": [5,6,7,8]}

# Creating the classifier
model = RandomForestClassifier(max_features=3, max_depth=2 ,n_estimators=10, random_state=3, criterion='entropy', n_jobs=1, verbose=1 )
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='recall')
grid_search.fit(X_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/088b43a7-b7f4-4bc1-9b17-c2e9a4ca7082)

# Running the fit
rf = RandomForestClassifier(max_depth=5, max_features = 7, n_estimators = 10)
rf.fit(X_train, y_train)
Output
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/8d9bb3e1-3d5d-4678-8345-7727d828c710)

# Printing the Training Score
print("Training score data: ")
print(rf.score(X_train, y_train))
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/22eca625-9213-4ce9-8740-e9509566a698)
#Testing the model 
#Predicting by X_test
y_pred = rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print_results("RF classification", y_test, y_pred)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/ef7fac57-baa7-44be-a6e7-a523aef5265a)

Feature importance plot
features = ["Time_min", 'Time_hours',"V2","V3","V4","V9","V10","V11","V12","V14","V16","V17","V18","V19","V27","Amount"]

# Credits to Gabriel Preda
# https://www.kaggle.com/gpreda/credit-card-fraud-detection-predictive-models
plt.figure(figsize = (9,5))

feat_import = pd.DataFrame({'Feature': features, 'Feature importance': rf.feature_importances_})
feat_import = feat_import.sort_values(by='Feature importance',ascending=False)

g = sns.barplot(x='Feature',y='Feature importance',data=feat_import)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title('Features importance - Random Forest',fontsize=20)
plt.show() 
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/97961e21-d438-432e-b8f0-678208f3373c)
As seen above, the top 4 features are V17, V14, V12, V10, correcponding to 75% of total.

The f2 score that is the median of recall and precision is also a considerable value

ROC CURVE - Random Forest
#Predicting proba
y_pred_prob = rf.predict_proba(X_test)[:,1]

# Generate precision recall curve values: precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/36890e61-6c95-43e3-be01-45433122d63c)
results = cross_val_score(rf,X_train, y_train, cv=10, scoring='recall')
results
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/6b99cc72-69b3-4bf3-9e11-516f79a296cd)

Modelling Logistic Regression with Hyper Parameters
param_grid = {'C': [0.01, 0.1, 1, 10],
             'penalty':['l1', 'l2']}

logreg = LogisticRegression(random_state=2)

grid_search_lr = GridSearchCV(logreg, param_grid=param_grid, scoring='recall', cv=5)

grid_search_lr.fit(X_train, y_train)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/0f02aaa6-f7df-4ac6-a262-53e8e104964f)

# The best recall obtained
print(grid_search_lr.best_score_)
#Best parameter on trainning set
print(grid_search_lr.best_params_)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/8bb3897f-a7c9-41ff-8b00-afd6c8ef1435)

Setting the best parameters as parameters of our model
# Creating the model 
logreg = LogisticRegression(C=10, penalty='l2',random_state=2)

#Fiting the model
logreg.fit(X_train, y_train)
           
# Printing the Training Score
print("Cross Validation of X and y Train: ")
print(cross_val_score(logreg,X_train, y_train, cv=5, scoring='recall'))
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/8aa74afc-787c-478e-bb32-f4cd14a17145)
# Predicting with the best params
y_pred = logreg.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("")
print_results("LogReg classification", y_test, y_pred)
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/6dc29751-f143-4a16-b159-ea80a7b66f7c)
70% of accuracy is not too bad, but we found a high vale on the Random Forest Model

Precision Recall Curve of Logistic Regression
#Predicting proba
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate precision recall curve values: precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot(precision, recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()
![image](https://github.com/mk2287/Fraud-Detection-in-Online-Transactions/assets/152664423/06de2298-91d1-4ca5-870c-c43f725990ce)

CONCLUSION:
The highest values of Normal transactions is 25691.16 while of Fraudulent transactions 2125.87. The average value of normal transactions is small(about 88.29) 
than fraudulent transactions that is 122.21. 
I obtained the best score when I used the SMOTE (OverSampling) + RandomForest, that performed a f2 score of approximately 0.8669
This is a considerably different compared to the second best model that is 0.8252, using just RandomForests with some Hyper Parameters.
The worst model was Logreg where I used GridSearchCV to get the Best params to fit and predict where the recall was approximately 0.6666 and f2 approximately 0.70
