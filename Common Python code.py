#################################################
#import libraries
#################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
import seaborn as sns
import utils

#################################################
#Read data
#################################################
df = pd.read_csv('./data/patient-data.csv')

#################################################
#Exploratory Data Analytics
#################################################
# columns
print("\n*** Columns ***")
print(df.columns)

# info
print("\n*** Structure ***")
print(df.info())

# summary
print("\n*** Summary ***")
print(df.describe())

# head
print("\n*** Head ***")
print(df.head())

# shape
print("\n*** Shape ***")
print(df.shape)

#################################################
# Dependent Variable (In case of Linear Regression)
#################################################
depVars = "median_house_value"
print("\n*** Dep Vars ***")
print(depVars)

#################################################
# Class Variable & Counts (In case of Classification)
#################################################

# store class variable  
# change as required
clsVars = "Species"
print("\n*** Class Vars ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# handle bad Species if any (Can be done in transformation as well when required)
df['Species'] = np.where(df['Species']=="setosa","Iris-setosa",df['Species'])
df['Species'] = df['Species'].str.title()
df['Species'] = df['Species'].str.strip()

# get unique Species names
print("\n*** Unique Species - Categoric Alpha***")
lnLabels = df[clsVars].unique()
print(lnLabels)

# convert string / categoric to numeric (To be done before transformation when it is for class variable)
print("\n*** Unique Species - Categoric Alpha to Numeric ***")
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df[clsVars] = le.fit_transform(df[clsVars])
lnCCodes = df[clsVars].unique()
print(lnCCodes)

#################################################
#Data Transformation
#################################################
# drop cols
print("\n*** Drop Cols ***")
dfId = df['Ser']     # store Id in dfID to recreate dataframe later(if required not to use always)
df = df.drop('Ser', axis=1)
print("Done ...")

# convert string / categoric to numeric (In case there you identify any categoric column)
print("Unique ocean_proximity")
print(df['ocean_proximity'].unique())
from sklearn import preprocessing
leOpr = preprocessing.LabelEncoder()
df['ocean_proximity'] = leOpr.fit_transform(df['ocean_proximity'])
print(df['ocean_proximity'].unique())
print("Done ...")
##############
# OR -->using categorical method from pandas
##############
print("\n*** Transformations ***")
lstLabels = ['Type','Origin','DriveTrain']
for label in lstLabels: 
    df[label] = pd.Categorical(df[label])
    df[label] = df[label].cat.codes
print("Done ...")
##############
OR -->using map method  (have written this with respect to Iris dataset)
##############
# use map
df['Species'] = df['Species'].map({"Iris-setosa": 0, "Iris-varginica": 1, "Iris-versicolor": 2})
print("Data Type: ")
print(df['InsStatus'].dtypes)

# convert sepal width to float (In case if there is any column of 'object' data type)
df['SepalWidth'] = pd.to_numeric(df['SepalWidth'], errors = "coerce")

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers if required
print('\n*** Outlier Handling ***')
df = utils.HandleOutlier(df)
print("Done ...")

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# handle zeros if required by making zeros null
print('\n*** Handle Zeros ***')
colNames = df.columns.tolist()
colNames.remove(clsVars) #(Remove is used just because we are handling zeros on complete df)
for colName in colNames:
    df[colName] = np.where(df[colName] == 0, None, df[colName])
print("Done ...")

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# check mean
print('\n*** Mean In Columns ***')
print(df.mean())

# handle normalization if required (when there is huge difference in individual columns)
print('\n*** Normalize Data ***')
df = utils.NormalizeData(df)
print('Done ...')

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required 
# drop col with more than 50% nulls
print('\n*** Handle Nulls ***')
colNames = df.columns.tolist()
colNames.remove(clsVars) #(Remove is used just because we are handling zeros on complete df)
for colName in colNames:
    df[colName] = df[colName].fillna(df[colName].mean())
print("Done ...")

#################################################
# For Linear Regression Check Correlation
#################################################
# check relation with corelation - table
print("\n*** Correlation Table ***")
pd.options.display.float_format = '{:,.3f}'.format
dfc = df.corr()
print(dfc)

#################################################
# Visual Data Anlytics
#################################################
# Heatmap (Used for Regression problem to plot correlation)
print("\n*** Heat Map ***")
plt.figure(figsize=(8,8))
ax = sns.heatmap(df.corr(), annot=True, cmap="PiYG")
# data.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()

# Pair plot 
print("\n*** Pair Plot ***")
plt.figure()
sns.pairplot(df, height=2)
plt.show()

# Boxplot (All columns in same figure, not aplicable for more columns)
print("\n*** Box Plot ***")
plt.figure(figsize=(10,5))
sns.boxplot(data=df)
plt.show()

# Boxplot 
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
for colName in colNames:
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# Histograms
print("\n*** Histogram Plot ***")
colNames = df.columns.tolist()
colNames.remove(clsVars)
print('Histograms')
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# Scatterplots (Used for Regression problem)
print('\n*** Scatterplot ***')
colNames = df.columns.tolist()
colNames.remove(depVars)
print(colName)
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.regplot(data=df, x=depVars, y=colName, color= 'b', scatter_kws={"s": 5})
    plt.title(depVars + ' v/s ' + colName)
    plt.show()

# Distribution plot / class count plot (Used for categorical columns)
colNames = ["ocean_proximity"]
print("\n*** Distribution Plot ***")
for colName in colNames:
    plt.figure()
    sns.countplot(df[colName],label="Count")
    plt.title(colName)
    plt.show()

# Check class (Used for classification problem)
print("\n*** Group Counts ***")
print(df.groupby(clsVars).size())
print("")

# Distribution plot / class count plot (Used for categorical columns, in classification problem used for class variable)
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df[clsVars],label="Count")
plt.title('Class Variable')
plt.show()


#################################################
# For Linear Regression
#################################################
#################################################
# Ordinary Least Square Creation & Fitting 
#################################################
# all cols except dep var 
print("\n*** OLS Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(depVars)
print(allCols)

# ols summary 
print("\n*** Regression Summary ***")
import statsmodels.api as sm
X = sm.add_constant(df[allCols])
y = df[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# check Adj R-Sq

# remove columns with p-value > 0.05
# chnage as require
print("\n*** Drop Cols ***")
print(allCols)
allCols.remove('random_income')
print(allCols)

# regression summary for feature
print("\n*** Regression Summary Again ***")
X = sm.add_constant(df[allCols])
y = df[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

#################################################
# Split Train & Test | Create Regression Model
#################################################
# split into data & target
print("\n*** Prepare Data ***")
dfTrain = df.sample(frac=0.8, random_state=707)
dfTest = df.drop(dfTrain.index)
print("Train Count:",len(dfTrain.index))
print("Test Count :",len(dfTest.index))

# train data
print("\n*** Regression Data For Train ***")
X_train = dfTrain[allCols].values
y_train = dfTrain[depVars].values
# print
print(X_train.shape)
print(y_train.shape)
print(type(X_train))
print(type(y_train))
print("Done ...")

# create model
print("\n*** Regression Model ***")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
print(model)
print("Done ...")

#################################################
# predict with train data - academic purpose only
#################################################
# predict
print("\n*** Predict - Train Data ***")
p_train = model.predict(X_train)
dfTrain['predict'] = p_train
print("Done ...")

#################################################
# Model Evaluation Train Data - academic purpose only
#################################################
# visualize 
print("\n*** Scatter Plot ***")
plt.figure()
#sns.regplot(data=dfTrain, x=depVars, y='predict', color='b', scatter_kws={"s": 5})
sns.regplot(x=y_train, y=p_train, color='b', scatter_kws={"s": 5})
plt.show()

# R-Square
print('\n*** R-Square ***')
from sklearn.metrics import r2_score
#r2 = r2_score(dfTrain[depVars], dfTrain['predict'])
r2 = r2_score(y_train, p_train)
print(r2)

# adj r-square  
print('\n*** Adj R-Square ***')
adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))
print(adj_r2)

# mae 
print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_train, p_train)
print(mae)

# mse 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_train, p_train)
print(mse)
   
# rmse 
#https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# scatter index (SI) is defined to judge whether RMSE is good or not. 
# SI=RMSE/measured data mean. 
# If SI is less than one, your estimations are acceptable.
print('\n*** Scatter Index ***')
si = rmse/y_train.mean()
print(si)

#################################################
# confirm with test data 
#################################################
# test data
print("\n*** Regression Data For Test ***")
print(allCols)
# split
X_test = dfTest[allCols].values
y_test = dfTest[depVars].values
# print
print(X_test.shape)
print(y_test.shape)
print(type(X_test))
print(type(y_test))
print("Done ...")

# predict
print("\n*** Predict - Test Data ***")
p_test = model.predict(X_test)
dfTest['predict'] = p_test
print("Done ...")

#################################################
# Model Evaluation - Test Data
#################################################
# visualize 
print("\n*** Scatter Plot ***")
plt.figure()
#sns.regplot(data=dfTest, x=depVars, y='predict', color='b', scatter_kws={"s": 5})
sns.regplot(x=y_test, y=p_test, color='b', scatter_kws={"s": 5})
plt.show()

# R-Square
print('\n*** R-Square ***')
from sklearn.metrics import r2_score
r2 = r2_score(y_train, p_train)
print(r2)

# adj r-square  
print('\n*** Adj R-Square ***')
adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))
print(adj_r2)

# mae 
print("\n*** Mean Absolute Error ***")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, p_test)
print(mae)

# mse 
print("\n*** Mean Squared Error ***")
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, p_test)
print(mse)
   
# rmse 
# RMSE measures the error.  How good is an error depends on the amplitude of your data. 
# RMSE should be less 10% for mean(depVars)
print("\n*** Root Mean Squared Error ***")
rmse = np.sqrt(mse)
print(rmse)

# scatter index
# scatter index less than 1; the predictions are decent
print('\n*** Scatter Index ***')
si = rmse/y_test.mean()
print(si)

##############################################################
# predict from new data 
##############################################################
# regression summary for feature - just fyi reminderr
print("\n*** Regression Summary Again ***")
X = sm.add_constant(df[allCols])
y = df[depVars]
OlsSmry = sm.OLS(y, X)
LRModel = OlsSmry.fit()
print(LRModel.summary())

# now create linear regression model
print("\n*** Regression Model ***")
X = df[allCols].values
y = df[depVars].values
model = LinearRegression()
model.fit(X,y)
print(model)

# read dataset
dfp = pd.read_csv('./data/california-housing-prd.csv')

print("\n*** Structure ***")
print(dfp.info())

# not required
print("\n*** Drop Cols ***")
print("N/A ... ")

# transformation
# change as required
print("\n*** Transformation ***")
dfp['ocean_proximity'] = leOpr.transform(dfp['ocean_proximity'])
print("Done ... ")

# handle normalization if required
print('\n*** Normalize Data ***')
dfp = utils.NormalizeData(dfp, ['ser',depVars])
print('Done ...')

# check nulls
print('\n*** Columns With Nulls ***')
print(dfp.isnull().sum()) 
print("Done ... ")

# split X & y
print("\n*** Split Predict Data ***")
print(allCols)
print(depVars)
X_pred = dfp[allCols].values
y_pred = dfp[depVars].values
print(X_pred)
print(y_pred)

# predict
print("\n*** Predict Data ***")
p_pred = model.predict(X_pred)
dfp['predict'] = p_pred
print("Done ... ")

# show predicted values
print("\n*** Print Predict Data ***")
for idx in dfp.index:
     print(dfp['ser'][idx], dfp['predict'][idx])
print("Done ... ")


#################################################
# For Classification
#################################################
#################################################
# set X & y
#################################################
# split into data & target
print("\n*** Prepare Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(clsVars)
print(allCols)
X = df[allCols].values
y = df[clsVars].values

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

# head
print("\n*** Prepare Data - Head ***")
print(X[0:4])
print(y[0:4])

#################################################
# Split Train & Test
#################################################
# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,           #separating train and test data X=all columns & y=class variable column
                                test_size=0.2, random_state=707)

# shapes
print("\n*** Train & Test Data ***")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))

#################################################
# actual model ... create ... fit ... predict
#################################################
# import all model & metrics
print("\n*** Importing Models ***")
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import GaussianNB
print("Done ...")

# classifier object
print("\n*** Classfier Model ***")
#model = KNeighborsClassifier()
#model = LogisticRegression(random_state=707)
#model = DecisionTreeClassifier(random_state=707)
#model = GaussianNB()
#model = SVC(random_state=707)
model = RandomForestClassifier(random_state=707)
print(model)
# fit the model
model.fit(X_train, y_train)                # create model & fit data with the train data created
print("Done ...")

#################################################
# Classification  - Predict Train
# evaluate : Accuracy & Confusion Metrics
#################################################
# classifier object
print("\n*** Predict Train ***")
# predicting the Test set results
p_train = model.predict(X_train)            # use model ... predict
print("Done ...")

# accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train, p_train)*100
print("\n*** Accuracy ***")
print(accuracy)

# confusion matrix for actual data
# X-axis Actual | Y-axis Actual - to see how cm of original is
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_train)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix for predicted data
# X-axis Predicted | Y-axis Actual
cm = confusion_matrix(y_train, p_train)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
from sklearn.metrics import classification_report
print("\n*** Classification Report ***")
cr = classification_report(y_train,p_train)
print(cr)

# make dftrain --> we are separating training dataframe to understand what data was used for training the model
# not to be done in production
print("\n*** Recreate Train ***")
dfTrain =  pd.DataFrame(data = X_train)
dfTrain.columns = allCols
dfTrain[clsVars] = y_train
dfTrain['Predict'] = p_train
dfTrain[clsVars] = le.inverse_transform(dfTrain[clsVars])
dfTrain['Predict'] = le.inverse_transform(dfTrain['Predict'])
print("Done ...")

#################################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
#################################################
# classifier object
print("\n*** Predict Test ***")
# predicting the Test set results
p_test = model.predict(X_test)            # use model ... predict
print("Done ...")

# accuracy
accuracy = accuracy_score(y_test, p_test)*100
print("\n*** Accuracy ***")
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
cm = confusion_matrix(y_test, y_test)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix
# X-axis Predicted | Y-axis Actual
cm = confusion_matrix(y_test, p_test)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_test,p_test)
print(cr)

# make dftest ---> only for showing and testing if model is predicting properly and we understand with which data we tested the model
# not to be done in production
print("\n*** Recreate Test ***")
dfTest =  pd.DataFrame(data = X_test)
dfTest.columns = allCols
dfTest[clsVars] = y_test
dfTest['Predict'] = p_test
dfTest[clsVars] = le.inverse_transform(dfTest[clsVars])
dfTest['Predict'] = le.inverse_transform(dfTest['Predict'])
print("Done ...")

#################################################
# Production Data
#################################################
# Final Prediction with Production data i.e.we are now deploying the model for live data
# Create model Object from whole data
# Predict Species --> In production data we don't have any values in class variable column
#################################################
# classifier object
print("\n*** Classfier Model ***")
#model = KNeighborsClassifier()
#model = LogisticRegression(random_state=707)
#model = DecisionTreeClassifier(random_state=707)
#model = GaussianNB()
#model = SVC(random_state=707)
model = RandomForestClassifier(random_state=707)
print(model)
# fit the model
model.fit(X, y)                # create model ... fit data
print("Done ...")

# read dataset --> we need to read the production data file assuming this is the live data coming in to model
print("\n*** Read Data For Prediction ***")
dfp = pd.read_csv('./data/iris-m-prd.csv')
print(dfp.head())

# not required
print("\n*** Data For Prediction - Drop Cols***")
print("N/A ...")

# convert string / categoric to numeric
print("\n*** Data For Prediction - Class Vars ***")
dfp['Species'] = dfp['Species'].str.title()
dfp['Species'] = dfp['Species'].str.strip()
print(dfp[clsVars].unique())
dfp[clsVars] = le.transform(dfp[clsVars])
print(dfp[clsVars].unique())

# change as required ... same transformtion as done for main data
print("\n*** Data For Prediction - Transform***")
print("None ...")

# check nulls
print('\n*** Data For Prediction - Columns With Nulls ***')
print(dfp.isnull().sum()) 
print("Done ... ")

# Handle nulls if any for column where we find null values
print('\n*** Data For Prediction - Handle Columns With Nulls ***')
dfp['PetalWidth'] = dfp['PetalWidth'].fillna(df['PetalWidth'].mean())
print("Done ... ")

# split into data & outcome
print("\n*** Data For Prediction - X & y Split ***")
print(allCols)
print(clsVars)
X_pred = dfp[allCols].values
y_pred = dfp[clsVars].values
print(X_pred)
print(y_pred)

# predict from model
print("\n*** Prediction ***")
p_pred = model.predict(X_pred)
# actual
print("Actual")
print(y_pred)
# predicted
print("Predicted")
print(p_pred)

# accuracy
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_pred, p_pred)*100
print(accuracy)

# confusion matrix - actual
cm = confusion_matrix(y_pred, y_pred)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix - predicted
cm = confusion_matrix(y_pred, p_pred)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_pred, p_pred)
print(cr)

# update data frame -->we are updating the production data file with predicted values
print("\n*** Update Predict Data ***")
dfp['Predict'] = p_pred
dfp[clsVars] = le.inverse_transform(dfp[clsVars])
dfp['Predict'] = le.inverse_transform(dfp['Predict'])
print("Done ...")


#################################################
# For Clustering
#################################################
#################################################
# Prepare Data ---> In clustering we don't have class variable i.e. no labels are present
#################################################
# split into data & target
print("\n*** Prepare Data ***")
allCols = df.columns.tolist()
print(allCols)
X = df[allCols].values

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(type(X))

# head
print("\n*** Prepare Data - Head ***")
print(X[0:4])

#################################################
# Knn Clustering
#################################################

# imports
from sklearn.cluster import KMeans

# how to decide on the clusters ---> within cluster sum of squares errors - wcsse
# elbow method ... iterations should be more than 10
print("\n*** Compute WCSSE ***")
vIters = 20
lWcsse = []
for i in range(1, vIters):
    kmcModel = KMeans(n_clusters=i)
    kmcModel.fit(X)
    lWcsse.append(kmcModel.inertia_)
for vWcsse in lWcsse:
    print(vWcsse)

# plotting the results onto a line graph, allowing us to observe 'The elbow'
print("\n*** Plot WCSSE ***")
plt.figure()
plt.plot(range(1, vIters), lWcsse)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSSE') #within cluster sum of squares error
plt.show()

# programatically
#!pip install kneed --->Kneed is used to find the best fit K 
print("\n*** Find Best K ***")
import kneed
kl = kneed.KneeLocator(range(1, vIters), lWcsse, curve="convex", direction="decreasing")
vBestK = kl.elbow
print(vBestK)

# k means cluster model -->we are fitting this model with our data which is stored in X
print("\n*** Model Create & Train ***")
model = KMeans(n_clusters=vBestK, random_state=707)
model.fit(X)

# result
print("\n*** Model Results ***")
print(model.labels_)
df['PredKnn'] = model.labels_

# counts for knn
print("\n*** Counts For Knn ***")
print(df.groupby(df['PredKnn']).size())

# class count plot
print("\n*** Distribution Plot - KNN ***")
plt.figure()
sns.countplot(data=df, x='PredKnn', label="Count")
plt.title('Distribution Plot - KNN')
plt.show()

################################
# Hierarchical Clustering
###############################

# linkage
print("\n*** Linkage Method ***")
from scipy.cluster import hierarchy as hac
vLinkage = hac.linkage(df, 'ward')
print("Done ...")

# # make the dendrogram
# print("\n*** Plot Dendrogram ***")
# print("Looks Cluttered")
# plt.figure(figsize=(8,8))
# hac.dendrogram(vLinkage, 
#                orientation='left')
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Index')
# plt.ylabel('Linkage (Ward)')
# plt.show

# # make the dendrogram - large so readable
# # make the dendrogram
# print("\n*** Plot Dendrogram ***")
# print("No Groups")
# plt.figure(figsize=(8,80))
# hac.dendrogram(vLinkage, 
#                leaf_font_size=10.,
#                orientation='left')
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Index')
# plt.ylabel('Linkage (Ward)')
# plt.show

# # make the dendrogram - truncated
# # make the dendrogram
# print("\n*** Plot Dendrogram ***")
# print("With Groups")
# plt.figure(figsize=(8,10))
# hac.dendrogram(vLinkage,
#                truncate_mode='lastp',   # show only the last p merged clusters
#                p=5,                     # p number of clusters
#                leaf_font_size=12.,
#                show_contracted=True,    # to get a distribution impression in truncated branches
#                orientation='left'       # left to right
#                )
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Index')
# plt.ylabel('Linkage (Ward)')
# plt.show

# create cluster model
print("\n*** Agglomerative Clustering ***")
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=vBestK, affinity='euclidean', linkage='ward')  
# train and group together
lGroups = model.fit_predict(df)
print(lGroups)
# update data frame
df['PredHeir'] = lGroups
print("Done ...")

# counts for heir
print("\n*** Counts For Heir ***")
print(df.groupby(df['PredHeir']).size())

# class count plot
print("\n*** Distribution Plot - Heir ***")
plt.figure(),
sns.countplot(data=df, x='PredHeir', label="Count")
plt.title('Distribution Plot - Heir')
plt.show()

# counts for knn
print("\n*** Counts For Knn ***")
print(df.groupby(df['PredKnn']).size())

# counts for heir
print("\n*** Counts For Heir ***")
print(df.groupby(df['PredHeir']).size())





