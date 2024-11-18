'''
Battery_Power: Total energy a battery can store in one time measured in mAh
Clock_Speed: speed at which microprocessor executes instructions
FC: Front Camera mega pixels
Int_Memory: Internal Memory in Gigabytes
Mobile_D: Mobile Depth in cm
Mobile_W: Mobile Weight
Cores: Number of cores of processor
PC: Primary Camera mega pixels
Pixel_H: Pixel Resolution Height
Pixel_W: Pixel Resolution Width
Ram: Random Access Memory in Megabytes
Screen_H: Screen Height of mobile in cm
Screen_W: Screen Width of mobile in cm
Talk_Time: Longest time that a single battery charge will last when you are
Four_G: Has Four_G or not
Three_G: Has Three_G or not
Touch_Screen: Has Touch Screen or not
Dual_SIM: Has Dual SIM support or not
Bluetooth: Has Bluetooth or not
WiFi: Has WiFi or not
Price_Range: ??? ( Output Variable)
'''

import pandas as pd # for dataset manipulation
import numpy as np # # for arithmetic calculations
import matplotlib.pyplot as plt # for visualisation
import seaborn as sns # for visualisation
import os
import joblib
import pickle

from sqlalchemy import create_engine
from sklearn.tree import plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Importing Mobile_df dataset using pandas
MobileData = pd.read_csv(r"C:\Users\Frenn\Desktop\UE\Data Engineering\Project\train.csv")

user = 'root' #userid
db = 'mobile_db' # databasename
pw = 'password' #password

engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
MobileData.to_sql('mobile_table', con = engine, if_exists = 'replace', chunksize = 1000 , index = False) #.lower() is optional

sql = 'select * from mobile_table;'

Mobile_df = pd.read_sql_query(sql, engine)
Mobile_df.head()
Mobile_df.columns

Mobile_df.info()

def checkNull (d):    
    is_Null = d.isnull ().sum ().to_frame (name = 'is_Null').T
    is_Na = d.isna ().sum ().to_frame (name = 'is_Na').T
    Unique = d.nunique ().to_frame (name = 'Unique').T
    return pd.concat ([Unique , is_Na , is_Null]) 

checknull_values = checkNull (Mobile_df).T
checknull_values.to_csv("Null.csv")

print ("Duplicate values in Mobile_df are:" , Mobile_df.duplicated ().sum ())


Mobile_df.describe()
detailed_summary_stats = Mobile_df.describe (include = "all").T
detailed_summary_stats.to_csv('detailed_summary_stats.csv')

# Missing values
Mobile_df.isnull().sum()



plt.figure (figsize = [18,12] ,  dpi = 150 )
plt.title ("Correlation Graph" , fontsize = 14)
matrix = np.triu (Mobile_df.corr ())
sns.heatmap (Mobile_df.corr(), annot = True, cmap = 'coolwarm', mask = matrix , cbar = False)
plt.savefig ("Corrilation_Matrix.png")
plt.show ()


plt.figure (figsize = (3 , 8) , dpi = 100)
heatmap = sns.heatmap (Mobile_df.corr()[['Price_Range']].sort_values (by = 'Price_Range', ascending = False), vmin = -1, vmax = 1, annot = True, cmap = 'coolwarm')
heatmap.set_title ('Features Correlating with Price Range', fontdict = {'fontsize':16} , pad = 10);
plt.savefig ("Corrilation_Range.png")


'''### Seperate Numeric and Category Columns###'''
numeric_features = ['Battery_Power', 'Clock_Speed', 'FC', 'Int_Memory', 'Mobile_D', 'Mobile_W', 'Cores', 'PC', 'Pixel_H', 'Pixel_W', 'Ram','Screen_H', 'Screen_W', 'Talk_Time']
category_features = ['Four_G', 'Three_G', 'Touch_Screen', 'Dual_SIM', 'Bluetooth', 'WiFi']



'''### Histogram ###'''
Mobile_df.hist(figsize=(15, 15), color = 'Orange')
plt.subplots_adjust(wspace=0.5, hspace=1)  # Increase spacing between subplots
plt.savefig ("Histogram.png")
plt.show()



'''### Box Plot###'''
# Plotting the box plots with thicker lines and filled color
fig, axes = plt.subplots(nrows=1, ncols=len(numeric_features), figsize=(18, 6), sharey=False)

for i, ax in enumerate(axes):
    Mobile_df[numeric_features[i]].plot(kind='box', ax=ax, vert=True, patch_artist=True,
                                        boxprops=dict(facecolor='orange', color='black'),
                                        medianprops=dict(color='black', linewidth=1),
                                        whiskerprops=dict(color='black', linewidth=1),
                                        capprops=dict(color='black', linewidth=1),
                                        flierprops=dict(marker='o', color='orange', alpha=0.5))

plt.subplots_adjust(wspace=2)  # Adjust the width of the padding between subplots
plt.savefig("Boxplot.png")
plt.show()


'''### Features with Outliers###'''
plt.boxplot(Mobile_df.FC)
plt.boxplot(Mobile_df.Pixel_H)
plt.boxplot(Mobile_df.Screen_W)


FC_IQR = Mobile_df.FC.quantile(0.75) - Mobile_df.FC.quantile(0.25) 
FC_Lower_Limit = Mobile_df.FC.quantile(0.25) - (FC_IQR * 1.5) 
FC_Upper_Limit = Mobile_df.FC.quantile(0.75) + (FC_IQR * 1.5) 


Pixel_H_IQR = Mobile_df.Pixel_H.quantile(0.75) - Mobile_df.Pixel_H.quantile(0.25) 
Pixel_H_Lower_Limit = Mobile_df.Pixel_H.quantile(0.25) - (Pixel_H_IQR * 1.5) 
Pixel_H_Upper_Limit = Mobile_df.Pixel_H.quantile(0.75) + (Pixel_H_IQR * 1.5) 


Screen_W_IQR = Mobile_df.Screen_W.quantile(0.75) - Mobile_df.Screen_W.quantile(0.25) 
Screen_W_Lower_Limit = Mobile_df.Screen_W.quantile(0.25) - (Screen_W_IQR * 1.5) 
Screen_W_Upper_Limit = Mobile_df.Screen_W.quantile(0.75) + (Screen_W_IQR * 1.5) 


Outlier_FC = np.where(Mobile_df.FC > FC_Upper_Limit, True, np.where(Mobile_df.FC < FC_Lower_Limit , True, False))
print(sum(Outlier_FC))

Outlier_Pixel_H = np.where(Mobile_df.Pixel_H > Pixel_H_Upper_Limit, True, np.where(Mobile_df.Pixel_H < Pixel_H_Lower_Limit , True, False))
print(sum(Outlier_Pixel_H))

Outlier_Screen_W = np.where(Mobile_df.Screen_W > Screen_W_Upper_Limit, True, np.where(Mobile_df.Screen_W < Screen_W_Lower_Limit , True, False))
print(sum(Outlier_Screen_W))


## Auto EDA
# D-tale
##########
import dtale
dtale_df = dtale.show(Mobile_df)
dtale_df.open_browser()


''' #### Create an Imputaion method and Windsorization in case of outliers ####'''
'''num_pipeline = Pipeline(steps = 
                        [('impute', SimpleImputer(strategy = 'mean')), 
                         ('scale', MinMaxScaler())])
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])
imp_enc_scale = preprocessor.fit(Input_Columns)
joblib.dump(imp_enc_scale, 'NumPipeline') #Save the imputation model using joblib
Mobile_df = pd.DataFrame(imp_enc_scale.transform(Input_Columns), columns = imp_enc_scale.get_feature_names_out())

#cat_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'most_frequent'))])
#preprocessor = ColumnTransformer(transformers = [('num', cat_pipeline, category_feature)])

cat_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent'))])
preprocessor = ColumnTransformer(transformers=[('cat_imputation', cat_pipeline, category_features)])

preprocessor.fit(Mobile_df)
Mobile_df_transformed = preprocessor.transform(Mobile_df)
joblib.dump(preprocessor, 'CatPipeline')
print(Mobile_df_transformed)  '''


'''#### Define pipelines for numeric and categorical features ####'''
num_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean')), ('scale', MinMaxScaler())])
cat_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent'))])

# Define the ColumnTransformer
preprocessor = ColumnTransformer(transformers=[('num', num_pipeline, numeric_features), ('cat', cat_pipeline, category_features)])
preprocessor.fit(Mobile_df) # Fit the preprocessor on the DataFrame
transformed_names = preprocessor.get_feature_names_out()
Mobile_df_transformed = pd.DataFrame(preprocessor.transform(Mobile_df), columns=transformed_names) # Transform the DataFrame
# Save the preprocessor for future use
joblib.dump(preprocessor, 'Num_CatPipeline')  # This is the correct pipeline to save
print(Mobile_df_transformed) # Print the transformed DataFrame



'''#### Outlier treatment for Numeric columns ####'''
columns_to_transform = ['num__Battery_Power', 'num__Clock_Speed', 'num__FC', 'num__Int_Memory', 'num__Mobile_D',
                        'num__Mobile_W', 'num__Cores', 'num__PC', 'num__Pixel_H', 'num__Pixel_W', 'num__Ram',
                        'num__Screen_H', 'num__Screen_W', 'num__Talk_Time']

winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = columns_to_transform)
outlier = winsor.fit(Mobile_df_transformed[columns_to_transform])
# Save the winsorizer model 
joblib.dump(outlier, 'winsor')
print(Mobile_df.head())
Mobile_df_transformed[columns_to_transform] = outlier.transform(Mobile_df_transformed[columns_to_transform])                                                                 



'''### Boxplot after outlier treatment###'''
fig, axes = plt.subplots(nrows=1, ncols=len(numeric_features), figsize=(25, 6), sharey=False)

for i, ax in enumerate(axes):
    Mobile_df_transformed[columns_to_transform[i]].plot(kind='box', ax=ax, vert=True, patch_artist=True,
                                        boxprops=dict(facecolor='lightgreen', color='black'),
                                        medianprops=dict(color='black', linewidth=1),
                                        whiskerprops=dict(color='black', linewidth=1),
                                        capprops=dict(color='black', linewidth=1),
                                        flierprops=dict(marker='o', color='lightgreen', alpha=0.5))

plt.subplots_adjust(wspace=2)  # Adjust the width of the padding between subplots
plt.savefig("Boxplot_NoOutliers.png")
plt.show()


Outlier_FC = np.where(Mobile_df_transformed.num__FC > FC_Upper_Limit, True, np.where(Mobile_df_transformed.num__FC < FC_Lower_Limit , True, False))
print(sum(Outlier_FC))

Outlier_Pixel_H = np.where(Mobile_df_transformed.num__Pixel_H > Pixel_H_Upper_Limit, True, np.where(Mobile_df_transformed.num__Pixel_H < Pixel_H_Lower_Limit , True, False))
print(sum(Outlier_Pixel_H))

Outlier_Screen_W = np.where(Mobile_df_transformed.num__Screen_W > Screen_W_Upper_Limit, True, np.where(Mobile_df_transformed.num__Screen_W < Screen_W_Lower_Limit , True, False))
print(sum(Outlier_Screen_W))


'''### Convert the output variable to string''' 
Target = Mobile_df['Price_Range']
Target = Target.astype(str)
Target = Target.map({'0': 'Low', '1': 'Medium-Low', '2': 'Medium-High', '3': 'High'})
print(Target)


''' #### Splitting data into training and testing data set ####'''
x_train, x_test, y_train, y_test = train_test_split(Mobile_df_transformed, Target, test_size = 0.25, random_state= 42, stratify = Target )
print ("Number of rows in train data =" , x_train.shape [0])
print ("Number of rows in test data =" , x_test.shape [0])



'''### Machine Learning DT_Models  ###'''
'''### KNN ###'''
# Initialize the KNN model
KNN_Model = KNeighborsClassifier(n_neighbors=5)  # You can change the number of neighbors as needed

# Fit the model on the training data
KNN_Model.fit(x_train, y_train)

# Prediction on Train Data
predictions_train = KNN_Model.predict(x_train)
pd.crosstab(y_train, predictions_train, rownames=['Actual'], colnames=['Predictions'])

# Train Data Accuracy
train_accuracy = accuracy_score(y_train, predictions_train)
print("KNN Training Data Accuracy: ", train_accuracy) # Training Data Accuracy:  0.628

# Prediction on Test Data
predictions_test = KNN_Model.predict(x_test)
pd.crosstab(y_test, predictions_test, rownames=['Actual'], colnames=['Predictions'])

# Test Data Accuracy
test_accuracy = accuracy_score(y_test, predictions_test)
print("Test Data Accuracy: ", test_accuracy) # Test Data Accuracy:  0.418

# Print the accuracies
print("KNN Test Data Accuracy: ", test_accuracy)
print("KNN Training Data Accuracy: ", train_accuracy)


# Confusion matrix for training data
cm_train = confusion_matrix(y_train, predictions_train)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.title('KNN_Confusion Matrix - Training Data')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.savefig ("KNN_Confusion Matrix - Training Data.png")
plt.show()

# Confusion matrix for testing data
cm_test = confusion_matrix(y_test, predictions_test)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('KNN_Confusion Matrix - Testing Data')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.savefig ("KNN_Confusion Matrix - Testing Data.png")
plt.show()

# Accuracy bar plot
accuracies = [train_accuracy, test_accuracy]
labels = ['Training Accuracy', 'Testing Accuracy']

plt.figure(figsize=(8, 5))
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylim(0, 1)
plt.title('KNN_Model Accuracy')
plt.xlabel('Data Set')
plt.ylabel('Accuracy')
plt.savefig ("KNN_Model Accuracy.png")
plt.show()


'''### Hyperparameter Tunning for KNN ###'''
# Define the parameter grid for KNN
param_grid = {
    'n_neighbors': range(1, 31),  # Number of neighbors to test
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric to use
}

# Initialize the KNN model
knn_model = KNeighborsClassifier()

# GridSearchCV with cross-validation
knn_gscv = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy', verbose=1)
knn_gscv.fit(x_train, y_train)  # Train

# Best parameters found by GridSearchCV
print("Best Parameters: ", knn_gscv.best_params_)  # Example: {'metric': 'minkowski', 'n_neighbors': 5, 'weights': 'uniform'}

# Best estimator (KNN model with best parameter values)
KNN_best = knn_gscv.best_estimator_

# Prediction on Train Data
hyp_predictions_train = KNN_best.predict(x_train)
print(pd.crosstab(y_train, hyp_predictions_train, rownames=['Actual'], colnames=['Predictions']))
KNN_hyp_train_accuracy_score = accuracy_score(y_train, hyp_predictions_train)
print("KNN Training Data Accuracy: ", KNN_hyp_train_accuracy_score)

# Prediction on Test Data
hyp_predictions_test = KNN_best.predict(x_test)
print(pd.crosstab(y_test, hyp_predictions_test, rownames=['Actual'], colnames=['Predictions']))
KNN_hyp_test_accuracy_score = accuracy_score(y_test, hyp_predictions_test)
print("KNN Test Data Accuracy: ", KNN_hyp_test_accuracy_score)


# Confusion matrix for training data
cm_train = confusion_matrix(y_train, hyp_predictions_train)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.title('KNN_Hyp_Confusion Matrix - Training Data')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.savefig ("KNN_Hyp_Confusion Matrix - Training Data.png")
plt.show()

# Confusion matrix for testing data
cm_test = confusion_matrix(y_test, predictions_test)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('KNN_Hyp_Confusion Matrix - Testing Data')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.savefig ("KNN_Hyp_Confusion Matrix - Testing Data.png")
plt.show()


# Accuracy bar plot
accuracies = [KNN_hyp_train_accuracy_score, KNN_hyp_test_accuracy_score]
labels = ['KNN Hyp Training Accuracy', 'KNN Hyp Testing Accuracy']

plt.figure(figsize=(8, 5))
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylim(0, 1)
plt.title('KNN_Model Accuracy')
plt.xlabel('Data Set')
plt.ylabel('Accuracy')
plt.savefig ("KNN_Model_Hyp_Accuracy.png")
plt.show()



'''#### Random forest ####'''
from sklearn.ensemble import RandomForestClassifier as RF

RF_Classifier = RF(n_estimators=500, n_jobs=1, random_state=42)
RF_Classifier.fit(x_train, y_train)

confusion_matrix(y_test, RF_Classifier.predict(x_test)) #Matrix for test 
accuracy_score(y_test, RF_Classifier.predict(x_test)) # Test Data Accuracy is 0.88 
#Test data accuracy for decision tree was 0.82 and for Random forest is 0.88, accuracy has been increased by 0.6%.

confusion_matrix(y_train, RF_Classifier.predict(x_train)) #Matrix for train
accuracy_score(y_train, RF_Classifier.predict(x_train)) # Train Data Accuracy is 1.0

print("RF Test Data Accuracy: ", accuracy_score(y_test, RF_Classifier.predict(x_test)))
print("RF Trainning Data Accuracy: ", accuracy_score(y_train, RF_Classifier.predict(x_train)))
#Test data accuracy is 0.78 and train data accuracy is 1 so the DT_Model is overfitting

# Confusion matrix for training data
cm_train = confusion_matrix(y_train, RF_Classifier.predict(x_train))
plt.figure(figsize=(10, 7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.title('RF_Confusion Matrix - Training Data')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.savefig ("RF_Confusion Matrix - Training Data.png")
plt.show()

# Confusion matrix for testing data
cm_test = confusion_matrix(y_test, RF_Classifier.predict(x_test))
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('RF_Confusion Matrix - Testing Data')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.savefig ("RF_Confusion Matrix - Testing Data.png")
plt.show()

# Accuracy bar plot
train_accuracy = accuracy_score(y_train, RF_Classifier.predict(x_train))
test_accuracy = accuracy_score(y_test, RF_Classifier.predict(x_test))
accuracies = [train_accuracy, test_accuracy]
labels = ['Training Accuracy', 'Testing Accuracy']

plt.figure(figsize=(8, 5))
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylim(0, 1)
plt.title('RF_Model Accuracy')
plt.xlabel('Data Set')
plt.ylabel('Accuracy')
plt.savefig ("RF_Model Accuracy.png")
plt.show()



'''#### Hyperparameter Tuning for Random Forest ####'''
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)] # Number of trees in random forest
max_features = ['auto', 'sqrt'] # Number of features to consider at every split
max_depth = [2,4] # Maximum number of levels in tree
min_samples_split = [2, 5] # Minimum number of samples required to split a node
min_samples_leaf = [1, 2] # Minimum number of samples required at each leaf node
bootstrap = [True, False] # Method of selecting samples for training each tree

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)

RF = RandomForestClassifier()

# GridsearchCV with cross-validation to perform experiments with parameters set
rf_gscv = GridSearchCV(RF, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)
rf_gscv.fit(x_train, y_train) # Train

# Print the best parameters and best score
print("RF Best parameters found: ", rf_gscv.best_params_)
print("RF Best accuracy found: ", rf_gscv.best_score_)
rf_best = rf_gscv.best_estimator_ # Model with best parameter values
rf_best

best_tree = rf_best.estimators_[0] # Extract one of the trees (for example, the first tree)

# Define the predictors and target class names
predictors = list(x_train.columns)
target_class_names = y_train.unique().astype(str)

# Plot the decision tree using matplotlib
plt.figure(figsize=(40, 22))
plot_tree(best_tree, 
          filled=True, 
          rounded=True, 
          feature_names=x_train.columns, 
          class_names=np.unique(y_train).astype(str),
          fontsize=10)  # Adjust fontsize as needed for readability
plt.title("Random Forest - Decision Tree", fontsize=8)
plt.savefig("random_forest_tree.png")  # Save as PNG file
plt.show()

# Prediction on Train Data
hyp_preds_train = rf_best.predict(x_train)
hyp_preds_train
pd.crosstab(y_train, hyp_preds_train, rownames = ['Actual'], colnames = ['Predictions']) # Confusion Matrix

print('RF Accuracy on Train Data: ', accuracy_score(y_train, hyp_preds_train)) # Accuracy

# Prediction on Test Data
hyp_preds_test = rf_best.predict(x_test)
hyp_preds_test
pd.crosstab(y_test, hyp_preds_test, rownames = ['Actual'], colnames= ['Predictions'])
# Accuracy
print('RF Accuracy on Test Data: ', accuracy_score(y_test, hyp_preds_test))


# Confusion matrix for training data
cm_train = confusion_matrix(y_train, hyp_preds_train)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.title('RF_Hyp_Confusion Matrix - Training Data')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.savefig ("RF_Hyp_Confusion Matrix - Training Data.png")
plt.show()

# Confusion matrix for testing data
cm_test = confusion_matrix(y_test, hyp_preds_test)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title('RF_Hyp_Confusion Matrix - Testing Data')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.savefig ("RF_Hyp_Confusion Matrix - Testing Data.png")
plt.show()

# Accuracy bar plot
train_accuracy = accuracy_score(y_train, hyp_preds_train)
test_accuracy = accuracy_score(y_test, hyp_preds_test)
accuracies = [train_accuracy, test_accuracy]
labels = ['Training Accuracy', 'Testing Accuracy']

plt.figure(figsize=(8, 5))
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylim(0, 1)
plt.title('RF_Hyp_Model Accuracy')
plt.xlabel('Data Set')
plt.ylabel('Accuracy')
plt.savefig ("RF_Hyp_Model Accuracy.png")
plt.show()


### Save the Best Model with pickel library
pickle.dump(KNN_best, open('KNN.pkl', 'wb'))
pickle.dump(rf_best, open('RF.pkl', 'wb'))
#pickle.dump(DT_best, open('DT.pkl', 'wb'))

os.getcwd()

