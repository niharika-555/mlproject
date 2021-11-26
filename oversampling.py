# IMPORT PACKAGES
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df=df.drop(columns=["id"],axis=1)
df['gender'] = df['gender'].replace(['Male','Female','Other'],[0,1,2])
df['ever_married'] = df['ever_married'].replace(['Yes','No'],[0,1])
df['work_type'] = df['work_type'].replace(['Private','Self-employed','Govt_job','children','Never_worked'],[0,1,2,3,4])
df['Residence_type'] = df['Residence_type'].replace(['Urban','Rural'],[0,1])
df['smoking_status'] = df['smoking_status'].replace(['formerly smoked','never smoked','smokes','Unknown'],[0,1,2,3])
df['bmi'] = df['bmi'].fillna((df['bmi'].mean()))
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
#saving file to csv file
# df.to_csv('stroke_file.csv')
#Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# define oversampling strategy
SMOTE = SMOTE()
# df=df.
# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))

#PART 2
# # import SVM libraries
# from sklearn.svm import SVC
#
#from sklearn.metrics import classification_report, roc_auc_score,accuracy_score
#
# model=SVC()
# y_test
# clf_SMOTE = model.fit(X_train_SMOTE, y_train_SMOTE)
# pred_SMOTE = clf_SMOTE.predict(X_test)
# acc_km = accuracy_score(y_test, pred_SMOTE)
#
# print("ROC AUC score for oversampled SMOTE data: ", roc_auc_score(y_test, pred_SMOTE))
