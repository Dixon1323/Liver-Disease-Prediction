import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import warnings
import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.rcParams['figure.dpi'] = 300

from joblib import dump

from imblearn.pipeline import Pipeline


from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm, tree

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,f1_score


train_df = pd.read_csv("//Users/dixonsmac/Desktop/Projects_2024/Liver_Syrosis/Liver-Disease-Prediction/Dataset/train_dataset.csv")
test_df = pd.read_csv("/Users/dixonsmac/Desktop/Projects_2024/Liver_Syrosis/Liver-Disease-Prediction/Dataset/test_dataset.csv")
print(train_df.head(15))

# print("============= TRAIN SET =============")
# print("Missing Values: " ,np.sum(train_df.isnull().any(axis=1)))
# print("Shape: ", train_df.shape)
# print("Columns: ", len(train_df.columns))
# print("Data types:")
# print(train_df.dtypes)
# print("============= TEST SET =============")
# print("Missing Values: ",(np.sum(test_df.isnull().any(axis=1))))
# print("Shape: ", test_df.shape)
# print("Columns: ", len(test_df.columns))
# print("Data types:")
# print(test_df.dtypes)

print("*" * 90)
print("after dropping")
train_df = train_df.drop(columns=['ID', 'N_Days'])
test_df = test_df.drop(columns=['ID', 'N_Days'])


train_df.isnull()
test_df.isnull()
train_df.describe()
train_df.info()

duplicate_train_df = train_df[train_df.duplicated()]
duplicate_train_df.shape
duplicate_test_df = test_df[test_df.duplicated()]
duplicate_test_df.shape
percent_missing1 = round((train_df.isnull().mean() * 100),2)
percent_missing2 = round((test_df.isnull().mean() * 100),2)
missing_value_df1 = pd.DataFrame({'column_name': train_df.columns,
                                 'percent_missing': percent_missing1})
missing_value_df2 = pd.DataFrame({'column_name': test_df.columns,
                                 'percent_missing': percent_missing2})
missing_value_df1
missing_value_df2
test_df = test_df.bfill()
test_df.dropna()
test_df.info()
le = LabelEncoder()
train_df.Status = le.fit_transform(train_df.Status)
train_df.Drug = le.fit_transform(train_df.Drug)
train_df.Sex = le.fit_transform(train_df.Sex)
train_df.Cholesterol = le.fit_transform(train_df.Cholesterol)
train_df.Ascites = le.fit_transform(train_df.Ascites)
train_df.Hepatomegaly = le.fit_transform(train_df.Hepatomegaly)
train_df.Spiders = le.fit_transform(train_df.Spiders)
train_df.Edema = le.fit_transform(train_df.Edema)
train_df.Albumin = le.fit_transform(train_df.Albumin)
train_df.Copper= le.fit_transform(train_df.Copper)
train_df.Alk_Phos= le.fit_transform(train_df.Alk_Phos)
train_df.SGOT= le.fit_transform(train_df.SGOT)
train_df.Tryglicerides= le.fit_transform(train_df.Tryglicerides)
train_df.Platelets = le.fit_transform(train_df.Platelets)
train_df.Prothrombin = le.fit_transform(train_df.Prothrombin)
test_df.Status = le.fit_transform(test_df.Status)
test_df.Drug = le.fit_transform(test_df.Drug)
test_df.Sex = le.fit_transform(test_df.Sex)
test_df.Ascites = le.fit_transform(test_df.Ascites)
test_df.Hepatomegaly = le.fit_transform(test_df.Hepatomegaly)
test_df.Spiders = le.fit_transform(test_df.Spiders)
test_df.Edema = le.fit_transform(test_df.Edema)
test_df.Albumin = le.fit_transform(test_df.Albumin)
test_df.Copper= le.fit_transform(test_df.Copper)
test_df.Alk_Phos= le.fit_transform(test_df.Alk_Phos)
test_df.SGOT= le.fit_transform(test_df.SGOT)
test_df.Tryglicerides= le.fit_transform(test_df.Tryglicerides)
test_df.Platelets = le.fit_transform(test_df.Platelets)
test_df.Prothrombin = le.fit_transform(test_df.Prothrombin)
train_df['Age']= train_df['Age'].astype('int64')
train_df['Bilirubin']= train_df['Bilirubin'].astype('int64')
train_df['Cholesterol']= train_df['Cholesterol'].astype('int64')
train_df['Albumin']= train_df['Albumin'].astype('int64')
train_df['Copper']= train_df['Copper'].astype('int64')
train_df['Alk_Phos']= train_df['Alk_Phos'].astype('int64')
train_df['SGOT']= train_df['SGOT'].astype('int64')
train_df['Tryglicerides']= train_df['Tryglicerides'].astype('int64')
train_df['Platelets']= train_df['Platelets'].astype('int64')
train_df['Prothrombin']= train_df['Prothrombin'].astype('int64')




test_df['Age']= test_df['Age'].astype('int64')
test_df['Bilirubin']= test_df['Bilirubin'].astype('int64')
test_df['Cholesterol']= test_df['Cholesterol'].astype('Int64')
test_df['Albumin']= test_df['Albumin'].astype('int64')
test_df['Copper']= test_df['Copper'].astype('int64')
test_df['Alk_Phos']= test_df['Alk_Phos'].astype('int64')
test_df['SGOT']= test_df['SGOT'].astype('int64')
test_df['Tryglicerides']= test_df['Tryglicerides'].astype('int64')
test_df['Platelets']= test_df['Platelets'].astype('int64')
test_df['Prothrombin']= test_df['Prothrombin'].astype('int64')

test_df.info()
x_1 = train_df.loc[train_df['Stage'] == 1.0]
x_2 = train_df.loc[train_df['Stage'] == 2.0]
x_3 = train_df.loc[train_df['Stage'] == 3.0]
x_4 = train_df.loc[train_df['Stage'] == 4.0]
x_2_1 = x_2.iloc[:753]
x_2_2 = x_2.iloc[754:1507]
print(len(x_2_1),len(x_2_2))
x_3_1 = x_3.iloc[:660]
x_3_2 = x_3.iloc[661:1322]
print(len(x_3_1),len(x_3_2))
x_4_1 = x_4.iloc[:875]
x_4_2 = x_4.iloc[876:1751]
x_4_3 = x_4.iloc[1752:2627]
x_4_4 = x_4.iloc[2628:3504]
print(len(x_4_1),len(x_4_2),len(x_4_3),len(x_4_4))
x_4_1 = x_4.iloc[:875]
x_4_2 = x_4.iloc[876:1751]
x_4_3 = x_4.iloc[1752:2627]
x_4_4 = x_4.iloc[2628:3504]
print(len(x_4_1),len(x_4_2),len(x_4_3),len(x_4_4))
x_1 = pd.concat([x_1,x_2_1,x_3_1,x_4_1])
x_2 = pd.concat([x_1,x_2_1,x_3_1,x_4_2])
x_3 = pd.concat([x_1,x_2_1,x_3_1,x_4_2])
x_4 = pd.concat([x_1,x_2_1,x_3_1,x_4_4])
x_5 = pd.concat([x_1,x_2_2,x_3_1,x_4_1])
x_6 = pd.concat([x_1,x_2_2,x_3_1,x_4_2])
x_7 = pd.concat([x_1,x_2_2,x_3_1,x_4_3])
x_8 = pd.concat([x_1,x_2_2,x_3_1,x_4_4])
x_1 = x_1.sample(frac=1).reset_index(drop=True)
x_2 = x_2.sample(frac=1).reset_index(drop=True)
x_3 = x_3.sample(frac=1).reset_index(drop=True)
x_4 = x_4.sample(frac=1).reset_index(drop=True)
x_5 = x_5.sample(frac=1).reset_index(drop=True)
x_6 = x_6.sample(frac=1).reset_index(drop=True)
x_7 = x_7.sample(frac=1).reset_index(drop=True)
x_8 = x_8.sample(frac=1).reset_index(drop=True)
x_1.info(),x_2.info(),x_3.info(),x_4.info(),x_5.info(),x_6.info(),x_7.info(),x_8.info()
x_1.info()
x_2.info()
x_3.info()
x_4.info()
x_5.info()
x_6.info()
x_7.info()
x_8.info()
y_1 = x_1['Stage']
x_1 = x_1.drop(['Stage'],axis=1)

y_2 = x_2['Stage']
x_2 = x_2.drop(['Stage'],axis=1)

y_3 = x_3['Stage']
x_3 = x_3.drop(['Stage'],axis=1)

y_4 = x_4['Stage']
x_4 = x_4.drop(['Stage'],axis=1)

y_5 = x_5['Stage']
x_5 = x_5.drop(['Stage'],axis=1)

y_6 = x_6['Stage']
x_6 = x_6.drop(['Stage'],axis=1)

y_7 = x_7['Stage']
x_7 = x_7.drop(['Stage'],axis=1)

y_8 = x_8['Stage']
x_8 = x_8.drop(['Stage'],axis=1)



X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(x_1, y_1, test_size=0.1, random_state=2, shuffle=True)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(x_2, y_2, test_size=0.1, random_state=2, shuffle=True)
X_3_train, X_3_test, y_3_train, y_3_test = train_test_split(x_3, y_3, test_size=0.1, random_state=2, shuffle=True)
X_4_train, X_4_test, y_4_train, y_4_test = train_test_split(x_4, y_4, test_size=0.1, random_state=2, shuffle=True)
X_5_train, X_5_test, y_5_train, y_5_test = train_test_split(x_5, y_5, test_size=0.1, random_state=2, shuffle=True)
X_6_train, X_6_test, y_6_train, y_6_test = train_test_split(x_6, y_6, test_size=0.1, random_state=2, shuffle=True)
X_7_train, X_7_test, y_7_train, y_7_test = train_test_split(x_7, y_7, test_size=0.1, random_state=2, shuffle=True)
X_8_train, X_8_test, y_8_train, y_8_test = train_test_split(x_8, y_8, test_size=0.1, random_state=2, shuffle=True)
clf1 = tree.DecisionTreeClassifier(max_depth=32)
cvs_1 = (np.mean(cross_val_score(clf1, X_1_train, y_1_train, cv=10)))
cvs_2 = (np.mean(cross_val_score(clf1, X_2_train, y_2_train, cv=10)))
cvs_3 = (np.mean(cross_val_score(clf1, X_3_train, y_3_train, cv=10)))
cvs_4 = (np.mean(cross_val_score(clf1, X_4_train, y_4_train, cv=10)))
cvs_5 = (np.mean(cross_val_score(clf1, X_5_train, y_5_train, cv=10)))
cvs_6 = (np.mean(cross_val_score(clf1, X_6_train, y_6_train, cv=10)))
cvs_7 = (np.mean(cross_val_score(clf1, X_7_train, y_7_train, cv=10)))
cvs_8 = (np.mean(cross_val_score(clf1, X_8_train, y_8_train, cv=10)))       
clf1.fit(X_1_train,y_1_train)
clf1.fit(X_2_train,y_2_train)
clf1.fit(X_3_train,y_3_train)
clf1.fit(X_4_train,y_4_train)
clf1.fit(X_5_train,y_5_train)
clf1.fit(X_6_train,y_6_train)
clf1.fit(X_7_train,y_7_train)
clf1.fit(X_8_train,y_8_train)
print(clf1.score(X_1_test,y_1_test))
print(clf1.score(X_2_test,y_2_test))
print(clf1.score(X_3_test,y_3_test))
print(clf1.score(X_4_test,y_4_test))
print(clf1.score(X_5_test,y_5_test))
print(clf1.score(X_6_test,y_6_test))
print(clf1.score(X_7_test,y_7_test))
print(clf1.score(X_8_test,y_8_test))

dump(clf1, 'dt_model.joblib')


y_pred = clf1.predict(X_5_test)
y_preds = np.round(y_pred)
print('Accuracy Score is: ', accuracy_score(y_5_test,y_preds))
Y_1 = pd.concat([y_1_test,y_2_test,y_3_test,y_4_test,
                 y_5_test,y_6_test,y_7_test])
cm = confusion_matrix(y_5_test,y_preds)
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
df_target=pd.DataFrame(Y_1)

 # its important for comparison. Here "test_new" is your new test dataset
df_target.columns = ['Stage']
df_target

df_target.to_csv('prediction_results.csv', index = True) 