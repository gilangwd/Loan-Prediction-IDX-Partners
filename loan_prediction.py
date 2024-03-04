# Created by Gilang Wiradhyaksa

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV #, RandomizedSearchCV
import pickle

import warnings
warnings.filterwarnings('ignore')

# Original Data
df_ori = pd.read_csv('loan_data_2007_2014.csv', index_col=0)

df = df_ori.copy()
df

df.info()
df.describe()
df.columns

# Check duplicated data
df.duplicated().sum()

# Check missing values
df.isnull().sum()

df.shape
df = df.dropna(axis=1, how='all')
df.shape

# Count null values in each column
null_counts = df.isnull().sum()

# Filter columns with more than 0 null values
columns_with_null = null_counts[null_counts > 0]
columns_with_null

# List column with few row is null
few_null = ['annual_inc', 'title', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec', 'total_acc', 
            'last_credit_pull_d', 'collections_12_mths_ex_med', 'acc_now_delinq']

# Remove Row with few column null
print('before remove row : ', df.shape)
df = df.dropna(subset=few_null)
print('after remove row : ', df.shape)

# Count null values in each column
null_counts = df.isnull().sum()

# Filter columns with more than 0 null values
columns_with_null = null_counts[null_counts > 0]
columns_with_null

# List not-important column with many null value
ni_null_col = ['emp_title', 'emp_length', 'desc', 'revol_util', 'mths_since_last_delinq', 'mths_since_last_record', 
               'last_pymnt_d', 'next_pymnt_d', 'mths_since_last_major_derog', 'tot_coll_amt', 'total_rev_hi_lim']

print('before remove column : ', df.shape)
df_selection = df.drop(columns=ni_null_col)
print('after remove column : ', df_selection.shape)

# Replace null in current balance with 0
df_selection['tot_cur_bal'] = df_selection['tot_cur_bal'].fillna(0)

# Count null values in each column
null_counts = df_selection.isnull().sum()

# Filter columns with more than 0 null values
columns_with_null = null_counts[null_counts > 0]
columns_with_null

# DF Shape after missing value handling
df_selection.shape

df_selection.columns

# Check target (y) unique value
df_selection['loan_status'].unique()

# Mapping target (y)
def loanStatus(row):
   if row['loan_status'] == 'Fully Paid' or row['loan_status'] == 'Current' or row['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid':
      return 1
   else:
      return 0
   
df_selection['loan_status'] = df_selection.apply(lambda row: loanStatus(row), axis=1)
print(df_selection['loan_status'].unique())

df_selection.head()

pd.reset_option('display.max_columns')

# Feature Selection
imp_col = ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'grade', 'home_ownership', 'annual_inc', 'verification_status', 
           'dti', 'delinq_2yrs', 'open_acc', 'pub_rec', 'total_acc', 'initial_list_status', 'total_pymnt', 'total_pymnt_inv', 
           'last_pymnt_amnt', 'acc_now_delinq', 'tot_cur_bal', 'loan_status']

df_selection['home_ownership'].unique()
# Own, Mortgage = Own, else None

df_selection['verification_status'].unique()
# Source Verified = Verified

print('before feature selection : ', df_selection.shape)
df_selection = df_selection[imp_col]
print('after feature selection : ', df_selection.shape)

# Remove 'months' from term
df_selection['term'] = df_selection['term'].str.replace('months', '').str.strip()

# Replace Source verified to verified from verification status
df_selection['verification_status'] = df_selection['verification_status'].str.replace('Source Verified', 'Verified').str.strip()
df_selection['verification_status'].unique()

# Mapping Home Ownership
def homeOwnership(row):
   if row['home_ownership'] == 'OWN' or row['home_ownership'] == 'MORTGAGE':
      return 'OWN'
   else:
      return 'NONE'
   
df_selection['home_ownership'] = df_selection.apply(lambda row: homeOwnership(row), axis=1)
print(df_selection['home_ownership'].unique())

df_selection.head()

df_selection.info()

def exploreNumCol(df,  col):
    mean = df[col].mean()
    median = df[col].median()
    modus = df[col].mode().values[0]

    min = df[col].min()
    max = df[col].max()

    print(f'Mean {col} = {mean:.2f}')
    print(f'Median {col} = {median}')
    print(f'Modus {col} = {modus}')
    print(f'Min {col} = {min}')
    print(f'Max {col} = {max}')

    skew = df[col].skew()
    if skew < 0.5:
        print(f'Skewness {col} = {skew}, data distribution is normal')
    else:
        print(f'Skewness {col} = {skew}, data distribution is not normal')


exploreNumCol(df_selection, 'loan_amnt')

sns.histplot(df_selection['loan_amnt'], bins=20, kde=True).set(title='Loan Amount')

exploreNumCol(df_selection, 'annual_inc')

def sns_barplot(df, groupby_column, label):
    plt.figure(figsize=(3, 5))
    df_barplot = df.groupby(groupby_column).size().reset_index(name='counts')
    ax = sns.barplot(data=df_barplot, x=groupby_column, y='counts', orient='v')
    ax.bar_label(ax.containers[0]) if label == True else None
    ax.set(title=f'Count of {groupby_column}')
    plt.show()

sns.histplot(df_selection['annual_inc'], bins=20, kde=True).set(title='Annual Income')

sns_barplot(df_selection, 'loan_status', True)

# Split X and y
X = df_selection.drop('loan_status', axis=1)
y = df_selection['loan_status']

#Split train and test (80% Train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=21)
print('Train Size : ', X_train.shape)
print('Test Size : ', X_test.shape)
print('y Train Size : ', y_train.shape)
print('y Test Size : ', y_test.shape)

X_train.head()

def diagnostic_plots(df, variable):
    # Define figure size
    plt.figure(figsize=(16, 4))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df, bins=30) if variable is None else sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df) if variable is None else sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()

def getDescribe(df, col):
    min_value = df[col].min()
    max_value = df[col].max()
    average_value = df[col].mean()
    mode_value = df[col].mode().iloc[0]

    # Print the results
    print("Minimum:", min_value)
    print("Maximum:", max_value)
    print("Average:", average_value)
    print("Mode:", mode_value)

getDescribe(X_train, 'annual_inc')

diagnostic_plots(X_train, 'annual_inc')

getDescribe(X_train, 'tot_cur_bal')

diagnostic_plots(X_train, 'tot_cur_bal')

def getTukeysRuleBoundary(df, col, iqr_wide):
    IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
    lower_boundary = df[col].quantile(0.25) - (IQR * iqr_wide)
    upper_boundary = df[col].quantile(0.75) + (IQR * iqr_wide)

    print(f'Lower Boundary {col} : {lower_boundary}')
    print(f'Upper Boundary {col} : {upper_boundary}')

    return lower_boundary, upper_boundary

def getPrecentageOutliers(df, col, upper_b):
    print('Total Data : {}'.format(len(df)))
    print('Data which ' + col + ' more than ' + str(upper_b) + ' : {}'.format(len(df[df[col] > upper_b])))
    print('% Data which ' + col + ' more than ' + str(upper_b) + ' : {}'.format(len(df[df[col] > upper_b])/len(df) * 100))


lower_b_annual_inc, upper_b_annual_inc = getTukeysRuleBoundary(X_train, 'annual_inc', 1.5)
lower_b_tot_cur_bal, upper_b_tot_cur_bal = getTukeysRuleBoundary(X_train, 'tot_cur_bal', 1.5)

getPrecentageOutliers(X_train, 'annual_inc', upper_b_annual_inc)
print('-----------------------------------------------------------------')
getPrecentageOutliers(X_train, 'tot_cur_bal', upper_b_tot_cur_bal)

# Flag the outliers in column `annual_inc`
outliers_annual_inc = np.where(X_train['annual_inc'] > upper_b_annual_inc, True, np.where(X_train['annual_inc'] < lower_b_annual_inc, True, False))

# Flag the outliers in column `tot_cur_bal`
outliers_tot_cur_bal = np.where(X_train['tot_cur_bal'] > upper_b_tot_cur_bal, True, np.where(X_train['tot_cur_bal'] < lower_b_tot_cur_bal, True, False))

# Trimming the dataset
X_train_trimmed = X_train.loc[~(outliers_annual_inc) + (outliers_tot_cur_bal)]
y_train_trimmed = y_train.loc[~(outliers_annual_inc) + (outliers_tot_cur_bal)]

print('Size X_train - Before trimming : ', X_train.shape)
print('Size X_train - After trimming  : ', X_train_trimmed.shape)
print('')
print('Size y_train - Before trimming : ', y_train.shape)
print('Size y_train - After trimming  : ', y_train_trimmed.shape)

# Split Numercal and Categorical
X_train_trimmed.head()

# num_col_normal = []
num_col_skew = ['loan_amnt', 'funded_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'open_acc', 'pub_rec', 
                'total_acc', 'total_pymnt', 'total_pymnt_inv', 'last_pymnt_amnt', 'acc_now_delinq', 'tot_cur_bal']
cat_col_ordinal = ['grade']
cat_col_nominal = ['term', 'home_ownership', 'verification_status', 'initial_list_status']

X_train_num_s = X_train_trimmed[num_col_skew]
X_train_cat_o = X_train_trimmed[cat_col_ordinal]
X_train_cat_n = X_train_trimmed[cat_col_nominal]

X_test_num_s = X_test[num_col_skew]
X_test_cat_o = X_test[cat_col_ordinal]
X_test_cat_n = X_test[cat_col_nominal]

#Scaling Numerical Column with Skew Data
scaler_mm = MinMaxScaler()
scaler_mm.fit(X_train_num_s)

X_train_num_s_scaled = scaler_mm.transform(X_train_num_s)
X_test_num_s_scaled = scaler_mm.transform(X_test_num_s)

X_train_num_s_scaled

print(X_train['grade'].unique())

gradeOrder = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

od_encoder = OrdinalEncoder(categories=[gradeOrder])
od_encoder.fit(X_train_cat_o)

X_train_cat_o_encoded = od_encoder.transform(X_train_cat_o)
X_test_cat_o_encoded = od_encoder.transform(X_test_cat_o)

X_train_cat_o_encoded

oh_encoder = OneHotEncoder(handle_unknown='ignore')
oh_encoder.fit(X_train_cat_n)

X_train_cat_n_encoded = oh_encoder.transform(X_train_cat_n).toarray()
X_test_cat_n_encoded = oh_encoder.transform(X_test_cat_n).toarray()

X_train_cat_n_encoded

X_train_final = np.concatenate([X_train_num_s_scaled, X_train_cat_o_encoded, X_train_cat_n_encoded], axis=1)
X_test_final = np.concatenate([X_test_num_s_scaled, X_test_cat_o_encoded, X_test_cat_n_encoded], axis=1)

model_logreg = LogisticRegression()
model_knn = KNeighborsClassifier(n_neighbors=3)
model_rf = RandomForestClassifier()

model_logreg.fit(X_train_final, y_train_trimmed)
model_knn.fit(X_train_final, y_train_trimmed)
model_rf.fit(X_train_final, y_train_trimmed)

# Predict Train and test set
y_pred_train_lg = model_logreg.predict(X_train_final)
y_pred_test_lg = model_logreg.predict(X_test_final)

print(classification_report(y_train_trimmed, y_pred_train_lg))
print(classification_report(y_test, y_pred_test_lg))
print('Precision Score Train - LogReg : ', precision_score(y_train_trimmed, y_pred_train_lg))
print('Precision Score Test - LogReg : ', precision_score(y_test, y_pred_test_lg))

cm_train = confusion_matrix(y_train_trimmed, y_pred_train_lg)
ConfusionMatrixDisplay(cm_train).plot()

precision_train_cross_val = cross_val_score(model_logreg, X_train_final, y_train_trimmed, cv=5, scoring="precision")
print('Precision Train - All - Cross Validation  : ', precision_train_cross_val)
print('Precision Train - Mean - Cross Validation : ', precision_train_cross_val.mean())

# Predict Train and test set
y_pred_train_knn = model_knn.predict(X_train_final)
y_pred_test_knn = model_knn.predict(X_test_final)

print(classification_report(y_train_trimmed, y_pred_train_knn))
print(classification_report(y_test, y_pred_test_knn))
print('Precision - KNN : ', precision_score(y_train_trimmed, y_pred_train_knn))
print('Precision - KNN : ', precision_score(y_test, y_pred_test_knn))

cm_train = confusion_matrix(y_train_trimmed, y_pred_train_knn)
ConfusionMatrixDisplay(cm_train).plot()

precision_train_cross_val = cross_val_score(model_knn, X_train_final, y_train_trimmed, cv=5, scoring="precision")
print('Precision Train - All - Cross Validation  : ', precision_train_cross_val)
print('Precision Train - Mean - Cross Validation : ', precision_train_cross_val.mean())

y_pred_train_rf = model_rf.predict(X_train_final)
y_pred_test_rf = model_rf.predict(X_test_final)

print(classification_report(y_train_trimmed, y_pred_train_rf))
print(classification_report(y_test, y_pred_test_rf))
print('Precision - RF : ', precision_score(y_train_trimmed, y_pred_train_rf))
print('Precision - RF : ', precision_score(y_test, y_pred_test_rf))

cm_train = confusion_matrix(y_train_trimmed, y_pred_train_rf)
ConfusionMatrixDisplay(cm_train).plot()

precision_train_cross_val = cross_val_score(model_rf, X_train_final, y_train_trimmed, cv=5, scoring="precision")
print('Precision Train - All - Cross Validation  : ', precision_train_cross_val)
print('Precision Train - Mean - Cross Validation : ', precision_train_cross_val.mean())