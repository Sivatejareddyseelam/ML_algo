# Load the necessary libraries 
# Set the seed to 123
import pandas as pd
import numpy as np



# load the dataset into the memory
data = pd.read_csv('Logistic_regression.csv')
# Pre-processing steps
'''You may need to clean the variables, impute the missing values and convert the categorical variables to one-hot encoded

following variables need to be converted to one_hot encoded
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

your final table should be like the following 

array(['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate',
       'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'y',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur',
       'job_housemaid', 'job_management', 'job_retired',
       'job_self-employed', 'job_services', 'job_student',
       'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'marital_unknown', 'education_Basic', 'education_high.school',
       'education_illiterate', 'education_professional.course',
       'education_university.degree', 'education_unknown', 'default_no',
       'default_unknown', 'default_yes', 'housing_no', 'housing_unknown',
       'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
       'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug',
       'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may',
       'month_nov', 'month_oct', 'month_sep', 'day_of_week_fri',
       'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue',
       'day_of_week_wed', 'poutcome_failure', 'poutcome_nonexistent',
       'poutcome_success'], dtype=object)
'''
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1 = data.join(cat_list)
    data = data1
cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data=data[to_keep]
'''
separate the features and the target variable 
'''
x = data.loc[:, data.columns != 'y']
y = data.loc[:, data.columns == 'y']
# x = Your code goes here 
# y = Your code goes here


'''
as your target class is imbalanced you need to use SMOTE function to balance it, first separate your data into training and testing set.
then use SMOTE function on the Training set.remember to not touch the testing set.
'''

from imblearn.over_sampling import SMOTE

s=SMOTE(random_state=0)
x_train=x.sample(frac=0.7,random_state=0)
x_test=x.loc[~data.index.isin(x_train.index)]
y_train=y.sample(frac=0.7,random_state=0)
y_test=y.loc[~data.index.isin(x_train.index)]
columns=x_train.columns
s_data_x,s_data_y=s.fit_sample(x_train,y_train)
s_data_x=pd.DataFrame(data=s_data_x,columns=columns)
s_data_y=pd.DataFrame(data=s_data_y,columns=["Y"])#print(s_data_x.columns)
'''
You need to eliminate variables with p-values larger than 0.05. To do so you can use the following function
# import statsmodels.api as sm

'''

import statsmodels.api as sm
model=sm.OLS(s_data_y,s_data_x).fit()
p=model.pvalues
#print(p)
for c in columns:
    if p[c]>0.05:
        s_data_x.drop(c,axis=1)
print(s_data_x.head())

'''
Logistic Regression Model Fitting - iterative approach 
you need to complete the following functions 
'''


'''
parameters : scores
does : calculate the sigmoid value of the scores
return : the sigmoid value
'''
def sigmoid(z):
    return 1/(1+np.exp(-z))

'''
paramters : Input variables, Target variable and weights 
does : calculate the log-likelihood 
return : the log-likelihood'''
def log_likelihood(x, y , w):
    z=np.dot(x,w)
    h=sigmoid(z)
    return (-y*np.log(h)-(1-y)*np.log(1-h)).mean()



    '''
    parameters : features, target, num_steps, learning_rate and add_intercept = False
    does : calculate the logistic regression weights, i have provided the add_intercept section of the code, to run you need to change the value from False to True 
    return : The logistic regression weights'''
    # Don't modify this part
def logistic_regression(x,y,n,r,add_intercept = False):
    if add_intercept:
        intercept = np.ones((x.shape[0], 1))
        features = np.hstack((intercept,x))

    # Your code goes here
    weights = np.zeros((x.shape[1],1))
    ''' You need to iterate over the number of steps and update the weights in each iteration based on the gradient '''
    for step in range(n):
        # Calculate the prediction value
        z=np.dot(x,weights)
        h=sigmoid(z)
        x_t=np.transpose(x)
        gradient=((x_t.dot(h-y)))/51226
        # Update weights with gradient
        weights-=r*gradient
        # Print log-likelihood every so often - don't change this part 
        #if step % 10000 == 0:
            #print(log_likelihood(x, y, weights))

    return(weights)

print(logistic_regression(s_data_x,s_data_y,1000,0.001))