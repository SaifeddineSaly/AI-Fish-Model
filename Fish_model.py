# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 03:05:54 2023

@author: Me
"""

################################# p002_data_collect
def p002_data_collect (CSV_file):
    import pandas as pd
    # i) Reading the dataset or Load CSV file: df 
    df = pd.read_csv(CSV_file)
    # ii) Print ('the size of the csv weather data frame is: ‘, df.shape)
        # Result: the size of the csv weather data frame is:  (145460, 24) =>
        # The dimension of the data frame is:  145 460 rows and 24 columns
    
    # iii) Display the first five observations in our data frame
    five_rows = df[0:5]
    # print (five rows)
    return df , five_rows
    #df, five_rows  = p002_data_collect (r"C:\17_IUT_AI\Model\02_weather.csv")
    #nf = r"D:\All of Université\S5 Document & videos\2022-2023\Intélligence Artificiel\Machine Learning\02_weather.csv"
nf = r"C:\Users\acc\Desktop\AI\04-Fish.csv"
df, five_rows  = p002_data_collect (nf)
print (df, five_rows )
print(str(df))

# In[ ]:


################################ p003_data_preparation
def p003_data_preparation (df):
    from scipy import stats
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    
    # a. Checking Null or missing values: The count() method counts the number of not empty values for each row, 
            # or column if you specify the axis parameter as axis='columns', 
            # and returns a Series object with the result for each row (or column).
    null_val = df.count().sort_values()
    #print (null_val)
    #return null_val
#val_null = p003_data_preparation (df)
#print(val_null)

    # b. Remove unwanted and redundant columns
      # i) Unecessary data will always increase our computations that is why it is always better
          # to remove them.
      # ii) So apart from removing the unnecessary variables, we also will remove the 
          # "location variables” and we will remove the "date variable" because both 
          # of these two variables are not needed in order to predict whether it will rain 
          # tomorrow or not.         
      # iii) # We will also remove “RISK_MM" variable because this tells us the amount 
                  # of rain that might occur the next day. 
             # This is a very informative variable and it may actually leak some information to our model. 
             # By using this variable it will be able easy to predict 'RainTom' 
             # This variable will give us too much information and that is why we are going to remove it.            
             # Because we will let the model to discover whether it rains or not based on the training process            
                 # and since this variable, leaks a lot of information so it should be dropped from the dataSet.    
    rain_drop_unwanted = df.drop(['Cross'],axis=1)
    
    # c. Remove null values from the Last data Frame
    rain_drop_unwanted_and_null = rain_drop_unwanted.dropna(how='any')
    
    ###
    #zscore calculte and add
    #rain_drop_unwanted_and_null['zscore'] = (rain_drop_unwanted_and_null.Weight - rain_drop_unwanted_and_null.Weight.mean() ) /  rain_drop_unwanted_and_null.Weight.std()
    # print(str(rain_drop_unwanted_and_null.MinTemp.mean()) +" and " + str(rain_drop_unwanted_and_null.MinTemp.std()))
    #import seaborn as sns
    #sns.boxplot(rain_drop_unwanted_and_null['Temp9am'])
    #return rain_drop_unwanted_and_null

    ###
    
    #return rain_drop_unwanted, rain_drop_unwanted_and_null 

#rain_drop_unwanted, rain_drop_unwanted_and_null = p003_data_preparation (df)

    # d. Remove Outliers (Tmin = 113 instead of 11.3)
       # i) Now it is the time to remove the outliers inside the dataframe.     
      # ii) The outlier is a data that is very different from the other observations.         
      # iii) Outlier’s usually occur because of miscalculations while collecting the data.    
      # iiii) (ex: T=115 instead of 11.5). These are some sort of errors in the data set.
    rain_drop_unwanted_and_null_rmvOutliers = np.abs(stats.zscore(rain_drop_unwanted_and_null._get_numeric_data())) 
    rain_drop_unwanted_and_null_rmvOutliersNull = rain_drop_unwanted_and_null [(rain_drop_unwanted_and_null_rmvOutliers < 3).all(axis=1) ]
    
    #return rain_drop_unwanted_and_null_rmvOutliers, rain_drop_unwanted_and_null_rmvOutliersNull

#rain_drop_unwanted_and_null_rmvOutliers, rain_drop_unwanted_and_null_rmvOutliersNull = p003_data_preparation (df)

# e. Handling categorical variable     
       # i) Now what we will be doing is we will be assigning 0 and 1 to the place of yes and no.
       # ii) That means we are going to change the categorical variables from yes and no to 0 and 1 .
    df_lables = rain_drop_unwanted_and_null_rmvOutliersNull
    #df_lables['RainToday'].replace    ({'No'  :  0 , 'Yes' : 1} , inplace=True    )
    #df_lables['Weight'].replace ({'No'  :  0 , 'Yes' : 1} , inplace=True    )  
    #return df_lables
#de_lables = p003_data_preparation (df)

# f. Handling unique keys,character values will be changed into integer values
       # If we have unique values such as any character values which are not supposed to be there, 
            # we will change them into integer values
    df_char_to_num = df_lables    
    # See unique values and change them into int using pd.getDummiies()
    categorical_columns = ['Species']
    for col in categorical_columns:
        print(np.unique(df_char_to_num[col]))
        print ("")
    #Transform the categorical columns
    dfFinal = pd.get_dummies(df_lables , columns=categorical_columns)
    # print (df.iloc[4:9])
    # print (df) #[107868 rows x 62 columns]    
    #return dfFinal
#dfFinal = p003_data_preparation (df)

    # g. Normalize all data in the recent data-frame
       # i) Now we will be proceeding to normalizing Data
       # ii) Standardize our data by using MinMaxScaler
       # iii) Google: how to standardize data in python??: https://www.askpython.com/python/examples/standardize-data-in-python
       # iiii) This normalizing process is very important because to reduce or avoid any biases in your output 
       # v) you should normalize your input variables. 
       # vi) It was done by using the function Minm=MaxScaler provided by python in a package known as #sklearn.    
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(dfFinal)
    dfFinal_Std = pd.DataFrame(scaler.transform(dfFinal), index=dfFinal.index, columns=dfFinal.columns)
    return dfFinal_Std

dfFinal_Std1 = p003_data_preparation (df)
    # return (null_val, rain_drop_unwanted_and_null_rmvOutliersNull,  df_lables, dfFinal ,dfFinal_Std)

# nullValues , rain_drop_unwanted_and_null_rmvOutliersNull , df_lables, dfFinal,  df_data_prep  =  p003_data_preparation (df)
# print (df_data_prep)

# In[ ]:

################################ p004_data_explarotary
# Now we well go to the step EDA, Exploratory Data Analysis
def p004_data_explarotary (df):
    # Now what we are going to do, is get analyzed and identify the significant
     #variables that will help us to forecast the dependant variable (RainTom).
      # i) To do this, we will use the 'selctKeyBest' function getting from Sklearn library.             
      # ii) Using this function, we will select the most significant independent variables in our dataset.
      # iii) Google: how to select features using chi squared in python?
                    # OR
            # Google: how to select features  in python?    
    from sklearn.feature_selection import SelectKBest, chi2
    
    X = df.loc[: , df.columns != 'Weight']
    Y = df['Weight']
    selector = SelectKBest(chi2, k=2)
    Y = Y.astype('int')
    selector.fit(X, Y)
    x_new  = selector.transform(X)
    print (X.columns[selector.get_support(indices=True)])    
    # i) Hence, we get the most significant independent variables in our dataset that influence the
    #    dependent variable 'RainTomorrow'
    # ii) Index(['Rainfall', 'Humidity3pm', 'RainToday'], dtype='object')
    # iii) We just enough to feed our models by these 3 variables instead of all variables dataset as input
    # iv) This simplifies the computation process
    # vi) Basically we will create a data frame of the significant variables overall
    #when k = 3 => #Index(['Vertical', 'Diameter', 'Species_Pike'], dtype='object')
    #Index(['Diameter', 'Species_Pike'], dtype='object')
    df_Best_Feature = df[['Diameter',  'Species_Pike',  'Weight']]
    
    
    # vii) What would be done later is assigning one of these significant variables as input instead of 
             #taking all three variables to predict the  'RainTomorrow' variable
    # viii) Let's use only one feature 'Diameter'
    X_Best_Feature  =  df[['Diameter']] 
    # ix) Obviously our outcome is  'RainTomorrow' the variable to be predicted.
    Y_Target_Fearure =  df[['Weight']]    
    return  df_Best_Feature, X_Best_Feature, Y_Target_Fearure   
    
df_Best_Feature, X_Best_Feature, Y_Target_Fearure= p004_data_explarotary(dfFinal_Std1)


# In[ ]:


# Part 05#################################### p005_Build_Model

    # i)  Now we are processing to data modeling 
    # ii) I suppose now that we are aware of what data modeling is to solve this step.
    # iii) we will be using four classification algorithms over here in order to predict the outcome "RainTomorrow".
            #  - LogisticRegression
            #  - Random Forest 
            #  - Decision Tree Classifier
            #  - Support Vector Machine
    # iv)  Finally, we will check the best algorithm that will give us the best accuracy
    # v)   We will continue by applying the LogisticRegression algorithm

def p005_Build_Model (X_BestFeature, Y_TargetFearure):      
    # vi) Import all the necessary libraries for the 'LogisticRegression' algorithm     
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn import preprocessing
    from sklearn import utils

    
    # vii)  We are importing the 'time' libraries because we will calculate the accuracy
            # and the time taken by the algorithm to finish the model's execution.   
    import time    
    t0 =time.time()
    
    # Split data into 4 parts
    X_train, X_test, Y_train, Y_test = train_test_split (X_BestFeature,Y_TargetFearure, test_size=0.25)
    
    #  Create an instance of the 'LogisticRegression' algorithm
    model_LogesticRegresson = LogisticRegression(random_state=0)
    
    #convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    Y_train = lab.fit_transform(Y_train)
    # Building the model using the training data set, that means calculated  the coefficient of the model's equation
    model_LogesticRegresson.fit(X_train, Y_train)
    
    return X_test, Y_test, model_LogesticRegresson;

    
Xtst, Ytst, model_LR = p005_Build_Model (X_Best_Feature, Y_Target_Fearure)  


# In[ ]:
################################ p006_Model_Evaluation
 # i) In this step we should apply the model equation on the Xtest and generate Yhat (Yhat, Ypredcit)
 # ii) Compare the Ytest real with Yhat (making the difference)
 # iii) Check the efficiency of the model and how accurately, it can predict the outcome.

def p006_Model_Evaluation (model, Xtest, Ytest):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    import numpy
    from sklearn.metrics import r2_score
    import time    
    t0 =time.time()
    
    
    #print(type(Yhat))
    #rounded_arr = np.around(Yhat, decimals=0)
    #print(rounded_arr)
    #rounded_arr_2 = np.around(Ytest)
    #print(rounded_arr_2)
    #print(Ytest)
    #accuracy = accuracy_score(numpy.round(Yhat) , numpy.round(Ytest))
    #accuracy = accuracy_score(Ytest, round(Yhat))
    #accuracy = r2_score(Ytest, Yhat)
    Yhat = model.predict(Xtest)
    print(Yhat)
    accuracy = accuracy_score(Ytest, Yhat)
    
    print("Accuracy using 'LogisticRegression'   :  "  , accuracy)
    print("Time taken using 'LogisticRegression' :"   , time.time()-t0)
    
    return (Yhat, Ytest , accuracy)


Yhat, Ytest, precision = p006_Model_Evaluation (model_LR, Xtst, Ytst)   
#print(Yhat, Ytest, precision)