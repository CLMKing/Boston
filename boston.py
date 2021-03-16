####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf
####

st.set_page_config(layout = 'wide', initial_sidebar_state = 'expanded')
col1 = st.sidebar
col2,col3 = st.beta_columns((2,1))
col1.title('Boston House Prices')
col1.subheader('Navigation Page')
navigation = col1.radio('Selection', ['Information','Exploratory Data Analysis', 'Graphs and Charts', 'Modelling'])
col1.write('------')

####

data = load_boston()
X,Y = load_boston(return_X_y = True)
df = pd.DataFrame(X, columns = data.feature_names)
df['LABELS'] = Y

####
st.title('Boston House Prices')
st.write('------')

####
if navigation == 'Information':
    st.write(data.DESCR)

####

if navigation == 'Exploratory Data Analysis':
    col1.subheader('Options')
    if col1.checkbox('Raw Data'):
        st.subheader('Raw Data')
        st.write(df)
        st.success('Loading Successful')
        st.write('------')
    if col1.checkbox('Data Head'):
        st.subheader('Data Head')
        st.write(df.head(10))
        st.markdown('*First ten rows*')
        st.success('Loading Successful')
        st.write('------')
    if col1.checkbox('Data Tail'):
        st.subheader('Data Tail')
        st.write(df.tail(10))
        st.markdown('*Last ten rows*')
        st.success('Loading Successful')
        st.write('------')
    if col1.checkbox('Summary'):
        st.subheader('Summary')
        st.write(df.describe())
        st.success('Loading Successful')
        st.write('------')
    if col1.checkbox('Shape and Columns'):
        st.subheader('Shape')
        st.write(df.shape)
        st.subheader('Columns')
        st.write(df.columns)
        st.success('Loading Successful')
        st.write('------')

#####
if navigation == 'Graphs and Charts':
    col1.subheader('Please Select Figure')
    select_ = col1.selectbox('', ['Default','Heat Map Correlation', 'PRICE vs CRIM', 'PRICE vs ZN', 'PRICE vs INDUS', 'PRICE vs CHAS'
                                  'PRICE vs NOX', 'PRICE vs RM', 'PRICE vs AGE', 'PRICE vs DIS', 'PRICE vs RAD',
                                    'PRICE vs TAX', 'PRICE vs PTRATIO', 'PRICE vs B', 'PRICE vs LSTATS'])

    if select_ == 'Heat Map Correlation':
        st.subheader('Heat Map Correlation')
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(df.corr(), annot = True)
        st.pyplot(fig)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs CRIM':
        st.header('PRICE vs CRIM')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['CRIM'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('CRIM', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs ZN':
        st.header('PRICE vs ZN')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['ZN'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('ZN', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs INDUS':
        st.header('PRICE vs INDUS')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['INDUS'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('INDUS', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs NOX':
        st.header('PRICE vs NOX')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['NOX'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('NOX', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs RM':
        st.header('PRICE vs RM')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['RM'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('RM', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs AGE':
        st.header('PRICE vs AGE')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['AGE'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('AGE', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs DIS':
        st.header('PRICE vs DIS')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['DIS'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('DIS', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs RAD':
        st.header('PRICE vs RAD')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['RAD'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('RAD', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs TAX':
        st.header('PRICE vs TAX')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['TAX'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('TAX', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs PTRATIO':
        st.header('PRICE vs PTRATIO')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['PTRATIO'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('PTRATIO', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs B':
        st.header('PRICE vs B')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['B'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('B', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    if select_ == 'PRICE vs LSTATS':
        st.header('PRICE vs LSTATS')
        fig,ax = plt.subplots(figsize = (10,10))
        feature_name = df.columns
        plt.scatter(df['LSTATS'], df['LABELS'], alpha = 0.6)
        plt.ylabel('PRICE', fontsize = 15, size = 30)
        plt.xlabel('LSTATS', fontsize = 15, size = 30)
        st.pyplot(plt)
        st.success('Loading Successful')
        st.write('------')

    col1.write('------')
    if col1.checkbox('Custom Plot'):
        col1.subheader('Please Choose Two Features')
        opt = col1.multiselect('', df.columns)
        if len(opt) > 2:
            col1.error('Too many features')
        elif len(opt) < 2:
            col1.error('Need one more feature')
        elif len(opt) == 2:
            fig, ax = plt.subplots(figsize=(10, 10))
            feature_name = [opt[0],opt[1]]
            plt.scatter(df[opt[0]], df[opt[1]], alpha=0.6)
            plt.ylabel(opt[1], fontsize=15, size=30)
            plt.xlabel(opt[0], fontsize=15, size=30)
            st.pyplot(plt)
            st.success('Loading Successful')
            st.write('------')

#####
if navigation == 'Modelling':
    col1.subheader('Modelling Algorithm')
    opt1 = col1.selectbox('', ['Linear Regression', 'Multiple Linear Regression', 'K Neighbors Regressor'])
    cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    feat_cols = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
    targ_cols = df['LABELS']

    if opt1 == 'Linear Regression':
        col1.subheader('Please Choose a Feature')
        opt2 = col1.selectbox('', df.columns)
        col1.write('------')
        st.subheader('Linear Regression Using Two Features')
        X_train, X_test, Y_train, Y_test = train_test_split(df[[opt2]], targ_cols, random_state = 42, test_size = 0.20)
        linear_model = LinearRegression().fit(X_train, Y_train)
        score = linear_model.score(X_test, Y_test)
        y_predict = linear_model.predict(X_test)
        mse = mean_squared_error(Y_test, y_predict)
        ####
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.scatter(X_test, Y_test, alpha = 0.6)
        plt.scatter(X_test, y_predict, alpha = 0.6, color = 'red')
        plt.ylabel(opt2, fontsize=15, size=30)
        plt.xlabel('LABELS', fontsize=15, size=30)
        st.pyplot(plt)
        st.write('Score: ', score)
        st.write('Mean Squared Error: ', mse)
        st.write('Root Mean Square Error: ', np.sqrt(mse))
        if st.checkbox('First 20 items Actual vs Predicted labels'):
            lst_test_val = []
            lst_pred_val = []
            for i in range(1, 20):
                lst_test_val.append(Y_test.values[i])
                lst_pred_val.append(y_predict[i])
            df_1 = pd.DataFrame()
            df_1['Actual'] = lst_test_val
            df_1['Predicted'] = lst_pred_val
            st.write(df_1)
        st.success('Loading Successful')
        st.write('------')

    if opt1 == 'Multiple Linear Regression':
        st.subheader('Multiple Linear Regression / Multivariate ')
        opt3 = col1.multiselect('Select Features', cols, cols)
        col1.write('------')
        feat_cols = feat_cols[opt3]
        targ_cols = df['LABELS']
        X_train, X_test, Y_train, Y_test = train_test_split(feat_cols, targ_cols, random_state = 42, test_size = 0.20)
        linear_model = LinearRegression().fit(X_train, Y_train)
        score1 = linear_model.score(X_test, Y_test)
        y_predict1 = linear_model.predict(X_test)
        mse1 = mean_squared_error(Y_test, y_predict1)
        st.subheader('Metrics')
        st.write('Score: ', score1)
        st.write('Mean Squared Error: ', mse1)
        st.write('Root Mean Square Error: ', np.sqrt(mse1))
        if st.checkbox('First 20 items Actual vs Predicted labels'):
            lst_test_val = []
            lst_pred_val = []
            for i in range(1, 20):
                lst_test_val.append(Y_test.values[i])
                lst_pred_val.append(y_predict1[i])
            df_1 = pd.DataFrame()
            df_1['Actual'] = lst_test_val
            df_1['Predicted'] = lst_pred_val
            st.write(df_1)
        st.success('Loading Successful')
        st.write('------')

    if opt1 == 'K Neighbors Regressor':
        st.subheader('K Neighbors Regressor')
        st.subheader('Multiple Linear Regression / Multivariate ')
        opt4 = col1.multiselect('Select Features', cols, cols)
        col1.write('------')
        feat_cols = feat_cols[opt4]
        targ_cols = df['LABELS']

        cv_score = []
        n_neighbors = np.arange(1, 25)
        X_train, X_test, Y_train, Y_test = train_test_split(feat_cols, targ_cols, random_state=42, test_size=0.20)
        for i in n_neighbors:
            knn_model1 = KNeighborsRegressor(n_neighbors=i).fit(X_train, Y_train)
            scores = knn_model1.score(X_test, Y_test)
            cv_score.append(scores)
        if st.checkbox('Optimization Chart'):
            st.write('Optimization Chart')
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.lineplot(x = n_neighbors, y = cv_score)
            plt.xlabel('# of Neighbors', fontsize = 15, size = 30)
            plt.ylabel('Accuracy Score', fontsize = 15, size = 30)
            st.pyplot(plt)

        knn_model_reg = KNeighborsRegressor(n_neighbors=3, algorithm='ball_tree')
        knn_model_reg.fit(X_train, Y_train)
        knn_model_reg.score(X_test, Y_test)
        y_predict = knn_model_reg.predict(X_test)
        mse = mean_squared_error(Y_test, y_predict)
        score = knn_model_reg.score(X_test, Y_test)

        st.write('Score: ', score)
        st.write('Mean Squared Error: ', mse)
        st.write('Root Mean Square Error: ', np.sqrt(mse))
        st.success('Loading Successful')
        st.write('------')

        if st.checkbox('First 20 items Actual vs Predicted labels'):
            lst_test_val = []
            lst_pred_val = []
            for i in range(1, 20):
                lst_test_val.append(Y_test.values[i])
                lst_pred_val.append(y_predict[i])
            df_1 = pd.DataFrame()
            df_1['Actual'] = lst_test_val
            df_1['Predicted'] = lst_pred_val
            st.write(df_1)






