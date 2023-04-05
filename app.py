import streamlit as st 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import requests 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import re 
import os
import altair as alt
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")


DATA_PATH = os.path.join(dir_of_interest, "data", "laptop_details.csv")
st.set_page_config(layout="wide")

st.title("Laptop Price Prediction")
data_dir = os.path.join(os.getcwd(), "resources", "data")

st.header("This page is regarding laptop price market in :blue[India]")
st.write(" For this we initially webscarpped data from E-commerece :red[Flipkart] website. Sample dataset looks like")
# absolute path to the data file
data_file_path = os.path.join(data_dir, "laptop_details.csv")
df = pd.read_csv(data_file_path)

st.dataframe(df.head())
st.write("In the dataset we need to perform feature engineering to extract Processor, RAM, Storage, Operating System, Brand, Year. With the help of **:red[RegEx]** Operations we extract all those")
#Extracting processor of laptop and adding itto dataframe
regex = r'(?:AMD|Intel|Apple) (?:Core|Ryzen|Celeron|Athlon|Pentium|Dual|M1|M2) (?:[\s\w]|)+Processor'
df['Processor'] = df['Feature'].apply(lambda x : re.findall(regex, x))

#extracting RAM of laptop
regex = r'\d+\sGB[\s\w]+RAM'
df['RAM'] = df['Feature'].apply(lambda x : re.findall(regex,x))

#extracting Opearting System of laptop
regex = r'(?:\d+\sbit\s|)(?:Windows|Mac|DOS|Chrome)[\w\s]+Operating System'
df['Operating System'] = df['Feature'].apply(lambda x : re.findall(regex, x))

#extracting storage of laptop
regex = r'[\d]+\s(?:GB|TB)\s(?:SSD|HDD)'
df['Storage'] = df['Feature'].apply(lambda x: re.findall(regex, x))

#extracting Brand name of laptop
regex = r'^\w+'
df['Brand'] = df['Product'].apply(lambda x : re.findall(regex, x))

#extracting Warranty years 
regex = r'\d Year' 
df['Warranty'] = df['Feature'].apply(lambda x: re.findall(regex, x))

#converting extracting columns from list to string
def list_to_string(array): 
    for col in array: 
        df[col]= df[col].apply(lambda x: ''.join(x)) 
list_to_string(np.array(df.columns[3:])) 

#converting MRP and Rating columns to float type
df['MRP'] = df['MRP'].apply(lambda x : x.replace('₹','').replace(',','')).astype(float)
df['Rating'] = df['Rating'].astype(float) 

st.write("After all initial engineering on the data frame. Dataset looks like as below")
st.dataframe(df.head()) 
st.header("Exploratory Data Analysis") 
st.write("Here we have both catergorical and continous columns in the dataset. So we perform univariate analysis initially and then bivariate with respect to MRP, as MRP is the target variable")
#Count plot for catergorical variables 
def count_plot(column):
    st.write('### Countplot of  '+column)
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.countplot(x=column, data=df, ax= ax)
    plt.xticks(rotation = 90)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig) 
    plt.close()

#catergorical columns
configs = ['Processor', 'RAM', 'Storage', 'Brand', 'Operating System', 'Warranty']
st.subheader('Countplots of Configurations') 
#selecting catergorical column for countplot
counts = st.selectbox("Count plot of", configs)
count_plot(counts)

st.write("From the above count plots we notice maximum counts in dataset for")
st.text("1) Processor: Intel Core i5")
st.text("2) RAM: 8GB DDR4 RAM")
st.text("3) Storage: 512 GB SSD")
st.text("4) Brand: ASUS") 
st.text("5) Operating System: 64 bit Windows 11 Operating System")
st.text("6) Warranty: 1 year")
#histogram plot for MRP and Rating columns
def histogram(column):
# Group data together
    plt.figure(figsize=(8,3))
    sns.histplot(data=df, x=column)
    plt.title("Histogram plot of" + column)
    st.pyplot()
    plt.close()

#continous data columns 
conti_configs = ['MRP', 'Rating']
#selecting continous data variables for histogram plot
hist = st.selectbox("Select column to histogram:", conti_configs)
histogram(hist)
st.write("There were some missing values in Rating, we have imputed missing values with 4 rating")
df['Rating'].fillna(4, inplace=True)
st.subheader("Bivariate analysis")
st.subheader('Boxplot of MRP vs different configuration of laptop')

#boxplot of Catergorical column vs MRP
def boxplot1(cat_col, num_col):
    boxplot = alt.Chart(df).mark_boxplot().encode(
            x=cat_col,
            y=num_col
        )
    boxplot = boxplot.configure_title(
    fontSize=20,         # set font size
    anchor='start',      # set anchor to start
    color='red',         # set title color to red
    offset=10,           # set offset to 10 pixels
    orient='top',        # set title orientation to top    # set the main title text
    )
    st.altair_chart(boxplot, use_container_width=True)

#selecting categorical column to be plotted against MRP
box = st.selectbox("Select columns to have boxplot against MRP:", df.columns[4:])
boxplot1(box, 'MRP') 
st.write("From the boxplots it can be noticed that MRP changes for different Processors, Storage Capacities, RAM, Warranty given, Brand and Operating System.")
st.write("So we need to include this categorical column in the Machine learning model to predict price of Laptop")
st.write("Now we split the data into training and testing sets and build ML models to predict laptop price")

X, y = df[['Processor', 'RAM', 'Operating System', 'Storage', 'Brand', 'Warranty']], df['MRP'] 
#transfoming data using label encoder 
le = LabelEncoder()
X['process_encode']= le.fit_transform(X['Processor'])
X['ram_encode']= le.fit_transform(X['RAM']) 
X['os_encode']= le.fit_transform(X['Operating System'])
X['storage_encode']= le.fit_transform(X['Storage'])
X['brand_encode'] = le.fit_transform(X['Brand'])
X['warr_encode'] = le.fit_transform(X['Warranty'])

#splitting the data into 80% training and 20% testing dataset
trainX, testX, trainY, testY = train_test_split(X.iloc[:,6:], y, test_size= 0.2, random_state=42) 

#fitting Linear regrression model
st.write("Using :red[Linear regressor] model")
model = LinearRegression().fit(trainX, trainY)
print (model, "\n")
# Evaluate the model using the test data
predictions = model.predict(testX)

#calculating RMSE and R-sqaured values
def metrics(testY, predictions):
    mse = mean_squared_error(testY, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(testY, predictions)
    st.write("RMSE value:", rmse)
    st.write("R-squared value:", r2)
    
#plotting actual and predicted variables
def plot(testY, predictions):
    # Plot predicted vs actual
    plt.figure(figsize=(6,4))
    plt.scatter(testY, predictions)
    plt.xlabel('Actual price')
    plt.ylabel('Predicted price')
    plt.title('Laptop price Predictions')
    # overlay the regression line
    z = np.polyfit(testY, predictions, 1)
    p = np.poly1d(z)
    plt.plot(testY,p(testY), color='magenta')
    st.pyplot() 

#metrics and plot for linear regression model
plot(testY, predictions)
metrics(testY, predictions)
st.text("Here we can notice that R-squared value is really low and does not explain the data well. So we build model using boosting algorithms")
st.write("Using :red[Gradient boosting] model")
#building Gradient boosting model
model2 = GradientBoostingRegressor().fit(trainX, trainY)
print (model2, "\n")
# Evaluate the model using the test data
predictions = model2.predict(testX)
#metrics and plot for Gradient boosting regression model
plot(testY, predictions)
metrics(testY, predictions)
st.write("We notice that there is a change in RMSE and R-squared values, and this model performs really better than Linear Regressor model. However, there is chance to improve the model by tuning hyperparameters")
st.write("**Tuning Gradient boosting** model")
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}
st.write("To tune hyperparameters for Gradient boosting model, we tried this combination",param_grid) 
st.write("In those the best parameters were n_estimators=150, learning_rate=0.2, max_depth=3, min_samples_leaf=2, min_samples_split=4")
#model for Gradient boosting after tuning hyperparameters
alg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.2, max_depth=3,min_samples_leaf=2,min_samples_split=4)
# Get the best model
model3=alg.fit(trainX, trainY)
print(model3, "\n")
# Evaluate the model using the test data
predictions = model3.predict(testX)
plot(testY, predictions)
metrics(testY, predictions)
st.write("There is very little change in r-squared and RMSE values. In reality this little also accounts accurate results")
# Define a function to make predictions using the model
def predict_price(model,features):
    feature_values = list(features.values())
    features_arr = np.array(feature_values).reshape(1, -1)  # Reshape to a 2D array
    # Use the trained model to predict the price
    prediction = model.predict(features_arr)[0]
    return prediction

# Create a Streamlit interface for the user to input features and generate a prediction
st.write('# Price Prediction')
#select processor of laptop
processor = st.selectbox('Processor', df['Processor'].unique()) 
processor = X.loc[X['Processor'] == processor, 'process_encode'].iloc[0]
#select RAM of laptop
ram = st.selectbox('RAM', df['RAM'].unique())
ram = X.loc[X['RAM'] == ram, 'ram_encode'].iloc[0]
#select operting system of laptop
operating_system = st.selectbox('Operating System', df['Operating System'].unique())
operating_system = X.loc[X['Operating System'] == operating_system, 'os_encode'].iloc[0]
#select brand of laptop
brand = st.selectbox('Brand', df['Brand'].unique())
brand = X.loc[X['Brand'] == brand, 'brand_encode'].iloc[0]
#select warranty given to laptop
warranty = st.selectbox('Warranty', df['Warranty'].unique())
warranty = X.loc[X['Warranty'] == warranty, 'warr_encode'].iloc[0]
#select storage capacity of laptop
storage = st.selectbox('Storage', df['Storage'].unique())
storage = X.loc[X['Storage'] == storage, 'storage_encode'].iloc[0]

#putting all the new featues in a list
features = {'Processor': processor, 'RAM': ram, 'Operating System':operating_system, 
            'Storage': storage,'Brand': brand, 'Warranty': warranty}

#select model for price prediction
models = ['Linear Regression', 'Gradient Boosting', 'Tuned Gradient Boosting Model']
mod = st.selectbox("Select Model:",models)

if st.button('Predict'):
    # Make the prediction
    #if linear model is selected
    if mod == 'Linear Regression':
        prediction = predict_price(model, features)
    #if gradient boosting model is selected
    elif mod == 'Gradient Boosting':
        prediction = predict_price(model2, features)
    #if tuned gradient boosting model is selected
    elif mod == 'Tuned Gradient Boosting Model':
        prediction = predict_price(model3, features)
    st.write(f'Predicted price: ₹{prediction:,.2f}')