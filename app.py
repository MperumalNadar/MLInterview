import streamlit as st

with st.expander('1.Self introduction'):
    options = ["introduction","Roles & Responsibilities","Why ML selected"]     
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=="introduction":
        st.markdown("""
        Good morning! My name is Murugan. I completed my Bachelor’s Degree in Commerce from Mumbai University in 2016, 
        and I hold an AWS Machine Learning certification.

        I have been working as an IT Analyst at TCS for the past two years, 
        bringing over 6 years of experience in the IT field.
        
        In my current role, I handle the Parle Application Rollout, which involves both Desktop and Mobile application services
        which is targeted to the Distributors in the Client market.
        As Desktop application holds the Database sequences and make the Product 
        distribution hierarchy smoother with the resellers data. Also having the DMS consoles, 
        Database installation procedure to help out client market process. Mobile application used to collect market data and 
        to proceed it to DMS application.
        
        I have good knowledge in the ML Algorithms ,Python Pandas,Data pre-processing and MySQL . 
        
        My key strengths are leadership skills, time management and I tend to be team oriented .
        
        In my personal time, I enjoy reading books and playing cricket. I am fluent in Tamil, English, Hindi, and Marathi, 
        which allows me to communicate effectively across different groups.
        
        I’m currently looking for an opportunity to further develop my career in the Machine Learning field, 
        which is why I’ve selected Wings2 Machine Learning Job Path.
        
        That's a brief about me, and thanks for this opportunity.
        """)
    if selection=="Roles & Responsibilities":
        st.markdown("""
        

* Providing Solutions to the Customer regarding Database Issues with MySQL

* Including DB Recovery, DB Suspect, DB Purging as well as ensuring smooth connectivity between SFA and DMS Application.

* Debugging between Mobile and DMS applications with the help of MySQL Database and some console apps like Xnapp server.

* Also includes the scratch installation for MySQL database and the DB connection with both applications. In some cases, DB purging tool was used for purging the Database to smooth working with high efficiency.

* Managing the application backups with the AWS server and restoration wherever required due to system corrupt Troubleshooting the bugs within the application and finding the best way to solve thase

* for easy application run. Creating the SQL. scripts to automate some procedures to make time afficiency

* Suggesting the new version deploy ideas for better user experience
        
       """ )
    if selection=="Why ML selected":
        st.markdown("""
        I enjoy working with data, finding patterns, and building models that make accurate predictions or automate tasks. 
        The challenge of constantly learning new algorithms and techniques keeps me engaged. Additionally, 
        the impact of my work can often lead to meaningful solutions that improve user experiences or decision-making processes.
        
        
        """)

with st.expander('2.Project Explanation'):
    options = ["emails as spam or not spam",'Movie Hit or Flop Prediction','stock trend prediction in machine learning','crop recommendation',"ML life cycle",'ML syndex','AWS ML syndex','Coffee Shop Sales']     
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=='emails as spam or not spam':
        st.markdown("""Project is a Spam Email Classification System, developed using Python, ad Scikit-learn, techniques. 
        
My role involved designing and implementing the machine learning pipeline
including data preprocessing, feature extraction, training and evaluating models, and fine-tuning them for better accuracy.
        
Key challenges we solved include dealing with the imbalance between spam and non-spam emails
handling noisy and unstructured email content, and improving the efficiency of the model without compromising on false positives.
        
We learned the importance of feature selection and engineering in improving model performance and 
how to fine-tune algorithms like Naive Bayes and Random Forest for better classification. Additionally
we gained insights into evaluating models effectively using precision, recall, and F1-score metrics to ensure optimal email filtering for users.
        
1. Start with the Problem Statement

    "The goal of my project was to build a model to classify emails as spam or not spam to improve email filtering efficiency for users."

2. Explain the Dataset

    "I used a labeled dataset containing 5,000 emails, with features like subject line, word frequencies, and sender details. The data was collected from public repositories."

3. Describe Your Approach

    "I chose a Naive Bayes classifier because it's fast, works well with text data, and is effective for binary classification."

4. Mention Challenges and Solutions

     "One challenge was handling imbalanced data since there were more non-spam emails than spam. I used techniques like oversampling the minority class to balance the dataset."

5. Share Results and Impact

    "The model achieved 95% accuracy with a precision of 92% and recall of 90%. This significantly reduced false positives, ensuring users didn’t miss important emails."

6. Highlight Tools and Skills Used

    "I used Python, libraries like pandas and scikit-learn for preprocessing and modeling, and matplotlib for visualizing results."

7. Relate it to Real-World Use

     "This project is relevant to industries like email services or customer communication platforms, where automated filtering improves efficiency and user satisfaction."

        """)
    if selection=='Coffee Shop Sales':
        st.markdown("""
        - I developed a Linear Regression model to predict the daily revenue of a coffee shop. 
        - I cleaned the data, engineered date features, encoded categorical variables, scaled the features, selected the top 15 features using SelectKBest, and trained the model. 
        - I evaluated it using R², MAE, MSE, and RMSE. The model achieved about 95.7% R² on training and 94.7% R² on testing, showing good prediction accuracy and generalization. 
        - The solution can help the business improve inventory planning, staffing, and promotional decisions."
        """)
    if selection=='Movie Hit or Flop Prediction':
        st.markdown("""
        - My project is about predicting whether a movie will be a Hit or a Flop before its release. This is a Machine Learning classification project."
        - First, I cleaned the data. I removed duplicate rows and unwanted columns like movie ID. I converted important columns into numbers. I also handled missing values, removed negative values, and reduced outliers using the IQR method."
        - Next, I created the target column. If the movie revenue was greater than the budget, I marked it as Hit. Otherwise, I marked it as Flop. Then I removed the budget and revenue columns because they were used to create the target."
        - After that, I prepared the data for machine learning. I took the year, month, and day from the release date. I converted text data into numbers using Label Encoding. For columns like genres, keywords, production companies, and spoken languages, I used MultiLabelBinarizer."
        - Then, I split the data into 80% for training and 20% for testing. I selected the important features using Random Forest Feature Importance and SelectKBest. After that, I scaled the data using StandardScaler."
        - Next, I trained three models: Logistic Regression, Random Forest, and XGBoost. I compared their performance using Accuracy, Precision, Recall, and F1-score. I also used Cross Validation and RandomizedSearchCV to improve the model."
        - Finally, I selected the best model, checked its performance using a confusion matrix, and saved the model using Joblib. This model can help movie producers and investors know whether a movie is likely to become a Hit or a Flop before its release."

        "What was your role?"
            - I did the complete project by myself. I cleaned the data, prepared the features, trained the models, compared the results, improved the best model, and saved the final model."
        """)
    if selection=='stock trend prediction in machine learning':
        st.markdown("""

Q1. What is stock trend prediction in machine learning?
A: Stock trend prediction is the process of forecasting the future movement (upward/downward) of a stock price based on historical data
using machine learning techniques. 
It typically focuses on predicting trends (direction) rather than exact prices.

---

Q2. What type of machine learning is used for stock prediction?

Supervised Learning (e.g., classification for trend, regression for price)

Unsupervised Learning (e.g., clustering stocks)

Time-Series Forecasting (e.g., ARIMA, LSTM for sequential prediction)

Reinforcement Learning (e.g., for trading strategies)
---

Q4. What are the most important features for predicting stock trends?
A:

Historical prices (Open, High, Low, Close, Volume)

Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)

Lagged features (previous N-day prices)

Sentiment scores (if using news or tweets)

Macroeconomic indicators

---
Q11. What are the main challenges in stock trend prediction?

High noise and volatility in the data

Overfitting due to small datasets

Market sentiment and external factors (news, policy changes)

Time lag between prediction and action

---
Q12. Is it possible to consistently beat the stock market with ML?
A: No, consistently beating the market is extremely difficult due to the Efficient Market Hypothesis (EMH).
While ML can help detect patterns, 
it cannot guarantee future performance, especially after transaction costs and market shifts.
""")
    if selection=='crop recommendation':
        st.markdown("""
Q1. What is crop recommendation using machine learning?
A: Crop recommendation is a supervised machine learning task where a model suggests the most suitable crop to grow based on input features like soil nutrients, temperature, humidity, rainfall, and pH level.

---
Q2. What kind of machine learning task is crop recommendation?
A: It is a multi-class classification problem where the model selects one crop from multiple categories based on environmental and soil features.

---
Q3. Which Python libraries are used in crop recommendation projects?

pandas – for data handling

numpy – for numerical operations

scikit-learn – for ML algorithms

matplotlib / seaborn – for visualization

joblib – to save/load models

Flask or Streamlit – for deployment
---
✅ Data & Features
Q4. What are typical features used for crop recommendation?
A:

N – Nitrogen level in soil

P – Phosphorus

K – Potassium

Temperature – in °C

Humidity – in %

pH – soil acidity

Rainfall – in mm
---
Q5. Where can you get crop recommendation datasets?
A: Common sources include:

Kaggle (Crop Recommendation Dataset)

Government agricultural APIs

ICAR (Indian Council of Agricultural Research)

FAO datasets
Or, synthetic data can be generated for prototyping.

✅ Modeling & Evaluation
Q6. Which ML models are suitable for crop recommendation?
A:

Decision Tree

Random Forest (most commonly used)

Naive Bayes

K-Nearest Neighbors (KNN)

Support Vector Machines (SVM)

Neural Networks (for advanced use)
---
Q7. How do you evaluate your crop recommendation model?
A:

Accuracy

Confusion Matrix (to check per-crop performance)

Classification Report (Precision, Recall, F1-score)

Cross-validation (to avoid overfitting)
---
Q8. Why is Random Forest a good model for crop recommendation?
A:

Handles non-linearity well

Works with both small and large datasets

Reduces overfitting through ensemble voting

Provides feature importance
---
""")
    if selection=="ML life cycle":
        st.markdown("""
**1 Data Collection**
Collect the required data from different sources. such as databases, APIs, sensors, surveys, or websites.
**Goal:**
Collect high-quality data that represents the problem accurately.
**2 Data Preprocessing**
Clean and prepare the data before training.
**Tasks include:**
Handling Missing Values
Removing Duplicates
Feature Scaling
Encoding Categorical Variables
Outlier Detection
**Goal:**
Convert raw data into a usable format.        
**3 Exploratory Data Analysis (EDA)**
Analyze and visualize data to understand patterns and relationships.
**Common Tools:**
Matplotlib
Seaborn
Pandas
**Goal:**
find useful information and identify possible problems."
**4 Feature Engineering**
Create or select the most useful features for the model.
**Examples:**
Feature Selection
Feature Extraction
Creating New Features
**Goal:**
Improve model performance.
     """)
    if selection=='ML syndex':
        st.markdown("""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('insurance.csv')#read CSV file
**check if there are any Null values**

sns.heatmap(insurance_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
**check if there are any Null values**

df.isnull().sum()
**Check the dataframe info**

df.info()
**Grouping by region to see any relationship between region and charges**
**Seems like south east region has the highest charges and body mass index**

df_region = df.groupby(by='region').mean()
df_region
**Check unique values in the 'sex' column**

insurance_df['sex'].unique()
**convert categorical variable to numerical**

insurance_df['sex'] = insurance_df['sex'].apply(lambda x: 0 if x == 'female' else 1)

X = insurance_df.drop(columns =['charges'])
y = insurance_df['charges']
X.shape
y.shape
        
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

**scaling the data before feeding the model**

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

**using linear regression model**
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

regresssion_model_sklearn = LinearRegression()
regresssion_model_sklearn.fit(X_train, y_train)

regresssion_model_sklearn_accuracy = regresssion_model_sklearn.score(X_test, y_test)
regresssion_model_sklearn_accuracy

y_predict_orig = scaler_y.inverse_transform(y_predict)
y_test_orig = scaler_y.inverse_transform(y_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 
        
        """)
    if selection=='AWS ML syndex':
        st.markdown("""
**Boto3 is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python**
**Boto3 allows Python developer to write software that makes use of services like Amazon S3 and Amazon EC2**

import sagemaker
import boto3
from sagemaker import Session

**Let's create a Sagemaker session**

sagemaker_session = sagemaker.Session()
bucket = Session().default_bucket() 
prefix = 'linear_learner' # prefix is the subfolder within the bucket.

**Let's get the execution role for the notebook instance.**
**This is the IAM role that you created when you created your notebook instance. You pass the role to the training job.**
**Note that AWS Identity and Access Management (IAM) role that Amazon SageMaker 
can assume to perform tasks on your behalf (for example, reading training results, called model artifacts, 
from the S3 bucket and writing training results to Amazon S3).**

role = sagemaker.get_execution_role()
print(role)

X_train.shape
Y_train.shape

import io # The io module allows for dealing with various types of I/O (text I/O, binary I/O and raw I/O). 
import numpy as np
import sagemaker.amazon.common as smac # sagemaker common libary

**Code below converts the data in numpy array format to RecordIO format
This is the format required by Sagemaker Linear Learner**

buf = io.BytesIO() # create an in-memory byte array (buf is a buffer I will be writing to)
smac.write_numpy_to_dense_tensor(buf, X_train, y_train.reshape(-1))
buf.seek(0) 

**When you write to in-memory byte arrays, it increments 1 every time you write to it
Let's reset that back to zero**

import os

**Code to upload RecordIO data to S3
 
Key refers to the name of the file**  
key = 'linear-train-data'

**The following code uploads the data in record-io format to S3 bucket to be accessed later for training**

boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)

**Let's print out the training data location in s3**

s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))

**create an output placeholder in S3 bucket to store the linear learner output**

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('Training artifacts will be uploaded to: {}'.format(output_location))


**This code is used to get the training container of sagemaker built-in algorithms
all we have to do is to specify the name of the algorithm, that we want to use**
**Let's obtain a reference to the linearLearner container image
Note that all regression models are named estimators
You don't have to specify (hardcode) the region, get_image_uri will get the current region name using boto3.Session**


from sagemaker.amazon.amazon_estimator import get_image_uri

container = get_image_uri(boto3.Session().region_name, 'linear-learner')

**We have pass in the container, the type of instance that we would like to use for training 
output path and sagemaker session into the Estimator.**
**We can also specify how many instances we would like to use for training**

linear = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count = 1, 
                                       train_instance_type = 'ml.c4.xlarge',
                                       output_path = output_location,
                                       sagemaker_session = sagemaker_session)


**We can tune parameters like the number of features that we are passing in, type of predictor like 'regressor' or 
'classifier', mini batch size, epochs**
**Train 32 different versions of the model and will get the best out of them (built-in parameters optimization!)**

linear.set_hyperparameters(feature_dim = 8,
                           predictor_type = 'regressor',
                           mini_batch_size = 100,
                           epochs = 100,
                           num_models = 32,
                           loss = 'absolute_loss')

**Now we are ready to pass in the training data from S3 to train the linear learner model**

linear.fit({'train': s3_train_data})

**Let's see the progress using cloudwatch logs**

**Deploying the model to perform inference**

linear_regressor = linear.deploy(initial_instance_count = 1,
                                          instance_type = 'ml.m4.xlarge')

**Since the result is in json format, we access the scores by iterating through the scores in the predictions**

predictions = np.array([r['score'] for r in result['predictions']])

y_predict_orig = scaler_y.inverse_transform(predictions)
y_test_orig = scaler_y.inverse_transform(y_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 

**Delete the end-point**

linear_regressor.delete_endpoint()
        """)
#####################################################################################
with st.expander('3.Defination Python,ML,DL,DL,NL''streamlit'):

    options = ["DS", "ML", "WHY ML","Supervised & Unsupervised learning",'PYTHON','Numpy','Scikit-learn','joblib','predict','DL','streamlit']
        
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=='DS':
        st.markdown("""
                    - Data Science is an interdisciplinary field 
                    - focused on extracting knowledge and insights from data
                    - using scientific methods, algorithms, and systems. 
                    - It combines aspects of statistics, computer science, and domain expertise.
        """)
    if selection=='ML':
        st.markdown("""
                    - Machine Learning (ML) is a branch of artificial intelligence (Al) 
                    - that focuses on developing algorithms 
                    - that allow computers to learn from and make decisions or predictions based on data. 
                    - Instead of being explicitly programmed to perform a task, 
                    - ML algorithms use patterns and insights from data to improve performance over time.
        """)
    if selection=='WHY ML':
        st.markdown("""
        machine learning is basically teaching computers to learn on their own. right
        so what machine learning basically done is 
        it will show the computer hundreds of pictures of say apples and oranges ok 
        so machine learning is basically to pick these two fruits and show them to the computer now what machine learning is doing here is basically it's not telling that apples are going to be red and a round
        similarly, for oranges are there going to be Orange in color and also going to be a round.
        instead of giving certain rules
        it is going to let the computer figure out by itself it is going through hundreds of images and pattern by itself so this a new kind of apple a new kind of orange is shown to the computer is going to be able to figure out them.
        so this the basic idea of behind of machine learning 
        basically reduces the repetitive manual work like. so say you want to filter out the spam emails. right 
        so you don't need to do that manually rather machine learning is going to do that.
        based on certain key words .so basically on email contains like free and offers keywords like that so based on that machine learning is going to help you identify. that these events could be spam machine learning reduces repetitive manual work. it has businesses make data driven decisions it powers personalized recommendations like Netflix and Amazon. notice every time you open Netflix. will Suggested movie series of the Other based on your previous watches. that is again machine learning behind it drives innovations like self-driving Cars and medical diagnosis. again it is not just limited to the basic applications we see like Netflix. infect it is seen every ware like.
        Self-driving Cars and medical diagnosis.

        """)
    if selection=='Supervised & Unsupervised learning':
        st.markdown("""
        - In Supervised Learning, the model learns from labeled data, where the correct output is already known.
        - Supervised learning can be categorized in Classification and Regression problems.
        - It includes various algorithms such as 
Linear Regression, Logistic Regression, Support Vector Machine, Multi-class Classification, Decision tree, 
.
        **Unsupervised Learning**
       - In Unsupervised Learning, the model learns from unlabeled data and discovers hidden patterns.
       - Unsupervised Learning can be classified in Clustering and Associations problems.
       - it includes various algorithms such as Clustering, KNN, and Apriori algorithm.

        """)
    if selection=='PYTHON':
        st.markdown("""
        python is object oriented and high level programming language that has become the de factor standard for data science 
        computational applications 
        it is designed to be highly readable language which uses a lot of English keywords 
        it also has pure syntactical Constructions compared to other programming languages

        it is an integrated language which means that it is processed at runtime by the interpreter .
        and there is no need to compile the programs before execution it is interactive and object oriented 
        which enables the users to encapsulate code within objects .

        python is portable an as cross-platform compatibility on various operating systems 
        it can be run on different hardware platforms and has the same interface on all platforms

         Python also support graphical user interface applications on various operating systems.

         its provide optimize  structure and support large and scalable programming .
        """)
    if selection=='Numpy':
        st.markdown("""
        NumPy stands for Numerical Python.
        
        when was created as a centralized library for working with arrays. 
        it also has functions for working in domains of linear algebra matrices and etc. .
        
        traditionally python as list that can serve purpose of array but as slow to process.
        
         something which Numpy addresses providing an array object that is up to 50 times faster than a traditional Python list.
        
         the array object in Numpy is called as nd-array and as a host of supporting functions that are very easy to work with .
        
        numpy  stored in  singular continuous place in memory  like traditional Python list .
        
        so process can access manipulating array a lot more effectively . they can making process faster 
        
        as it is optimized to work with the latest CPU architectures and libraries are predominantly returned in C or C plus plus. 
        
        numpy support varies Data type .such as  integer Boolean float date string amongst other .
        
        users create array define data type convert data type on an existing array .
        
        check whether arrays have their own data .
        
        check the shape of an array combine or split arrays.
        
         and many such features, Numpy enables sorting of an array in ascending or descending 
        
        where the values could be numeric or alphabetical.
        """)
    if selection=='Scikit-learn':
        st.markdown("""
Scikit-learn is a popular machine learning library in
Python that provides a wide range of algorithms and
tools for various tasks such as classification,
regression, clustering, dimensionality reduction, and
model evaluation. It is widely used for building
machine learning models and pipelines.
        """)
    if selection=='joblib':
        st.markdown("""
You can save a trained Scikit-learn model to disk
using the joblib module's dump() function. To load
a saved model, you can use the load() function.
For example:
Ans:
from sklearn.externals import joblib  
# Save the model 
joblib.dump(model, )  
# Load the model 
loaded_model = joblib.load( )
        """)
    if selection=='predict':
        st.markdown("""

Once you have trained a model in Scikit-learn, you
can make predictions on new data using the
predict() method. For example:

model = LinearRegression() 
model.fit(X_train, y_train) 
predictions = model.predict(X_test)
        """)
    if selection=='streamlit':
        st.markdown("""
Streamlit is an open-source Python framework for building interactive web applications for machine learning and data science projects.
It allows you to create web-based user interfaces with just a few lines of Python code.

**Why Use Streamlit?**

**Easy to use:** 
No need for frontend development (HTML, CSS, JavaScript).

**Fast development:** 
Convert Python scripts into web apps in minutes.

**Integration with ML & Data Science:** 
Supports Pandas, Matplotlib, Plotly, Scikit-learn, TensorFlow, etc.
Interactive widgets: Sliders, buttons, text inputs, file uploaders, etc.
        """)
    if selection=='DL':
        st.markdown("""
 Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model
 complex patterns in data. It is the foundation of modern AI and is widely used in computer vision, 
 natural language processing (NLP), speech recognition, and autonomous systems.
 
        """)
#####################################################################################
with st.expander('4.AWS'):
        options3 = ["Sagemaker","Sagemaker_COMPONENTS","What is AWS","What is cloud computing ","WHAT IS A REGION?","AVAILABILITY ZONE","AWS S3","S3 Type","AWS IAM","AWS Lambda","AWS Athena","AWS kinesis"]
        selectio = st.segmented_control("", options3, selection_mode="single")
        if selectio=="What is AWS":
            st.markdown("""
            Amazon Web Services (AWS) is the world’s top cloud platform. 
            offers more than 165 fully featured services . 
            adopted by millions of customers globally including small and large scale enterprises. 
            provides services for broad range of applications such as: 

            """)
        if selectio=="What is cloud computing ":
            st.markdown("""   
            Cloud computing is the on-demand delivery of services such as compute 
            and storage over the Internet with pay-as-you-go pricing. 
            
            Simply put, instead of buying a physical server or a computer, you can lease it!

            """)
        if selectio=="WHAT IS A REGION?":
            st.markdown("""   
            An AWS Region is a geographical location that contains a number of availability zones (data centers).
            
            Every region is physically separate from all other regions. 
            
            Every region has its own independent power and water supply.
            
            Regions are important to ensure:
             	(1) Data compliance
            	(2) Latency (data centers are 	placed close to users to reduce 	latency)  
             
            us-east-1 is the largest AWS region and contains of five zones. 


            """)
        if selectio=="AVAILABILITY ZONE":
            st.markdown("""   
            An AWS availability zone is a logical data center that is located in a certain region. 

            There are two or more availability zones in every AWS region. 

            Note: A data center consists of bunch of servers

            """)
        if selectio=="Sagemaker":
            st.markdown("""
    Amazon SageMaker is a fully-managed machine learning workflow platform that provides services on data labeling, 
    model building, training, tuning and deployment.
            
    SageMaker allows data scientists and developers to build scalable AI/ML models easily and efficiently.
            
    Models could be deployed in production at a much faster rate and with a fraction of the cost. 

    
    key features include:
        
    **1. Model Building:**
        
        * SageMaker offers data labeling service

        * Prebuilt available notebooks with state of the art algorithms 
        on AWS marketplace
        
    **2. Model Training:**
        
        * Train models using EC2 instances (on-demand and spot)

        * Manage environments for training

        * Hyperparameters optimization for model tuning

    
   **3. Model Deployment:**
        
        * Easily deploy and scale models

        * Autoscaling with 75% savings
        
    
 Additionally, SageMaker includes features like model monitoring, explainability tools, and 
 support for custom machine learning frameworks such as TensorFlow, PyTorch, and Scikit-learn.
    
 It streamlines the end-to-end machine learning workflow, enabling faster and more efficient production deployments.
    
            
            """)
        if selectio=="Sagemaker_COMPONENTS":
            st.markdown("""
Two components are present in Amazon SageMaker: 
    Model training 
    Model deployment.
                
To start training an ML model using Amazon SageMaker, we will need to create a training job.
            
**Amazon S3 bucket URL (training data):** where the training data is located.
**Compute resources:** Amazon SageMaker will train the model using instances managed by Amazon SageMaker.
**Amazon S3 bucket URL (Output):** this bucket will host the output from the training.
**Amazon Elastic Container Registry path:** where the training code is stored. 
                
SageMaker launches an ML compute instances once a training job is initiated. 
SageMaker uses: (1) training code and (2) training dataset to train the model. 
SageMaker saves the trained model artifacts in an S3 bucket.
            """)
        if selectio=="AWS S3":
            st.markdown("""
Amazon Simple Storage Service (Amazon S3) is a storage service that allows enterprises/individuals 
to store and protect any amount of data.

Amazon S3 offers numerous enhanced features such as:

(1) Scalability

(2) Data availability

(3) Security

(4) Performance

Amazon S3 is extremely easy to use and allows enterprises to organize their data and configure finely-tuned access controls.

Amazon S3 extremely durable to 99.999999999% (11 9's).

Amazon S3 is 99.9% available.
            """)
        if selectio=="S3 Type":
            st.markdown("""


1. Standard Bucket (S3 Standard)

Purpose: General-purpose storage for frequently accessed data.

Use Case: Websites, mobile applications, real-time big data analytics.

Durability: 99.999999999% (11 nines).

Availability: 99.99%.



2. S3 Intelligent-Tiering

Purpose: Automatically moves data between two access tiers (frequent and infrequent access) based on usage patterns.

Use Case: Unpredictable data access patterns.

Durability: 99.999999999%.

Availability: 99.9%.

Cost-Effective: Reduces cost by automatically shifting data to the cheaper tier when not accessed.


3. S3 Standard-IA (Infrequent Access)

Purpose: For data that is less frequently accessed but still requires rapid access when needed.

Use Case: Backup, disaster recovery, long-term data storage.

Durability: 99.999999999%.

Availability: 99.9%.

Lower Storage Cost: Higher retrieval cost.


4. S3 One Zone-IA (Infrequent Access)

Purpose: Similar to Standard-IA but data is stored in a single Availability Zone (AZ).

Use Case: Secondary backups, easily re-creatable data.

Durability: 99.999999999%.

Availability: 99.5%.

Lower Cost: Lower availability and resiliency.


5. S3 Glacier

Purpose: Long-term archival storage with low retrieval frequency.

Use Case: Compliance data, long-term backups, historical records.

Retrieval Time: Minutes to hours depending on the retrieval option.

Durability: 99.999999999%.



6. S3 Glacier Deep Archive

Purpose: Lowest-cost storage for long-term retention of data that is rarely accessed.

Use Case: Regulatory archives, digital preservation.

Retrieval Time: Up to 12 hours.

Durability: 99.999999999%.


7. S3 Outposts

Purpose: For customers who require data to remain on-premise due to compliance or low-latency needs.

Use Case: Local data processing and storage.

Durability: Same as other S3 classes, but within the user's data center.

            
            """)
        if selectio=="AWS IAM":
            st.markdown("""
            AWS IAM (Identity and access management

All the access relates to with help of AWS IAM

1 USER

-- Authentication with user creation

2 Policies--"

if you user, you required S3 Bucket Access with the help of policies you can attached .

3 Group

suppose your working in Big organisation The devops engineer will create department wise group. and required Access to the that group.

4 ROLE

if multiple services communicate with each other, Then create one AWS IAM ROLE
            """)
        if selectio=="AWS Lambda":
            st.markdown("""
            1 Aws Lambda is a Serverless, Compute Service.that lets you run code for virtually 
            any type of application or backend service without managing Servers.

        2 you can trigger. Lambda From wer 200 Aws services and only pay for what you use.
Benefits:

1 No need for managing the server.
2 Auto Scalling
3 pay as you Go
4 Performance optimization.

                Interview Question

Difference between EC2 and Lambda?

EC2 instance is server Computation engine we try to create a server and we only manage all the servers

IN THE CASE OF LAMBDA.

serverless computation engine, dont worry about any server creation, 
and every thing managed by AWS only we can only focus on code and run with help of AWS.
            """)
        if selectio=="AWS Athena":
            st.markdown("""
Amazon Athena is query service that makes easy to analyze data directly from:
S3 using Standard SQL

-- Athena is not database or not dada warehousy

-- it is designed to allow you write fast Sql query data without having to move it.

        pricing

with pert query billing, you can get started quickly and pay only for the data Scanned by queries you run

You are charged for the number of bytes scanned per query, rounded up to the nearest megabyte, with a 10 MB minimum per query.
            """)
        if selectio=="AWS kinesis":
            st.markdown("""
            Amazon kinesis

1 Amazon kinesis Data Streams to collect and process large streams of data record in real time.

2 you Can use kinesis Data Streams for rapid and continuous data in take and aggregation

3 Amazon kinesis Data used case include:-

        1 IT infrastructure log data,
        
        2 application bgs,
        
        3 Social media,
        
        4 market data feeds and
        
        5 web click stream data
        
            Type of kinesis 

1 kinesis streams:-

generally we collect and process real time data.

2 kinesis firehouse:-

generally we collect and process near real time data
because of (6o sec) and stored in different Services, ex S3.

3 Kinesis Analytics:-

Supposed real time data coming and based on that certain analysis in that kind on Scenario we can use this kinesis Analytics.
            """)
#####################################################################################  
with st.expander('5.Data Analysis(pandas,Numpy)'):
    options = ['Python','Pandas','NumPy','DataFrame and a Series','Matplotlib','data cleaning','histogram','data cleaning','data visualization','loc and iloc','one-hot encoding','map() function',"Bias & Varience", "Balancen & Imbalance", "FEATURE SCALING NORMALI-STANDARDIZATION",'Underfitting & Overfitting','Missing Data','Cross-validation','REGULARIZATION L1 L2','outliers','Bagging & boosting','Ensemble learning','Hyperparameter tuning','Feature enginnering']
    
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=='Python':
        st.markdown("""
        Python is a simple and powerful programming language used to write code for many tasks like automation, web apps, and Machine Learning.
        It is easy to learn because its syntax is close to normal English.
        """)
    if selection=='Pandas':
        st.markdown("""
        Pandas is a Python library used for data analysis and data manipulation.
        It provides DataFrame and Series to work with structured data like Excel.
        We use it to clean, filter, and analyze data for insights.

        """)
    if selection=='NumPy':
        st.markdown("""
        NumPy is a Python library used for numerical computations.
        It provides fast array operations and supports mathematical functions.
        It is widely used in machine learning for handling large datasets efficiently.
        """)
    if selection=='DataFrame and a Series':
        st.markdown("""
        A DataFrame is a 2-dimensional labeled data structure with columns of potentially
        different types. It can be thought of as a table with rows and columns. A Series, on the
        other hand, is a 1-dimensional labeled array capable of holding any data type.
                    """)
    if selection=='Matplotlib':
        st.markdown("""
        Matplotlib is a Python library used for data visualization. It provides a wide variety of
        plots and charts to visualize data, including line plots, bar plots, histograms, and scatter
        plots.

        import matplotlib.pyplot as plt

        """)
    if selection=='data cleaning':
        st.markdown("""
        Data cleaning is the process of identifying and correcting errors, inconsistencies, and
        missing values in a dataset to improve its quality and reliability for analysis. It involves
        tasks such as removing duplicates, handling missing data, and correcting formatting
        issues.

                    """)
    if selection=='histogram':
        st.markdown("""
        A histogram is a graphical representation of the distribution of numerical data. It
        consists of a series of bars, where each bar represents a range of values and the height of
        the bar represents the frequency of values within that range. Histograms are commonly
        used to visualize the frequency distribution of a dataset.

        import matplotlib.pyplot as plt
         plt.hist(data, bins=10) 
        """)
    if selection=='data visualization':
        st.markdown("""
        The purpose of data visualization is to communicate information and insights from
        data effectively through graphical representations. It allows analysts to explore patterns,
        trends, and relationships in the data, as well as to communicate findings to stakeholders
        in a clear and compelling manner
                    """)
    if selection=='loc and iloc':
        st.markdown("""
        loc is used for label-based indexing, where you specify the row and column labels,
        while iloc is used for integer-based indexing, where you specify the row and column
        indices.
        """)
    if selection=='one-hot encoding':
        st.markdown("""
        Interview answer (3 lines)
One Hot Encoding is used to convert categorical data into binary format.
Each category becomes a separate column with 0 and 1 values.
It helps machine learning models process non-numeric data correctly.
5. Why we use it and when
Why:
ML models cannot understand text
Avoids wrong relationships (like 1 < 2 confusion)
When:
When data is categorical (nominal) like city, product type, color
When categories have no order
                    """)
    if selection=='map() function':
        st.markdown("""
        The map() function applies a given function to each item of an iterable and returns a
        list of the results. In data analysis, it's useful for applying functions element-wise to data
        structures like lists or Pandas Series.
        """)
        
        
#####################################################################

    if selection=='Bias & Varience':
        st.markdown("""
        Bias 
        
        Bias happens when the model is too simple. 
        It cannot learn the important patterns from the data.
        This is called underfitting
        """)
        st.markdown("""
        VARIANCE
        
        Variance happens when the model is too complex. 
        It learns the training data too well, including the noise, so it performs poorly on new data. 
        This is called overfitting.
        """)
        st.markdown("""
        Bias-Variance Tradeoff

        
        Bias-Variance Tradeoff means balancing underfitting and overfitting so that the model works well on both training data and new data."
        """)
    if selection=='Underfitting & Overfitting':
        st.markdown("""UNDERFITTING

Underfitting happens when a model is too simple to capture the underlying patterns in the data and Grow occurs 
when the model's performance is poor not only on the training data but also on new, unseen data
         """)
        st.markdown("""OVERFITTING

Overfitting happens when a model becomes too complex and learns not only the underlying patterns but also the noise or random fluctuations in the training data. 
This causes the model to perform very well on the training data but poorly on new, unseen data
         """)
    
        st.markdown("""?????How do you handle the problem of overfitting in machine learning models????

Overfitting can be mitigated by using techniques like cross-validation, 
regularisation, early stopping, and reducing model complexity.
         """)
    if selection=='Balancen & Imbalance':
        st.markdown("""
        Imbalanced classes occur when one class (e.g., "yes") has significantly more samples than another class (e.g., "no").

        Example: Detecting fraud: 98% of transactions are genuine, only 2% are fraudulent.
        
        Imbalanced datasets can be handled using techniques like 

                - oversampling, 
                - undersampling, or using algorithms
                - designed for imbalanced data such as SMOTE 
                (Synthetic Minority Over-sampling Technique).
         """)
        st.markdown("""OVER SAMPLING

Oversampling is used when the quantity of data is insufficient. 
It tries to balance dataset by increasing the size of rare samples

        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


         """)
        st.markdown("""UNDER SAMPLING

This method is used when quantity of data is sufficient. 
By keeping all samples in the rare class and
randomly selecting an equal number of samples in the abundant class, 
a balanced new dataset can be retrieved for further modelling.

         from imblearn.under_sampling import RandomUnderSampler
         rus = RandomUnderSampler(random_state=42)
         X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

         """)
    if selection=='FEATURE SCALING NORMALI-STANDARDIZATION':
        st.markdown("""FEATURE SCALING

Feature scaling is a preprocessing technique in machine learning used to standardize or normalize the range of independent variables or features of data. 
The goal of feature scaling is to ensure that all features have similar scales or magnitudes.
         """)
        st.markdown("""MIN-MAX SCALING (NORMALIZATION)


This method scales the features to a specific range, typically between 0 and 1.
It's like changing all your ingredients to be on a scale from 0 to 1, 
where O means the smallest amount, and 1 means the largest amount. 
Just like making sure all your ingredients are in a similar range.


          from sklearn.preprocessing import MinMaxScaler
          scaler = MinMaxScaler()
          normalized_data = scaler.fit_transform(data)
          normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
          print("Normalized Data:\n", normalized_df)
         """)
        st.markdown("""Z-score SCALING (STANDARDIZATION)

Standardization transforms features to have a mean of 0 and a standard deviation of 1. 
This one makes all your ingredients have an average (mean) of 0 and a standard deviation of 1. 
It's like making sure all your ingredients are centered around a common point and have similar Grow spreads.

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data)
        standardized_df = pd.DataFrame(standardized_data, columns=data.columns)
         print("Standardized Data:\n", standardized_df)
         """)
    if selection=='Missing Data':
        st.markdown("""

Missing data can be handled by techniques such as mean/median imputation, mode imputation, 
or using advanced methods like multiple imputation or K-Nearest Neighbors imputation.
         """)
    if selection=='Cross-validation':

        st.markdown("""Cross Validationi technique to evaluate a machine lear..
        model by training and testing it multiple times on different parts of the data.
        It gives a more reliable performance than testing only once.
                         """)
    if selection=='REGULARIZATION L1 L2':
        st.markdown("""
        
        Regularization is a technique used to avoid overfitting.
        It controls the model so it does not learn too much from the training data. 
        This helps the model work better on new data. 
        There are two types: L1 and L2.
        L1 (LASSO) removes unimportant features by making some coefficients zero.
        L2 (RIDGE) keeps all features but makes their coefficients smaller. 
        I use L1 when some features are not important, and 
        I use L2 when all features are important but I want to reduce overfitting.

        what is the Albha?
        Alpha is the regularization strength. A larger alpha means a stronger penalty,
        which makes the model simpler. A smaller alpha means a weaker penalty

        When to choose L1 LASSO REGRESSION ? 
        
        If you believe that some features are not important and you can afford to lose them, then L1 regularization is a good choice. 
        The output might become sparse since some features might have been removed.
                  from sklearn.linear_model import Lasso
                  lasso = Lasso(alpha=0.1)  # Regularization strength, adjust alpha as needed
                  lasso.fit(X_train, y_train)
        
        When to choose L2 RIDGE REGRESSION? 
        
        If you believe that all features are important and you’d like to keep them but weigh them accordingly.
                  from sklearn.linear_model import Ridge
                  ridge = Ridge(alpha=1.0)  # Regularization strength, adjust alpha as needed
                  ridge.fit(X_train, y_train)


        """)
    if selection=='outliers':
        st.markdown("""
        Outliers are data points that significantly differ from the majority of the data in a dataset. 
        These data points are unusual or exceptional in some way and can have a 
        substantial impact on data analysis and statistical modelin


        Removing outliers 
        from data is an important step to ensure the quality and accuracy of machine learning models.

        IQR (Interquartile Range) Method
How It Works: The IQR is the range between the 25th (Q1) and 75th (Q3) percentiles of the data. Outliers are typically considered to be any points outside the range:
Lower bound: 𝑄1−1.5×𝐼𝑄𝑅

Upper bound:  Q3+1.5×IQR

How to Apply:
Calculate the IQR.
Remove data points below the lower bound or above the upper bound.

        # Calculate Q1, Q3, and IQR
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = (data < lower_bound) | (data > upper_bound)
        
        # Remove outliers
        cleaned_data = data[~outliers]
        print("Cleaned data:", cleaned_data)           


        A boxplot is a quick way to visualize the spread of data and identify potential outliers. 
        Points that fall outside the "whiskers" of the boxplot can be considered outliers.
        
            You can use matplotlib or seaborn to plot a boxplot.
            
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            # Example data
            data = [10, 12, 14, 18, 19, 100, 23, 22, 21]
            
            # Create a boxplot
            sns.boxplot(data=data)
            plt.show()
        """)
    if selection=='Ensemble learning':
        st.markdown("""
    
        Ensemble Learning is the process of combining predictions from multiple Machine Learning models to produce a more accurate and robust final prediction.

        Ensemble methods help to:

        - Improve prediction accuracy
        - Reduce overfitting 
        - Increase model stability

Generalize better on unseen data
        """)
    if selection=='Bagging & boosting':
        st.markdown("""
        What is Bagging ?
        Bagging trains multiple models independently and in parallel using different samples of the data.
        The final output is made by combining their predictions.
        Example: Random Forest
        Bagging mainly helps reduce variance and avoids overfitting.

                  
        """)
        st.markdown("""
        What is Bootinng 
        
        Boosting trains models sequentially.
        Each new model tries to correct the mistakes made by the previous model.
        Example: AdaBoost, Gradient Boosting, XGBoost, LightGBM Boosting mainly helps reduce bias and improves model performance.
        """)
    if selection=='Hyperparameter tuning':
        st.markdown("""
        Hyperparameters are settings that control how a Machine Learning algorithm learns. 
        Unlike model parameters, they are set before training and are not learned from the data.


        - Grid Search is a hyperparameter tuning method. 
        We give it a list of different values for the hyperparameters, and it tries every possible combination. 
        After testing all combinations, it selects the one that gives the best model performance. 
        The advantage is that it usually finds the best combination, but the disadvantage is that it takes more time and requires more computing power."
        
        - Random Search is also a hyperparameter tuning method.
        it randomly selects and tests only some combinations. 
        This makes it much faster than Grid Search. 
        It works well when there are many hyperparameters, 
        but it may not always find the absolute best combination."
        """)
    if selection=='Feature enginnering':
        st.markdown("""
        What is Feature Engineering?

         Feature Engineering is the process of selecting, transforming, 
         and creating meaningful features from raw data to improve model performance.

         WHY IS IT IMPORTANT?
         -Models learn faster
         -Predictions become
         -Reduces overfitting
         -Improves generalization
        """)
    
with st.expander('7.ML Algorithms'):    
    st.markdown("""
    Classification is used to predict discrete categories 
    while regression is used to predict continuous quantities.
        """)
    options = ["Linear regression",'Advantage and DisAdvantage','Feature Scaling is required?']
    
    selection = st.segmented_control("1.Linear Regression", options, selection_mode="single")
 
    if selection=='Linear regression':
        st.markdown("""
        Linear regression is a method to find the best straight line that fits data points. 
        It helps us understand how one variable changes with another. 
        This line allows us to make predictions and see the overall trend in the data.
        """)
        st.write('#########')
        st.markdown("""
        With the line equation (Y = 10x + 50), we can make predictions. 
        For example, if a student studies for 6 hours (X = 6), 
        we can estimate their exam score by plugging the value into the equation: Y = 10 * 6 + 50 = 110.
        """)
    if selection=='Advantage and DisAdvantage':
        st.markdown("""
                    Advantages 
            
                    - Linear regression performs exceptionally well for linearly separable data
                    - Easy to implement and train the model
                    - It can handle overfitting using dimensionlity reduction techniques and cross validation and regularization
        """)
        st.markdown("""
                    Disadvantages
                    
                    - Sometimes Lot of Feature Engineering Is required
                    - If the independent features are correlated it may affect performance
                    - It is often quite prone to noise and overfitting
        """)
    if selection=='Feature Scaling is required?':
        st.write("Yes")

################################################################################################################


    options = [ "Logistic regression",'Advantage and DisAdvantage','Feature Scaling is required?']
    
    selection = st.segmented_control("2 Logistic Regression", options, selection_mode="single")
    
    if selection=='Logistic regression':
        st.markdown("""
        Logistic Regression is a popular machine learning algorithm used for binary classification tasks, 
        where the goal is to predict one of two possible outcomes, 
        such as whether an email is spam or not, or whether a customer will churn or stay.

        "Real-World Example:"  
        
        Imagine predicting whether a customer will buy a product (yes/no). 
        Logistic regression would take features like age, income, and browsing behavior and
        output the probability of the customer making a purchase. 
        If the probability is greater than a threshold (say 0.7), we predict they will buy the product.
        
        """)
    if selection=='Advantage and DisAdvantage':
        st.markdown("""
                    Advantages 
            
                    - Logistic Regression Are very easy to understand
                    - It requires less training
                    - Good accuracy for many simple data sets and it performs well when the dataset is linearly separable.
                    - It makes no assumptions about distributions of classes in feature space.
                    - Logistic regression is less inclined to over-fitting but it can overfit in high dimensional datasets.One may consider Regularization (L1 and L2) techniques to avoid over-fittingin these scenarios.
                    - Logistic regression is easier to implement, interpret, and very efficient to train.
        """)
        st.markdown("""
                    Disadvantages
                    
                    - Sometimes Lot of Feature Engineering Is required
                    
                    - If the independent features are correlated it may affect performance
                    
                    - It is often quite prone to noise and overfitting
                    
                    - If the number of observations is lesser than the number of features, Logistic Regression should not be used, otherwise,
                    it may lead to overfitting.
                    
                    - Non-linear problems can’t be solved with logistic regression because it has a linear decision surface. 
                    Linearly separable data is rarely found in real-world scenarios.
                    
                    - It is tough to obtain complex relationships using logistic regression. 
                    More powerful and compact algorithms such as Neural Networks can easily outperform this algorithm.
                    
                    - In Linear Regression independent and dependent variables are related linearly. 
                    But Logistic Regression needs that independent variables are linearly related to the log odds (log(p/(1-p)).
        """)
    if selection=='Feature Scaling is required?':
        st.write("Yes")
        
##############################################################################
    options1 = ["Decision Tree", "Real_Word eg","Advantage and DisAdvantage", "How Choose Root Node", "Entropy Formula","Gini index"]
    selection1 = st.segmented_control("3.Decision Trea", options1, selection_mode="single")
    if selection1=='Decision Tree':
        st.markdown("""
        A Decision Tree is a supervised machine learning algorithm used for both classification and regression.
        It makes predictions by asking a series of questions about the data. Each question splits the dataset into smaller groups based on the feature that gives the best split. 
        This process continues until the model reaches a leaf node, which gives the final prediction.
        -You can think of it like a flowchart where every decision leads closer to the final answer

        

        """)
 
    if selection1=='Real_Word eg':
        st.markdown("""
        A Decision Tree works like a human making decisions.
        It asks simple questions step by step, and each answer helps it reach the final prediction.
        """)
    if selection1=='Advantage and DisAdvantage':
        st.markdown("""
                    Advantages
                    - Easy to understand and visualize.
                    - Works with both numerical and categorical data.
                    - Requires very little data preprocessing.
        """)
        st.markdown("""
                    Disadvantages of Decision Trees
                    - It can easily overfit if the tree becomes too deep.
                    - Small changes in the data can produce a different tree.
                    - A single Decision Tree is usually less accurate than ensemble methods like Random Forest.
        """)

    if selection1=='How Choose Root Node':
        st.markdown(""" Find the Decision Tree Root node we find the best fit for the solution to get the maximum information GAIN
        We use various features like the Guinea or Enropy
        """)
    if selection1=='Entropy Formula':
        st.markdown(""" Entropy is formula is Negation of Summation of the Probability of Instance into Log of Probability
        """)
    if selection1=='Gini index':
        st.markdown(""" 
        The Gini index is used to measure the impurity or the homogeneity of a node in a decision tree, 
        helping to determine the best split for creating a more accurate decision tree.
        """)
####################################################################
    options2 = ["Random Forest",'Advantage and DisAdvantage','Hyperparameters in Random Forest','Feature Scaling is required?']
    selection2 = st.segmented_control("4.Random Forest", options2, selection_mode="single")
    if selection2=='Random Forest':
        st.markdown("""
                    "Random Forest is a supervised machine learning algorithm that combines multiple Decision Trees to make a prediction. 
                    Instead of depending on one tree, it builds many trees using different random samples of the data and features.

                    - For classification, it takes the majority vote from all the trees. 
                    - For regression, it takes the average of all the predictions.
                    
                       """)
    if selection2=='Advantage and DisAdvantage':
        st.markdown("""
                     Advantages
                    - Doesn't Overfit

                    - Favourite algorithm for Kaggle competition

                    - Less Parameter Tuning required

                    - Decision Tree can handle both continuous and categorical variables.

                    -  No feature scaling (standardization and normalization) required in case of Random Forest as it uses DEcision Tree internally
                    - Suitable for any kind of ML problems
                    """)
        st.markdown("""
                    Disadvantages
                    
                    - Biased With features having many categories

                    - Biased in multiclass classification problems towards more frequent classes.
        """)
    if selection2=='Hyperparameters in Random Forest':
        st.markdown("""
                        Key Hyperparameters in Random Forest

                            1. Number of Trees (n_estimators):
                            How many decision trees to use.
                            
                            2. Max Depth: 
                            Controls how deep each tree grows.
                            
                            3. Min Samples Split/Leaf: 
                            Minimum data points required for splitting or keeping a leaf.
                            
                            4. Max Features: 
                            Number of features used to split nodes
            """)
    if selection2=='Feature Scaling is required?':
        st.write("NO")
#########################################################################
    options2 = ["SVM",'Advantage and DisAdvantage','Feature Scaling is required?','Linear SVM','non_linear SVM','hyperparameters']
    selection2 = st.segmented_control("5.Support Vector Machines (SVMs)", options2, selection_mode="single")
    if selection2=='SVM':
        st.markdown("""
        SVM is a supervised machine learning algorithm used for classification and regression. 
        It finds the best boundary to separate different classes. 
        It keeps the maximum distance between the classes, which helps improve accuracy and reduce mistakes on new data.

        Linear SVM
         - Best for linearly separable data.
        Non-Linear SVM
         - Uses the kernel trick to separate data that is not linearly separable.
         
        Commonly used kernels are:
                - linear:- Used when data is linearly separable.
                - polynomial :-Useful for capturing curved boundaries.
                - rbf. :- Most popular kernel. Works well for complex data.
                - sigmoid :- Similar to neural networks activation.
        """)
    if selection2=='Advantage and DisAdvantage':
        st.markdown("""
                    Advantages of SVMs
                    - Effective for high-dimensional datasets.
                    - Works well with clear class separation.
                    - Memory efficient.
                    - Robust and accurate for many classification problems.
                    - Supports different kernel functions.
                    """)
        st.markdown("""
                    Disadvantages of SVMs
                    - More Training Time is required for larger dataset.

                    - SVMs work better only with Preprocessed data.

                    - difficult to choose a good kernel function.
                    - It is not that easy to fine-tune these hyper-parameters.
        """)
    if selection2=='Linear SVM':
        st.markdown("""
                    A Linear Support Vector Machine (SVM) is a supervised learning algorithm primarily used for classification tasks. 
                    It finds the optimal hyperplane that separates data points into distinct classes by maximizing the margin between them.

                    How It Works
                    - 1 Hyperplane: In a 2D space, it's a straight line that separates the classes; in higher dimensions,
                    it's a hyperplane.
                    - 2 Support Vectors: Data points closest to the hyperplane that determine its position.
                    - 3 Margin Maximization: SVM seeks to maximize the distance between the support vectors and the hyperplane.

                    from sklearn.model_selection import train_test_split
                    from sklearn.svm import SVC
                    from sklearn.metrics import accuracy_score
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    # Train Linear SVM
                    clf = SVC(kernel='linear')
                    clf.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = clf.predict(X_test)
                    
                    # Accuracy
                    print("Accuracy:", accuracy_score(y_test, y_pred))
        """)
    if selection2=='non_linear SVM':
        st.markdown("""
                    where the data is not linearly separable. 
                    It achieves this by transforming the original feature space into a higher-dimensional space using kernel functions

                    How Non-Linear SVM Works:
                    
                    Kernel Trick: 
                    Instead of mapping the data explicitly to higher dimensions, SVM uses kernel functions to compute the inner products of transformed features efficiently.

                    Common Kernel Functions:
                        
                        RBF (Radial Basis Function or Gaussian Kernel):
                        
                        Polynomial Kernel:
                        
                        Sigmoid Kernel:    
                    # Create and train a non-linear SVM using RBF kernel
                    
                    svm_model = SVC(kernel='rbf', C=1, gamma='scale')
                    svm_model.fit(X_train, y_train)

        """)
    if selection2=='Feature Scaling is required?':
        st.write("Yes")
    if selection2=='hyperparameters':
        st.markdown("""
        Machine learning models usually have default hyperparameters, but tuning them can significantly improve the model’s performance. 
        It's important because, without tuning, the model might not generalize well or might not be as accurate as it could be.

        Grid Search:

        This is an exhaustive method where you specify a grid of hyperparameter values and the model is trained and 
        evaluated for every combination.
        Example: Trying different values of C, kernel types, etc. for an SVM.

        Random Search:
        
        Instead of testing every possible combination, random search randomly samples hyperparameters from the grid.
        It’s faster and sometimes effective at finding the right set of parameters.

            from sklearn.model_selection import GridSearchCV
            from sklearn.svm import SVC
            
            # Define the parameter grid
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': [0.01, 0.1, 1]
            }
            
            # Create the SVM model
            svm = SVC()
            
            # Setup the grid search
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
            
            # Fit the model to the data
            grid_search.fit(X_train, y_train)
            
            # Print the best parameters
            print("Best parameters:", grid_search.best_params_)

        """)
        

################################################################################################
    options5 = ["Naive Bayes",'Advantage and DisAdvantage','Feature Scaling is required?']
    selection5 = st.segmented_control("6.Naive Bayes", options5, selection_mode="single")
    if selection5=='Naive Bayes':
        st.markdown("""
        Naive Bayes is a simple yet powerful classification algorithm based on Bayes' theorem. 
        It is widely used for tasks such as text classification, spam filtering, sentiment analysis, and more.

        P(A∣B) is the posterior probability of class A given predictor 𝐵.      
         
        P(B∣A) is the likelihood of predictor B given class 𝐴.          
        
        P(A) is the prior probability of class 𝐴. 
        
        P(B) is the prior probability of predictor 𝐵.
        
        Types of Naive Bayes Classifiers:
        
        Gaussian Naive Bayes: For continuous data (assumes Gaussian distribution).
        
        Multinomial Naive Bayes: Suitable for discrete data (commonly used for text data).
        
        Bernoulli Naive Bayes: For binary data (e.g., spam vs. non-spam).

        Real-World Example:
        
        Imagine classifying emails as spam or not. 
        Naive Bayes will calculate the probability of an email being spam based on the frequency of certain words in the email,
        assuming that the presence of each word is independent of the others.

        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import make_pipeline
        
        # Sample data
        X_train = ["free money", "earn dollars", "hello friend", "meeting tomorrow"]
        y_train = ["spam", "spam", "not spam", "not spam"]
        
        # Build a Naive Bayes classifier pipeline
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict
        test = ["free meeting"]
        print(model.predict(test))  # Output: ['not spam']
        """)
    if selection5=='Advantage and DisAdvantage':
        st.markdown("""
                    Advantages 
                    - Work Very well with many number of features
                    - Works Well with Large training Dataset
                    - It converges faster when we are training the model
                    - It also performs well with categorical features
        """)
        st.markdown("""
                    Disadvantages 
                    - Correlated features affects performance
        """)
    if selection5=='Feature Scaling is required?':
        st.write("NO")

########################################################################################

    options = ["XGBOOST",'Advantage and DisAdvantage','WHAT IS BOOSTING?','WHAT IS ENSEMBLE LEARNING?','EC2 INSTANCE','HYPERPARAMETERS']
        
    selection5 = st.segmented_control("XGBOOST", options, selection_mode="single")
    if selection5=='XGBOOST':
        st.markdown("""
        XGBoost or Extreme gradient boosting is the algorithm of choice for many data scientists and 
        could be used for regression and classification tasks. 
        
        XGBoost is a supervised learning algorithm and implements gradient boosted trees algorithm. 
        
        The algorithm work by combining an ensemble of predictions from several weak models.
        
        It is robust to many data distributions and relationships and offers many hyperparameters to tune model performance.
        
        Xgboost offers increased speed and enhanced memory utilization.

        """)
    if selection5=='Advantage and DisAdvantage':
        st.markdown("""
                    Advantages 
                    * No need to perform any feature scaling

                    * Can work well with missing data

                    * Robust to outliers in the data

                    * Can work well for both regression and classification

                    * Computationally efficient and produce fast predictions

                    * Works with distributed training: AWS can distribute the training process and data on many machines
        """)
        st.markdown("""
                    Disadvantages 
                    * Poor extrapolation characteristics

                    * Need extensive tuning

                    * Slow training
        """)
    if selection5=='WHAT IS BOOSTING?':
        st.markdown("""
        - Boosting works by learning from previous mistakes to come up with better future predictions. 

        - Boosting is an ensemble machine learning technique that works by training weak models in a sequentialy.
        
        - Each model is trying to learn from the previous weak model and become better at making predictions. 
        - Boosting algorithms work by building a model from the training data, then the second model is built based on the mistakes 
         of the first model. 
         The algorithm repeats until the maximum number of models have been created or until the model provides good predictions.
        """)
    if selection5=='WHAT IS ENSEMBLE LEARNING?':
        st.markdown("""
                    Ensemble techniques such as bagging and boosting can offer an extremely powerful algorithm 
                    by combining a group of relatively weak ones. 
            
                    For example, you can combine several decision trees to create a powerful random forest algorithm.

            """)
    if selection5=='EC2 INSTANCE':
        st.markdown("""
                    XGBoost currently only trains using CPUs.
                    
                    XGboost is memory intensive algorithm so it does not require much compute.
                    
                    M4: General-purpose compute instance is recommended. 
            """)
    if selection5=='HYPERPARAMETERS':
        st.markdown("""
        over 40 hyperparameters to tune Xgboost algorithm with AWS SageMaker 
        
        most important ones:
        
        Max_depth (0 – inf):
        is critical to ensure that you have the right balance between bias and variance. 
        If the max_depth is set too small, will underfit the training data.
        
        If you increase the max_depth, the model will become more complex and will overfit the training data. Default value is 6.
        
        Gamma (0 – inf):
        Minimum loss reduction needed to add more partitions to the tree.
        
        Eta (0 – 1):
        step size shrinkage used in update to prevents overfitting and make the boosting process more conservative. 
        After each boosting step, you can directly get the weights of new features, and eta shrinks the feature weights.
        
        Alpha: L1 regularization term on weights. regularization term to avoid overfitting. 
        The higher the gamma the higher the regularization. If gamma is set to zero, no regularization is put in place.
        
        Lambda: L2 regularization


        
        """)
        

#################################################################################################

with st.expander('8.Evaluation'):
    st.markdown("""
                **True Positive(TP):** Model correctly predicts a positive class.
                
                **True Negative(TN):** Model correctly predicts a negative class.
                
                **False Positive(FP):** Model predicts positive when it is actually negative.
                
                **False Negative(FN):** Model predicts negative when it is actually positive..
                """)
    options = ["Accuracy","confusion matrix","precision and recall","F1 score","R2-squared","MAE","MSE","Root Mean Square Error","AOC & AUC"]
        
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=="Accuracy":
        st.markdown(""""
        - how many total predictions are correct.
        - It works well for balanced datasets but is not reliable for imbalanced datasets."
        ** from sklearn metrics import accuracy score**

        **Accuracy =TP+TN / TP+TN+FP+FN**
        
**When to Use Accuracy?**
when the target variable classes in data are approximately balanced. 
For example, if 60% of classes in a fruit dataset are of Apple, 
40% are Mango.In this Case we can use Accuracy 
        
**When not to use Accuracy?**

Suppose there is a model for a disease prediction in which, 
out of 100 people, only five people have a disease, 
and 95 people don't have one. In this case we can use Accuracy .

                    """)
    if selection=="confusion matrix":
        st.markdown("""
        - A Confusion Matrix is a table used to evaluate the performance of a classification model by comparing actual values with predicted values.
        - It helps us understand where the model is making correct predictions and where it is making mistakes.
        Why is it Important?

        - Instead of looking only at Accuracy, the Confusion Matrix gives a deeper understanding of model performance.
                    """)
    if selection=="precision and recall":
        st.markdown("""
        **precision**
        
        Precision is a classification evaluation metric. 
        It tells us, out of all the positive predictions made by the model,
        how many were actually correct.
        
        **The formula is:** Precision = True Positives / (True Positives + False Positives).
        - We use precision when the cost of a false positive is high.

        **Recall**
       
        Recall is a classification evaluation metric.
        It tells us, out of all the actual positive cases, how many the model correctly identified.
        **The formula is:** Recall = True Positives / (True Positives + False Negatives).
        - We use recall when missing a positive case is very costly.
        """)
    if selection=="F1 score":
        st.markdown("""
        F1-Score is the balance between Precision and Recall. 
        We use it when both false positives and false negatives are important, especially for imbalanced datasets.

        The F1 score ranges from 0 to 1, where 1 indicates perfect precision and recall.

        Formula:
                2*precision *recall
           F1= --------------------
                precision +recall
        """)
    if selection=="R2-squared":
        st.markdown("""Measures how much variance in the data is explained by the model.

        - Closer to 1 = Better Fit
        - Higher R2 and Lower MAE/MSE = Better Model.""")
    if selection=="AOC & AUC":
        st.markdown("""The ROC curve is a graphical representation of a classifier's performance, 
        plotting the true positive rate against the false positive rate. 
        AUC (Area Under the Curve) measures the entire two-dimensional area underneath the ROC curve.
        """)
    if selection=="Root Mean Square Error":
        st.markdown("""
        - Square root of MSE.
        - Easy to interpret because it's in the same unit as the target variable.
        """)
    if selection=="MAE":
        st.markdown("""
        **Mean Absolute Error - MAE** ⟶ 
        - average of absolute differences between actual and predicted values.
        """)
    if selection=="MSE":
        st.markdown("""
        **Mean Absolute Error - MSE** ⟶ 
        - average of squared differences between actual and predicted values.
        """)
        
with st.expander('9.Visualization'):
    
    options = ["Bar Graphs","Line Chart","Pie Chart","Scatter Chart","Histogram","Heatmap"]
        
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=="Bar Graphs":
        st.markdown("""
    hen to Use?
    
                Comparing sales figures across different regions. 
                When comparing values across different categories, such as
                sales by region or survey responses by category
        """)
    if selection=="Line Chart":
        st.markdown("""
    hen to Use?
    
                When visualizing trend over time, 
                Such as tracking website & stock prices
    Examples:
    
                Displaying monthly revenue over the
                course of a year
        """)
    if selection=="Pie Chart":
        st.markdown("""
    hen to Use?
    
                When displaying proportions of a whole, 
                such as market share or budget allocation
    Examples:
    
                Showing the market share of different products.
        """)
    if selection=="Scatter Chart":
        st.markdown("""
    hen to Use?
    
                When analyzing relationships between two variables, such as
                advertising spend versus sales revenue.
    Examples:
    
                Exploring the relationship between
                advertising spend and sales revenue
        """)
    if selection=="Histogram":
        st.markdown("""
    when to Use?
    
                When showing the distribution of asingle variable, such as customer
                ages or transaction amounts.
    Examples:
    
                Displaying the distribution of customer ages.
        """)
    if selection=="Heatmap":
        st.markdown("""
    when to Use?
    
                When visualizing data density or intensity, such as website click
                activity or sales performance across regions.
    Examples:
    
                Showing data density, such as
                website click activity.
        """)

with st.expander('10.Python'):
    options = ["OOPS","overloading in Python","List","tuple()","List comprehension","lambda function","overloading in Python","overriding in Python","difference between a class method and an instance method","Python Basics","Control Flow","Functions","Data Structures","Modules and Packages","File Handling","Exception Handling","Regular Expressions","Python Libraries","Decorators and Generators"]
        
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=="overloading in Python":
        st.markdown("""
Method overloading in Python refers to defining
multiple methods with the same name but different
parameters within a class. However, Python does not
support method overloading by default as it does in
languages like Java. In Python, you can achieve a
similar effect by using default arguments or using
variable-length arguments.
        """)
    if selection=="OOPS":
        st.markdown("""
**40. What is object-oriented programming (OOP)?**
- Answer: OOP is a programming paradigm that uses objects (instances of classes) to
model real- world entities.

**41. What is a class in Python, and how do you define one?**
- Answer: A class is a blueprint for creating objects. You define one using the `class`
keyword.

**42. Explain the difference between a class and an object.**
- Answer: A class is a template for objects, while an object is an instance of a class.

**43. How do you create an instance (object) of a class in Python?**
- Answer: You create an object by calling the class as if it were a function, e.g.,
`my_obj = MyClass()`.

**44. Describe encapsulation in Python's OOP.**
- Answer: Encapsulation is the practice of bundling data (attributes) and methods
(functions) that operate on the data into a single unit (class).

**45. What is inheritance in OOP, and how does it work in Python?**
- Answer: Inheritance allows a class to inherit attributes and methods from
another class. It promotes code reuse and establishes a parent-child relationship.

**46. Explain polymorphism in OOP, and provide an example in Python.**
Answer: Polymorphism allows objects of different classes to be treated as objects of
the same class. For example, you can use different classes with a common interface
interchangeably

**47. What is method overloading in Python, and how is it achieved?**
- Answer: Python doesn't support method overloading like some other languages.
Instead, you can use default argument values to achieve similar functionality.

**48. Describe the concept of method overriding in OOP.**
- Answer: Method overriding allows a subclass to provide a specific implementation of
a method that is already defined in its parent class.

**49. What is a constructor in Python, and how is it defined?**
- Answer: A constructor is a special method that gets called when an object is created.
In Python, the constructor is defined as ` init ()`.
        """)
    if selection=="overriding in Python":
        st.markdown("""
Method overriding in Python refers to defining a
method in a child class that already exists in its
parent class with the same name and signature. The
method in the child class overrides the method in the
parent class, providing a different implementation.
        """)
    if selection=="difference between a class method and an instance method":
        st.markdown("""
class method is a method bound to the class and
not the instance of the class. It is defined using the
@classmethod decorator and can access only classlevel
variables. On the other hand, an instance
method is bound to the instance of the class and can
access both instance and class-level variables.
        """)
    if selection=="tuple()":
        st.markdown("""
A tuple in Python is an ordered and immutable
collection of elements. It is defined using parentheses
() or the tuple() constructor. For example:

        
        """)
    if selection=="List":
        st.markdown("""
List comprehension is a concise way to create lists in
Python based on existing lists or other iterables. It
combines the creation of a new list with a loop and
optional conditional statements. For example:
Ans:
numbers = [ , , , , ] 
squared_numbers = [x** x numbers] 
(squared_numbers)
1 2 3 4 5
2 for in
print # Output: [1, 4, 9, 16,
25]
        """)

    if selection=="List comprehension":
        st.markdown("""
List comprehension is a concise way to create lists in
Python based on existing lists or other iterables. It
combines the creation of a new list with a loop and
optional conditional statements. For example:
Ans:
numbers = [ , , , , ] 
squared_numbers = [x** x numbers] 
(squared_numbers)
1 2 3 4 5
2 for in
print # Output: [1, 4, 9, 16,
25]

numbers = [ ] 
even_numbers = [x x numbers if x % ==
] 
(even_numbers)
        """)
    if selection=="lambda function":
        st.markdown("""
A lambda function is an anonymous function defined
using the lambda keyword. It is typically used for
short, one-line functions. For example:
Ans:
square = x: x**
(square( ))
lambda 2 
print 3 # Output: 9
        """)
    if selection=="Python Basics":
        st.markdown("""
 **1. What is Python, and why is it popular?**
- Answer: Python is a high-level, interpreted programming language known for
its simplicity and readability. It is popular for web development, data analysis,
machine learning, and more.

**2. How do you comment in Python?**
- Answer: You can use `#` for single-line comments and `'''` or  for multi-line
comments.

**3. What is the difference between Python 2 and Python 3?**
- Answer: Python 3 isthe latest version and has backward-incompatible changes
from Python 2, with improved Unicode support and other enhancements.

**4. Explain Python's main data types.**
- Answer: Python's main data typesinclude int, float,str, bool, list, tuple,set, and
dict.

**5. How do you declare and assign a variable in Python?**
- Answer: You declare a variable by assigning a value to it, e.g., `x = 10`.

**6. What is a tuple, and how is it different from a list?**
- Answer: A tuple is an ordered, immutable sequence of elements, while a list is
mutable.

**7. Explain list comprehension in Python.**
- Answer: List comprehension is a concise way to create lists by applying an
expression to each item in an iterable.

**8. How do you swap the values of two variables without using a temporary**
variable?
- Answer: You can use tuple packing and unpacking: `a, b = b, a`.

**9. Describe Python's garbage collection.**
- Answer: Python uses automatic garbage collection to manage memory by
deallocating objects no longer in use.

**10. What is the `None` value in Python?**
- Answer: `None` represents the absence of a value or a null value in Python.       
        """)
    if selection=="Control Flow":
        st.markdown("""
**11. What is an if statement in Python?**
- Answer: An if statement is used for conditional execution. It runs a block of
code if a condition is True.

**12. What are "and" and "or" operators used for in Python?**
- Answer: `and` returns True if both operands are True, while `or` returns
True if at least one operand is True.

**13. Explain the purpose of the `elif` statement in Python.**
- Answer: `elif` is used to specify multiple conditions in an if-elif-else block,
allowing alternative execution paths.

**14. What is a "for" loop in Python?**
- Answer: A `for` loop iterates over a sequence (e.g., a list) and executes a block
of code for each item.

**15. How do you terminate a loop prematurely in Python?**
- Answer: You can use the `break` statement to exit a loop prematurely.

**16. Explain the `range()` function in Python.**
- Answer: `range()` generates a sequence of numbers, often used in for loops.

**17. What is a "while" loop, and how is it different from a "for" loop?**
- Answer: A `while` loop repeatedly executes a block of code while a condition
is True, whereas a `for` loop iterates over a sequence.

**18. How do you handle exceptions in Python?**
- Answer: You can use a `try` and `except` block to catch and handle exceptions.

**19. What is the purpose of the `finally` block in exception handling?**
- Answer: The `finally` block is used to execute code regardless of
whether an exception occurred or not.

**20. Explain Python's `assert` statement.**
- Answer: `assert` is used for debugging. It raises an error if a given condition is
False.
        """)
    if selection=="Functions":
        st.markdown("""
**21. How do you define a function in Python?**
- Answer: You define a function using the `def` keyword, e.g., `def
my_function():`.

**22. What is the difference between parameters and arguments in a function?**
- Answer: Parameters are variables in a function's definition, while arguments
are values passed when calling the function.

**23. Explain the concept of a "return" statement in Python functions.**
- Answer: A `return` statement is used to specify the value a function should
return when it's called.

**24. How can you define default parameter values in Python functions?**
- Answer: You can define default valuesin the function's parameter
list, e.g., `def greet(name="Guest"):`.

**25. What is a lambda function in Python?**
- Answer: A lambda function is a small, anonymous function defined
using the `lambda` keyword.

**26. Explain the concept of function closures in Python.**
- Answer: A closure is a function that remembers values in the enclosing scope
even if they are not present in memory.

**27. How do you pass a variable number of arguments to a function in Python?**
- Answer: You can use `*args` for positional arguments and `**kwargs` for
keyword arguments.

**28. Describe the purpose of the `map()` and `filter()` functions in Python.**
- Answer: `map()` applies a function to each item in an iterable, while
`filter()` filters items based on a condition.

**29. What is recursion in Python, and when is it useful?**
Answer: Recursion is a technique where a function calls itself. It's useful for
solving problems

**30. Describe variable scope in Python.**
- Answer: Variables defined in a function have local scope, while those
defined outside have global scope.

        """)
    if selection=="Data Structures":
        st.markdown("""
**31. What is a list in Python, and how do you create one?**
- Answer: A list is an ordered collection of items. You create one using square
brackets, e.g.,
`my_list = [1, 2, 3]`.

**32. Explain the difference between a shallow copy and a deep copy of a list.**
- Answer: A shallow copy copies the list itself, while a deep copy copies both
the list and its elements.

**33. What is a dictionary in Python, and how do you create one?**
- Answer: A dictionary is an unordered collection of key-value pairs. You create
one using curly braces, e.g., `my_dict = {"key": "value"}`.

**34. How do you access and modify dictionary elements in Python?**
- Answer: You can access elements using keys and modify them by assigning
new values to keys.

**35. Describe the purpose of a set in Python.**
- Answer: A set is an unordered collection of unique elements used for
various mathematical operations.

**36. Explain the difference between a set and a frozenset.**
- Answer: A set is mutable, while a frozenset is immutable and can be used as a
key in

**37. What is a tuple in Python, and how do you create one?**
- Answer: A tuple is an ordered, immutable collection of elements created
using parentheses, e.g., `my_tuple = (1, 2, 3)`.

**38. How do you add and remove items from a list in Python?**
- Answer: You can use the `append()` method to add items and the `remove()`
method to remove items from a list.

**39. What are list comprehensions, and how do you use them?**
- Answer: List comprehensions provide a concise way to create lists by applying
an expression to each item in an iterable.
        """)

    if selection=="Modules and Packages":
        st.markdown("""
**50. What is a module in Python?**
- Answer: A module is a file containing Python code. It can define
functions, classes, and variables.

**51. How do you import a module in Python?**
- Answer: You can import a module using the `import` statement, e.g., `import
math`.

**52. Explain the purpose of the `if name == " main ":` construct.**
- Answer: This construct is used to check if a Python script is being run as the
main program or if it's being imported as a module.

**53. What is a package in Python, and how is it different from a module?**
- Answer: A package is a collection of related modules organized in
directories. It provides a way to structure and manage larger Python
projects.

**54. How do you create and use your own packages in Python?**
- Answer: You create packages by organizing modules in directories with a
special ` init .py` file. You can then import and use them like any other module.

        """)
    if selection=="File Handling":
        st.markdown("""
**55. How do you open and close files in Python?**
- Answer: You can open a file using the `open()` function and close it using the `close()`
method or the `with` statement.

**56. Explain the purpose of the "r", "w", and "a" modes when opening files.**
- Answer: "r" is for reading, "w" is for writing (creates a new file or overwrites an
existing one), and "a" is for appending to an existing file.

**57. How do you read the contents of a file in Python?**
- Answer: You can use methods like `read()`, `readline()`, or `readlines()` to read the
contents

**58. What is the purpose of the "with" statement in file handling?**
- Answer: The `with` statement is used to ensure that a file is properly closed
after its suite finishes executing.
        """)

    if selection=="Exception Handling":
        st.markdown("""
**59. What is an exception in Python?**
- Answer: An exception is an event that occurs during the execution of a
program, disrupting the normal flow of instructions.

**60. How do you handle exceptions in Python?**
- Answer: You use a `try` and `except` block to catch and handle exceptions.

**61. What is the purpose of the `finally` block in exception handling?**
- Answer: The `finally` block is used to execute code regardless of
whether an exception occurred or not.

**62. How can you raise a custom exception in Python?**
- Answer: You can raise a custom exception using the `raise`
statement, e.g., `raise ValueError("Custom error message")`.

**63. Describe common built-in exceptions in Python.**
- Answer: Common exceptions include `ZeroDivisionError`, `ValueError`,
`TypeError`,
`FileNotFoundError`, and `KeyError`.
        """)

    if selection=="Regular Expressions":
        st.markdown("""
**64. What are regular expressions (regex) in Python?**
- Answer: Regular expressions are patterns used forsearching and matching text.

**65. How do you use the `re` module for regular expressions in Python?**
- Answer: You import the `re` module and use its functions like `search()`,
`match()`, and
`findall()` to work with regular expressions.

**66. Explain the purpose of character classes in regular expressions.**
- Answer: Character classes allow you to match sets of characters, e.g., `[0-9]`
matches any digit.

**67. What is the difference between `match()` and `search()` functions in the `re`
module?**
Answer: `match()` matches the pattern only at the beginning of the string, while
`search()` searches for the pattern anywhere in the string.
        """)

    if selection=="Decorators and Generators":
        st.markdown("""
**83. What is a decorator in Python?**
- Answer: A decorator is a function that wraps another function to extend or
modify its behavior.

**84. How do you define and use decorators in Python?**
- Answer: You define decorators by using the `@`symbol above a function and
apply them to other functions.

**85. What are generator functions in Python?**
- Answer: Generatorfunctions are functionsthat use the `yield` keyword to yield
values one at a time, allowing for efficient memory usage.

**86. How do you create a generator object and iterate over its values?**
- Answer: You create a generator object by calling a generator function. You can
iterate over its values using a `for` loop or by calling `next()`.

**87. Explain the difference between a generator and a regular function.**
- Answer: A regular function runs to completion and returns a single value,
while a generator function can yield multiple values and retains its state
between calls.
        """)

    if selection=="Python Libraries":
        st.markdown("""
**88. What is NumPy, and why is it important in data science?**
- Answer: NumPy is a Python library for numerical computing. It
provides arrays and mathematical functions crucial for data science
and scientific computing.

**89. Describe the purpose of the Pandas library in Python.**
- Answer: Pandasis a library for data manipulation and analysis. It provides data
structures like DataFrames and Series for handling tabular data.

**90. How do you install and use Matplotlib for data visualization in Python?**
- Answer: You install Matplotlib using `pip` and use it to create various types
of plots, charts, and graphs.

**91. What is the purpose of the scikit-learn library in Python?**
- Answer: Scikit-learn is a machine learning library used for tasks like
classification, regression, clustering, and model evaluation.

**92. Explain the role of TensorFlow and PyTorch in Python's machine learning
ecosystem.**
- Answer: TensorFlow and PyTorch are deep learning frameworks used for
developing and training neural networks.

        """)


with st.expander('11.code'):
    options = ["happy numbers","even num",'area of a triangle','leap',"prime no","factorial"]
        
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=="happy numbers":
        code = '''
        def is_happy_number(num):
            seen = set()
            while num != 1 and num not in seen:
                seen.add(num)
                num = sum(int(i) ** 2 for i in str(num))
            return num == 1
       happy_numbers = []
       for num in range(1, 101):
          if is_happy_number(num):

             happy_numbers.append(num)

        print("Happy Numbers between 1 and 100:")

        print(happy_numbers)
        '''
        st.code(code, language='python')
    if selection=="even num":
        code = '''
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Using a list comprehension to filter even numbers
        even_numbers = [num for num in numbers if num % 2 != 0]
        # Print the even numbers
        print("Odd numbers in the list:", even_numbers)

  
        '''
        st.code(code, language='python')
    if selection=="area of a triangle":
        code='''
        base = float(input("Enter the length of the base of the triangle: "))

        height = float(input("Enter the height of the triangle: "))

        # Calculate the area of the triangle

        area = 0.5 * base * height

        # Display the result

        print(f"The area of the triangle is: {area}")
        
        '''
        st.code(code, language='python')
     
    if selection=="leap":
        code='''
        year = int(input("Enter a year: "))
        if (year % 400 == 0) and (year % 100 == 0):

         print("{0} is a leap year".format(year))

         elif (year % 4 ==0) and (year % 100 != 0):

          print("{0} is a leap year".format(year)

           else:

           print("{0} is not a leap year".format(year))
        '''
        st.code(code, language='python')
    if selection=="prime no":
        code = '''
        num = int(input("Enter a number: "))

        # define a flag variable
        flag = False
        if num == 1:
             print(f"{num}, is not a prime number")
       elif num > 1:

         # check for factors
             for i in range(2, num):

                 if (num % i) == 0:

                    flag = True # if factor is found, set flag to True
         # break out of loop
                    break
         # check if flag is True
        if flag:
            print(f"{num}, is not a prime number")
         else:
           print(f"{num}, is a prime number")
        
        '''
        st.code(code, language='python')
    if selection=="factorial":
        code = '''
        num = int(input("Enter a number: "))
        factorial = 1
        if num <0:
           print("Factirial does not exist for negative numbers")
        elif num == 0:
             print("Factorial of 0 is 1")
        else:
            for i in range(1, num+1):
                factorial = factorial*i
            print(f'The factorial of {num} is {factorial}')
        '''
        st.code(code, language='python')
    if selection==" ":
        code = '''
        
        '''
        st.code(code, language='python')
    



