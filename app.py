import streamlit as st

with st.expander('1.Self introduction'):
    options = ["introduction","Roles & Responsibilities","Why ML selected"]     
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=="introduction":
        st.markdown("""
        Hi, I am Murugan from Tamil nadu . I have completed my Bachelor Degree in Commerce from [Mumbai University in 2016.and
        I hold a AWS ML certified. 
        
        I have been working as an IT analyst at TCS for the past two and half years. Overalls 6+ years of the Experience in the IT field .
        
        My job deals with Parle Application Rollout .
        This involved Desktop and Mobile application services which is targeted to the Distributors in the Client market.
        As Desktop application holds the Database sequences and make the Product 
        distribution hierarchy smoother with the resellers data. Also having the DMS consoles, 
        Database installation procedure to help out client market process. Mobile application used to collect market data and 
        to proceed it to DMS application.
        
        I have good knowledge in the ML Algorithms ,Python Pandas,Data pre-processing and MySQL . 
        
        My key strengths are leadership skills, time management and I tend to be team oriented and trustworthy.
        
        Reading books and playing cricket are my favorite hobbies. 
        I have excellent communication skills and I am fluent in Tamil,English,Hindi and MArathi.
        
        I'm looking for an opportunity to expand my career in Machine learning field. 
        which is why I'm selected  Wings2 Machine learning Job path .
        
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
    options = ["emails as spam or not spam"]     
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=='emails as spam or not spam':
        st.markdown("""
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
#####################################################################################
with st.expander('3.Defination Python,ML,DL,DL,NL','PYTHON','Numpy'):

    options = ["DS", "ML", "WHY ML","Supervised & Unsupervised learning"]
        
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
        st.markdown("""*Supervised learning algorithms are trained using labeled data.
*Supervised learning model predicts the output.
*Supervised learning can be categorized in Classification and Regression problems.
*It includes various algorithms such as 
Linear Regression, Logistic Regression, Support Vector Machine, Multi-class Classification, Decision tree, 
Bayesian Logic, etc.

*Unsupervised learning algorithms are trained using unlabeled data.
*Unsupervised learning model finds the hidden patterns in data.
*Unsupervised Learning can be classified in Clustering and Associations problems.
*it includes various algorithms such as Clustering, KNN, and Apriori algorithm.

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
#####################################################################################
with st.expander('4.AWS'):
        options3 = ["Sagemaker","Sagemaker_COMPONENTS","What is AWS","What is cloud computing ","WHAT IS A REGION?","AVAILABILITY ZONE","AWS S3","AWS IAM","AWS Lambda","AWS Athena","AWS kinesis"]
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

    
        Key features include:
        
        1. Model Building: 
        
        * SageMaker offers data labeling service

        * Prebuilt available notebooks with state of the art algorithms on AWS marketplace
        
        2. Model Training: 
        
        * Train models using EC2 instances (on-demand and spot)

        * Manage environments for training

        * Hyperparameters optimization for model tuning

    
        3. Model Deployment:
        
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
            
                Amazon S3 bucket URL (training data): where the training data is located.
                Compute resources: Amazon SageMaker will train the model using instances managed by Amazon SageMaker.
                Amazon S3 bucket URL (Output): this bucket will host the output from the training.
                Amazon Elastic Container Registry path: where the training code is stored. 
                
            SageMaker launches an ML compute instances once a training job is initiated. 
            SageMaker uses: (1) training code and (2) training dataset to train the model. 
            SageMaker saves the trained model artifacts in an S3 bucket.
            """)
        if selectio=="AWS S3":
            st.markdown("""
            AWS S3 (Simple storage services)
* AWS s3 is a most popular storage services most of the organisation they uses. as storage of S3
Use Cases

1 Data Backup and archiving: 

s3 is commonly used for data backup and long-term archiving due to its durability and reliability.
organisations can store backup copies of their data to protect against data loss

2 Data storage for Application :-

S3 serves as a data repository for application, enabling them to store and.
retrieve files, assets, and other resovaces.
  This can include content for websites, mobile apps, and software.

3) Log and Event storage. :-

Application and systems can use S3 to Store logs, events, and telemetry data.
This data can later be analyzed for monitoring, purposes. , troubleshooting, and compliance purpose.

4) media storage and streaming:

S3 is used to store media files Such as video and audio that need to be accessed and streamed by users globally.

        Question and ANS 
      
1)S3 is global service but why do we require region selection?

Ans

it is the global service, for fastes performance and low letergy.

2) S3 can we upload - 10TB data at one time if it allows unlimited data upload.

Ans

No, we can upload 5 TB at one time

3) How many Bucket we can create in aws account ?

Ans
1oo, bucket we can create in l aws account , But that we can extend with raise the ticket with from AWS.
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
with st.expander('5.Data Preprocessing'):
    options = ["Bias & Varience", "Balancen & Imbalance", "FEATURE SCALING NORMALI-STANDARDIZATION",'Underfitting & Overfitting','Missing Data','Cross-validation','REGULARIZATION L1 L2','outliers','Bagging & boosting','Ensemble learning','Hyperparameter tuning']
    
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=='Bias & Varience':
        st.markdown("""
        The bias-variance trade-off is the balance between two sources of error in machine learning models: bias and variance.

        As bias decreases, variance increases, and vice versa.

        The goal is to find a balance where both bias and variance are minimized for the best generalization.

        --Too Simple (High Bias): 
        Poor training and test performance.

        --Too Complex (High Variance):
        Good training but poor test performance.
        """)
        st.markdown("""
        
        BIAS

In a machine learning model, bias means the model consistently predicts values or outcomes that are differentnd Grow from the true values in the data. 
A model with high bias might be too simple or have wrong assumptions,
 causing it to underperform and make inaccurate predictions.
        """)
        st.markdown("""
        VARIANCE

In a machine learning model, high variance means that it's sensitive to the specific data it's trained on. 
If you give it slightly different datasets, it might give wildly different predictions. 
A model with high variance is often too complex and has learned the training data's noise rather than the true patterns.
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

        st.markdown("""Cross-validation is a technique used in machine learning to evaluate how well a model performs on unseen data.
                It helps ensure that the model isn't just memorizing the training data but can generalize to new data.
                
???????How It Works:???????

1.Split the Data: 
The entire dataset is divided into several parts or "folds." A common choice is 5 or 10 folds.
                
2.Training and Testing:              
In each iteration, one fold is used for testing, and the remaining folds are used for training.
This process repeats until each fold has been used as a test set.
                
3.Evaluation: 
The performance scores from each fold are averaged to get a more reliable measure of how well the model will perform.
                
?????? Example: ??????

If you use 5-fold cross-validation:
                
The data is split into 5 parts.
The model trains on 4 parts and tests on the 1 remaining part.
This repeats 5 times, rotating the test set each time.

??????Why Use It???????

1.Better Model Evaluation: 
It gives a more accurate estimate of the model's performance compared to a simple train-test split.
                
2.Reduces Overfitting: 
By testing the model on different portions of data, it ensures the model isn't too tuned to just one dataset.
                
?????Types of Cross-Validation:????
k-Fold Cross-Validation: Most common type.
Stratified k-Fold: Ensures each fold has a similar distribution of target classes.
Leave-One-Out (LOOCV): Each data point becomes a test set once.
                         """)
    if selection=='REGULARIZATION L1 L2':
        st.markdown("""
        Regularization is a technique used to prevent overfitting by adding a penalty to the model's complexity during training.

        Regularization helps the model generalize better to new data.

        their is two type Regularization is L1 and L2.

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
    
        Ensemble learning involves combining multiple individual models 
        to improve the overall performance and predictive power of the learning algorithm.
        """)
    if selection=='Bagging & boosting':
        st.markdown("""
        What is Bagging ?
        。Stands for Bootstrap Aggregating.

        。Creates multiple models by training them on different random 
        subsets of the data (with replacement).

        。Each model makes a prediction, and their results are averaged (for regression) 
        or voted on (for classification).

        **Example:** 
         imagine you dividing your classroom into group and giving each group a problem. 
         and whatever answer each group comes up with . you average their answers
         this method is going to reduce the variance and it is also going to make final model most stable .
         Random forest is a classic example of bagging .

                  
        """)
        st.markdown("""
        What is Bootinng 
        
        。Combines models sequentially, where each model corrects 
        the errors of the previous one.

        。Focuses more on hard-to-predict data points.

        。 Models are trained one after another, with more weight given 
        to data points that were previously misclassified.

        where a teacher is helping the students. who struggle the most in the last test. 
        now in the boosting is going to be a very great way of correcting the bias and weak models .right
        so whichever the student  struggled in the last test . 
        are going to be train separately. and help improve in the next exam .
        Gradient Boosting, AdaBoost. is the example of boosting .
        """)
    if selection=='Hyperparameter tuning':
        st.markdown("""
        Hyperparameter tuning involves finding the best set of hyperparameters 
        for a machine learning model to optimise its performance and generalisation.
        
        """)
with st.expander('6.ML Algorithms'):    
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
        A decision tree Algorithm is a supervised Learning Algorithms,
        is a flowchart-like model that makes decisions by asking questions based on data features,
        leading to clear outcomes at the end. 
        It's a simple and intuitive tool used in machine learning for classification and regression tasks.
        """)
 
    if selection1=='Real_Word eg':
        st.markdown("""
        In the Decion Tree we can classify the data in different like the FLOWERS with DIFFERENT features
        like sample length red length to classify the Data to different groups
        """)
    if selection1=='Advantage and DisAdvantage':
        st.markdown("""
                    Advantages
                    - Simple and easy to understand: Decision Tree looks like simple if-else statements which are very easy to understand
                    - Decision Tree can be used for both classification and regression problems.
                    - Decision Tree can handle both continuous and categorical variables.
                    - No feature scaling required: like(standardization and normalization) required in case of Decision Tree as it uses rule based approach instead of distance calculation.
                    - Decision Tree can automatically handle missing values.
                    - Decision Tree is usually robust to outliers and can handle them automatically.
                    - Training period is less as compared to Random Forest because it generates only one tree unlike forest of trees in the Random Forest.
        """)
        st.markdown("""
                    Disadvantages of Decision Trees
                    - Decision trees sometimes become complex, which do not generalize well and leads to overfitting.
                    Overfitting can be addressed by placing the least number of samples needed at a leaf node or placing the highest depth of the tree.

                    -If data size is large, then one single tree may grow complex and lead to overfitting. 
                    So in this case, we should use Random Forest instead of a single Decision Tree.
                    
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
                    A Random Forest is a powerful and versatile machine learning algorithm 
                    mainly used for classification and regression tasks. 
                    It's based on the concept of "ensemble learning,
                    " where multiple models are combined to produce better results than individual models.
            
                    Real-World Example:
                    Imagine a bank wants to predict if a customer will default on a loan. 
                    One decision tree might focus heavily on income, another on credit history, 
                    and a third on spending patterns. 
                    The Random Forest combines all these perspectives to give a more accurate and reliable prediction.
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
        It separates data points into categories by drawing the "best possible line" (or boundary).

        Works well for both simple and complex data.

        The algorithm chooses the line that leaves the largest gap (margin) between data points of different categories.

        The data points closest to the boundary are called support vectors, and they influence where the line is drawn.

        If data can't be separated with a straight line, SVM uses mathematical functions (kernels) 
        to transform it into a higher dimension where a boundary can be drawn

Commonly used kernels are:
                - linear
                - polynomial
                - rbf
                - sigmoid
        """)
    if selection2=='Advantage and DisAdvantage':
        st.markdown("""
                    Advantages of SVMs
                    - SVM can distinguish the classes in a higher dimensional space.

                    - SVM algorithms are memory efficient.

                    - SVMs are versatile, and a different kernel can be used by a decision function.

                    - Works well with even unstructured and semi structured data like text, Images and trees.
                    - Risk of over-fitting is less in SVM.
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
        st.markdown("""Naive Bayes is a simple yet powerful classification algorithm based on Bayes' theorem. 
        It is widely used for tasks such as text classification, spam filtering, sentiment analysis, and more.

        **P(A∣B)** is the posterior probability of class A given predictor 𝐵.      
         
        **P(B∣A)** is the likelihood of predictor B given class 𝐴.          
        
        **P(A)** is the prior probability of class 𝐴. 
        
        **P(B)** is the prior probability of predictor 𝐵.
        
        **Types of Naive Bayes Classifiers:**
        
        **Gaussian Naive Bayes:** For continuous data (assumes Gaussian distribution).
        
        **Multinomial Naive Bayes:** Suitable for discrete data (commonly used for text data).
        
        **Bernoulli Naive Bayes:** For binary data (e.g., spam vs. non-spam).

        **Real-World Example:**
        
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

with st.expander('7.Evaluation'):
    st.markdown("""
                **True Positive(TP):** In this case, the prediction outcome is true, and it is true in reality, also.
                
                **True Negative(TN):** in this case, the prediction outcome is false, and it is false in reality, also.
                
                **False Positive(FP):** In this case, prediction outcomes are true, but they are false in actuality.
                
                **False Negative(FN):** In this case, predictions are false, and they are true in actuality.
                """)
    options = ["Accuracy","confusion matrix","precision and recall","F1 score","R-squared","AOC & AUC","Root Mean Square Error"]
        
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=="Accuracy":
        st.markdown("""The accuracy metric is one of the simplest Classification metrics to implement, 
                    and it can be determined as the number of correct predictions to the total number of predictions.

                    from sklearn metrics import accuracy score
        
                    **When to Use Accuracy?**
                    when the target variable classes in data are approximately balanced. 
                    For example, if 60% of classes in a fruit dataset are of Apple, 
                    40% are Mango.In this Case we can use Accuracy 
        
                    **When not to use Accuracy?**
                    when the target variable majorly belongs to one class. 
                    For example, 
                    Suppose there is a model for a disease prediction in which, 
                    out of 100 people, only five people have a disease, 
                    and 95 people don't have one. In this case we can use Accuracy .

                    """)
    if selection=="confusion matrix":
        st.markdown("""
                    A confusion matrix is a table used to evaluate the performance of a classification model. 
                    It shows the counts of true positives, true negatives, false positives, and false negatives.

                    It shows correct and incorrect predictions in an organized way.

                    Useful for understanding the types of mistakes the model makes
        """)
    if selection=="precision and recall":
        st.markdown("""
        Precision is a measure of how many of the positive predictions made by a classification model were actually correct.
        
        ratio of true positives to the sum of true Positive and false positives

        Recall is a measure of how many of the actual positive instances in the dataset were correctly predicted by the model.
        
        ratio of true positives to the sum of true positives and false negatives.
        """)
    if selection=="F1 score":
        st.markdown("""
        The F1 score is a metric that combines precision and recall into a single number.

        It provides a balance between precision and recall

        The F1 score ranges from 0 to 1, where 1 indicates perfect precision and recall.

        Formula:
                2*precision *recall
           F1= --------------------
                precision +recall
        """)
    if selection=="R-squared":
        st.markdown("""R-squared is a statistical measure that represents the proportion of the variance for a dependent variable that is explained by an independent variable in a regression model.""")
    if selection=="AOC & AUC":
        st.markdown("""The ROC curve is a graphical representation of a classifier's performance, 
        plotting the true positive rate against the false positive rate. 
        AUC (Area Under the Curve) measures the entire two-dimensional area underneath the ROC curve.
        """)
    if selection=="Root Mean Square Error":
        st.markdown("""The Root Mean Square Error is a commonly used metric for evaluating the accuracy of a regression model by measuring the differences between the predicted values and the actual values.
        """)
with st.expander('8.Visualization'):
    
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
                
