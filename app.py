import streamlit as st

with st.expander('1.Self introduction'):
    st.write('hai')
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
with st.expander('3.Defination Python,ML,DL,DL,NL'):

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
        Machine Learning is essential because it enables systems to automatically improve their performance and adapt to changing conditions without human intervention. It is used in various fields to solve complex problems, such as:

1. Automation: 

Reducing manual tasks by automating decision-making processes.

2. Predictive Analytics: 

Making accurate predictions based on historical data, such as stock market trends or customer behavior.

3. Personalization: 

Customizing user experiences in areas like marketing, content recommendations, and product suggestions.

4. Revolutionizing Industries: 

Healthcare, Finance, Retail
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
#####################################################################################
with st.expander('4.AWS'):
        options3 = ["Sagemaker","What is AWS","What is cloud computing ","WHAT IS A REGION?",]
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
        if selectio=="Sagemaker":
            st.markdown("""
            AWS SageMaker is a fully managed service that helps data scientists and developers 
            build, train, and deploy machine learning models quickly and efficiently. 
            It removes the heavy lifting from machine learning tasks 
            by providing easy access to scalable computing resources, integrated tools, and pre-built algorithms.
    
        Key features include:
        
        1. Model Building: 
        
        SageMaker Studio offers a fully integrated development environment (IDE) 
        for creating machine learning models with built-in data preparation tools and 
        notebook environments.
        
        
        2. Model Training: 
        
        SageMaker provides automatic model training and tuning using distributed infrastructure, 
        which significantly reduces the time required for training large datasets.
        
        
        3. Model Deployment:
        
        Once trained, models can be deployed at scale using SageMaker endpoints for real-time or batch predictions.
        
        
    
        Additionally, SageMaker includes features like model monitoring, explainability tools, and 
        support for custom machine learning frameworks such as TensorFlow, PyTorch, and Scikit-learn.
    
    It streamlines the end-to-end machine learning workflow, enabling faster and more efficient production deployments.
    
            
            """)
#####################################################################################        
with st.expander('5.Data Preprocessing'):
    options = ["Bias & Varience", "Balancen & Imbalance", "FEATURE SCALING NORMALI-STANDARDIZATION",'Underfitting & Overfitting','Missing Data','Cross-validation']
    
    selection = st.segmented_control("", options, selection_mode="single")
    if selection=='Bias & Varience':
        st.markdown("""BIAS

In a machine learning model, bias means the model consistently predicts values or outcomes that are differentnd Grow from the true values in the data. 
A model with high bias might be too simple or have wrong assumptions,
 causing it to underperform and make inaccurate predictions.
        """)
        st.markdown("""VARIANCE

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
        st.markdown("""Imbalanced datasets can be handled using techniques like 

                - oversampling, 
                - undersampling, or using algorithms
                - designed for imbalanced data such as SMOTE 
                (Synthetic Minority Over-sampling Technique).
         """)
        st.markdown("""OVER SAMPLING

Oversampling is used when the quantity of data is insufficient. 
It tries to balance dataset by increasing the size of rare samples
         """)
        st.markdown("""UNDER SAMPLING

This method is used when quantity of data is sufficient. 
By keeping all samples in the rare class and randomly selecting an equal number of samples in the abundant class, 
a balanced new dataset can be retrieved for further modelling.
         """)
    if selection=='FEATURE SCALING NORMALI-STANDARDIZATION':
        st.markdown("""FEATURE SCALING

Feature scaling is a preprocessing technique in machine learning used to standardize or normalize the range of independent variables or features of data. 
The goal of feature scaling is to ensure that all features have similar scales or magnitudes.
         """)
        st.markdown("""MIN-MAX SCALING (NORMALIZATION)


This method scales the features to a specific range, typically between 0 and 1. It's like changing all your ingredients to be on a scale from 0 to 1, where O means the smallest amount, and 1 means the largest amount. 
Just like making sure all your ingredients are in a similar range
         """)
        st.markdown("""Z-score SCALING (STANDARDIZATION)

Standardization transforms features to have a mean of 0 and a standard deviation of 1. This one makes all your ingredients have an average (mean) of 0 and a standard deviation of 1. 
It's like making sure all your ingredients are centered around a common point and have similar Grow spreads.
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
with st.expander('6.ML Algorithms'):    
    options = ["Regression", "Linear regression",'Advantage and DisAdvantage','Feature Scaling is required?']
    
    selection = st.segmented_control("1.Linear Regression", options, selection_mode="single")
    if selection=='Regression':
        st.markdown("""
        Regression is a method to find the relationship between two or more variables.
        It helps us make predictions by drawing a straight line through data points. 
        It's like having a ruler to estimate values based on known information.
        """)
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
    options1 = ["Decision Tree", "Real_Word eg","Advantage and DisAdvantage", "How Choose Root Node", "Entropy Formula"]
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
####################################################################
    options2 = ["Random Forest",'Advantage and DisAdvantage','Feature Scaling is required?']
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
        if selection2=='Feature Scaling is required?':
            st.write("NO")
#########################################################################
    options2 = ["SVM",'Advantage and DisAdvantage','Feature Scaling is required?','Linear SVM','non_linear SVM']
    selection2 = st.segmented_control("5.Support Vector Machines (SVMs)", options2, selection_mode="single")
    if selection2=='SVM':
        st.markdown("""Support Vector Machines (SVMs) separates data points based on decision planes, which separates objects belonging to different classes in a higher dimensional space.

SVM algorithm uses the best suitable kernel, which is capable of separating data points into two or more classes.

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
        

################################################################################################
    options5 = ["Naive Bayes",'Advantage and DisAdvantage','Feature Scaling is required?']
    selection5 = st.segmented_control("6.Naive Bayes", options5, selection_mode="single")
    if selection5=='Naive Bayes':
        st.markdown("""Naive Bayes is a simple yet powerful classification algorithm based on Bayes' theorem. 
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
        st.markdown("""A confusion matrix is a table used to evaluate the performance of a classification model. 
                    It shows the counts of true positives, true negatives, false positives, and false negatives.
        """)
    if selection=="precision and recall":
        st.markdown("""Precision is a measure of how many of the positive predictions made by a classification model were actually correct.
        
        **ratio of true positives to the sum of true Positive and false positives**

        Recall is a measure of how many of the actual positive instances in the dataset were correctly predicted by the model.
        
        **ratio of true positives to the sum of true positives and false negatives.**
        """)
    if selection=="F1 score":
        st.markdown("""The Fl score is the harmonic mean of precision and recall and is used to evaluate the balance between precision and recall in a classification model.
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
