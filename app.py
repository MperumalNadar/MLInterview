import streamlit as st

with st.expander('1.Self introduction'):
    st.write('hai')
with st.expander('2.Project Explanation'):
    st.write('hai')
#####################################################################################
with st.expander('3.Defination Python,ML,DL,DL,NL'):

    options = ["DS", "ML", "WHY ML",'Supervised & Unsupervised learning', "Python",'AWS']
        
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

1. Automation: Reducing manual tasks by automating decision-making processes.

2. Predictive Analytics: Making accurate predictions based on historical data, such as stock market trends or customer behavior.

3. Personalization: Customizing user experiences in areas like marketing, content recommendations, and product suggestions.

4. Revolutionizing Industries: Healthcare, Finance, Retail
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


with st.expander('4.AWS'):
    st.write('hai')
with st.expander('5.Data Preprocessing'):
    options = ["Bias & Varience", "Balancen & Imbalance", "FEATURE SCALING NORMALI-STANDARDIZATION",'Underfitting & Overfitting','Missing Data']
    
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
with st.expander('6.ML Algorithms'):    
    options = ["Regression", "Linear regression", "Logistic regression"]
    # st.header('Linear Regression and Logistic Regression')
    selection = st.segmented_control("1.Linear Regression and Logistic Regression", options, selection_mode="single")
    if selection=='Regression':
        st.markdown("""Regression is a method to find the relationship between two or more variables.
         It helps us make predictions by drawing a straight line through data points. 
        It's like having a ruler to estimate values based on known information.
        """)
    if selection=='Linear regression':
        st.markdown("""Linear regression is a method to find the best straight line that fits data points. 
        It helps us understand how one variable changes with another. 
        This line allows us to make predictions and see the overall trend in the data.
        """)
        st.write('#########')
        st.markdown("""With the line equation (Y = 10x + 50), we can make predictions. 
        For example, if a student studies for 6 hours (X = 6), 
        we can estimate their exam score by plugging the value into the equation: Y = 10 * 6 + 50 = 110.
        """)
    if selection=='Logistic regression':
        st.markdown("""Logistic regression is a statistical technique used for binary classification tasks, 
        where the goal is to predict one of two outcomes (e.g., yes or no, spam or not spam). 
        It models the relationship between independent variables (features) and the probability of a binary outcome, using the sigmoid function to map the predictions to probabilities between 0 and 1. The result is a decision boundary that separates the two classes and allows us to make probabilistic predictions.
        """)
##############################################################################
    options1 = ["Decision Tree", "Real_Word eg","Advantage and DisAdvantage", "How Choose Root Node", "Entropy Formula"]
    selection1 = st.segmented_control("2.Decision Trea", options1, selection_mode="single")
    if selection1=='Decision Tree':
        st.markdown("""A decision tree Algorithm is a supervised Learning Algorithms,
        is a flowchart-like model that makes decisions by asking questions based on data features,
        leading to clear outcomes at the end. 
        It's a simple and intuitive tool used in machine learning for classification and regression tasks.
        """)
 
    if selection1=='Real_Word eg':
        st.markdown(""" In the Decion Tree we can classify the data in different like the FLOWERS with DIFFERENT features
        like sample length red length to classify the Data to different groups
        """)
    if selection1=='Advantage and DisAdvantage':
        st.markdown("""
                    Advantages
                    - Decision Trees are easy to understand.
                    - They often do not require any preprocessing.
                    - Decision Trees can learn from both numerical and categorical data.
        """)
        st.markdown("""
                    Disadvantages of Decision Trees
                    - Decision trees sometimes become complex, which do not generalize well and leads to overfitting. Overfitting can be addressed by placing the least number of samples needed at a leaf node or placing the highest depth of the tree.

                    -A small variation in data can result in a completely different tree. This problem can be addressed by using decision trees within an ensemble.
        """)

    if selection1=='How Choose Root Node':
        st.markdown(""" Find the Decision Tree Root node we find the best fit for the solution to get the maximum information GAIN
        We use various features like the Guinea or Enropy
        """)
    if selection1=='Entropy Formula':
        st.markdown(""" Entropy is formula is Negation of Summation of the Probability of Instance into Log of Probability
        """)
####################################################################
    options2 = ["Random Forest"]
    selection2 = st.segmented_control("3.Random Forest", options2, selection_mode="single")
    if selection2=='Random Forest':
        st.markdown("""Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs 
        the mode of the classes or the mean prediction of the individual trees for classification and regression tasks, respectively.
        """)
#########################################################################
    options2 = ["SVM",'Advantage and DisAdvantage']
    selection2 = st.segmented_control("4.Support Vector Machines (SVMs)", options2, selection_mode="single")
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
        """)
        st.markdown("""
                    Disadvantages of SVMs
                    - SVMs do not perform well on high dimensional data with many samples.

                    - SVMs work better only with Preprocessed data.

                    - They are harder to visualize.
        """)


    options5 = ["Naive Bayes"]
    selection5 = st.segmented_control("5.Random Forest", options5, selection_mode="single")
    if selection5=='Naive Bayes':
        st.markdown("""Naive Bayes is a simple yet powerful classification algorithm based on Bayes' theorem. 
        It is widely used for tasks such as text classification, spam filtering, sentiment analysis, and more.
        """)

with st.expander('7.Evaluation'):
    options = ["F1 score","R-squared","AOC & AUC","Root Mean Square Error"]
        
    selection = st.segmented_control("", options, selection_mode="single")
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
