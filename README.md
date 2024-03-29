Loan Prediction Model Report

How to Run Section

The first step to running this project is to import the necessary libraries. Then import the test and training data set. After that you are free to run the entire file or run it by its different sections which are labeled with comments. I recommend running it in sections to accurately view all the sections of the model. To clarify, run each plotted graph individually to see them all. 


Topic Research

This project’s objective is to create a Loan prediction model that takes multiple variables and determines whether an individual will be granted a loan. Potential applications for this project are banks that are interested in using a model to determine the eligibility of a loan candidate or small companies that give personal loans can also use this model for the same reason. Some known challenges for building or implementing my model are data preparation and data cleaning. To clarify, it is important to have good data preparation so that the model can accurately weigh all the parameters. In addition, data cleaning is extremely important because missing data in the training set can give the model a bias toward a parameter that will cause the model to not give accurate predictions. Based on my research, the types of datasets that have been used in the past for loan prediction models are Categorical Datasets. These categorical Datasets represent features or characteristics of a person or an object. In this case it would be several characteristics of the individual applying for a loan such as applicant income or loan size. In the past the most common methods that were applied to loan prediction models were SVM, XGBoost, and random forest. In addition, it appears random forest and XGBoost are the best methods for accomplishing the loan prediction task. These two methods alternate on being the best depending on the quality of the dataset used during the training phase. The most common metric that is used to measure model success for the loan prediction task is accuracy.

Dataset

The dataset used in this project can be found on kaggle.com called Loan prediction Problem Dataset. It was last updated four years ago and has a size of 614 entries that consist of 13 parameters. These parameters include the loan ID, gender of the applicant, marital status, number of dependents, level of education, whether the individual is self-employed, income of the applicant, income of the co-applicant, the loan amount, the loan amount term, credit history of applicant, property area, and the loan status. The main challenge with the dataset is quantifying some of the different variables. For example, it is not explained whether the parameter called ApplicantIncome and CoApplicantIncome is a monthly income or a biweekly income, the parameter Loan_Amount_Term also does not specify if it is in months, and for the parameter of Credit_History it is a true or false rather than their actual credit history so it is hard to determine whether or not to include it as a parameter. When doing research on the dataset I could not find how the dataset was prepared nor labeled which is why I could not resolve the main change of the dataset which was quantifying some of the parameters. One potential bias that could be embedded in my dataset that could become a problem if this were to go into production is that the Gender parameter is not fairly represented in the data. Therefore, the model could become biased for or against women depending on the approval rate for women within the dataset.   

Data Analysis

There are 614 entries with unique loan ids. The amount of unique loan ids will determine the size of the training and validation set.  For gender, 80% are male applicants, 18% are female, and 2% are other. 

![image](https://user-images.githubusercontent.com/64668781/236993959-5745d524-98ff-4ef2-b9c1-d8f5dc5aaa68.png)

For marital status, 64.8% of applicants are married, while 35.2% are not married. Marital status is important because it can influence an applicant’s ability to repay the loan, which is an important factor in determining approval for a loan.

![image](https://user-images.githubusercontent.com/64668781/236994141-5125f90c-0486-416d-a9eb-ff19bd7dea0a.png)

For dependents, 56.1% have no dependents, 16.6% have one dependent, 16.4% have two dependents, while 8.3% have three or more dependents. This is an important parameter because it can influence the amount of money an applicant will have to repay the loan.  

![image](https://user-images.githubusercontent.com/64668781/236994224-72ef7748-55cb-4a11-bfa5-b1f850aa1fc4.png)

For self-employed, 81.4% are not self-employed, while 18.6% are self-employed. Self-employment can show how stable a person’s income is for paying the loan.

![image](https://user-images.githubusercontent.com/64668781/236994318-9b6d6479-3654-4c81-abe4-2906418ef174.png)

The range of the Loan amount and individual is applying for is represented by the graph below. Loan amount is used to calculate how much an applicant must pay a month to repay their loan.

![image](https://user-images.githubusercontent.com/64668781/236994561-e0aa86e8-98a8-470e-a672-09c3fe93305b.png)

The range of the Loan amount term is displayed in the bar chart below. Loan amount term is used to calculate how much an applicant must pay a month to repay their loan and the number of months the applicant will be paying for. 

![image](https://user-images.githubusercontent.com/64668781/236994647-1f302b76-e582-4d60-b465-ee2f23a09a75.png)

For education, 78% are graduates, while 22% are not graduates. The level of education can determine how much income an applicant can make, which is important in determining if an applicant can repay the loan. Applicant and co-applicant income are very important because they determine the amount of money an applicant has available to repay the loan. 

Data Cleaning

The data cleaning techniques I used were searching for missing values, replacing missing data with the most common entry of that parameter, and converting data types. The data cleaning method of searching for missing values is as the name suggests. Its goal is to identify how many data values are missing from each parameter. The reason I chose to use this on my dataset was to identify which parameters could be removed and which parameters needed to be fixed through data replacement. In continuation, data replacement of missing values is used to remove missing data from the dataset that can misdirect the weights of the parameters. Typically, in data replacement the average or most common entry is used to replace a null entry. The last data cleaning technique I used was converting data types which is essentially changing strings to int etc. I used this on my data set to convert string variables into int64 numbers which are easier for the model to understand and analyze. 

Data Processing

For my data processing, I implemented data discretization, generalization, and revising. Data discretization is used to turn continuous variables into categorical variables by splitting them into discrete ranges. I chose to use data discretization to test if my worse models could improve in accuracy is they did not have to deal with continuous values. I used data discretization on Total_Home_Income, LoanAmount, and Loan_Amount_Term. Data generalization is transforming a set of values into a more generic form. I chose to use generalization to make it easier for the model to train by having it compute similar values. I used data generalization on the fields called gender, married, dependents, education, self_employed, and property_area.  Data revision is a technique that involves changing the format of an existing dataset to make it more useful for analysis. I chose this technique so that I can combine similar fields into new fields. This will reduce the number of fields my model has to analyze without removing any data. I used data revision to combine the fields called Applicantincome and CoapplicantIcome into a new field called Total_Home_Income 

Model Implementation

I implemented SVM, K-Nearest Neighbor, Decision tree, Random Forest, and Gradient Boosting models for my loan prediction project. 

•	SVM is a supervised machine learning model that uses classification algorithms for two-group classification problems. The way it works is that it creates a hyperplane that best separates the two groups it is trying to classify and then uses it to make its predictions. The reason I chose to test this model is because my loan prediction model is a two-group classification problem. To clarify, it must predict whether the loan is approved or not. The strengths of SVM are: SVM works relatively well when there is a clear margin of separation between classes, more effective in high dimensional spaces, is effective in cases where the number of dimensions is greater than the number of samples and is relatively memory efficient. The weaknesses of SVM are: SVM is not suitable for large data sets, and poor performance when the data has significant noise.

•	K-Nearest neighbor is a nonparametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. This model works off the assumption that similar points can be found near one another. I chose to use this model since it is typically a good classification model, and a classification model is what I need. The strengths of k-nearest neighbor are: robust to noisy training data, and effective if the training data is large. The weaknesses of k-nearest neighbor are: need to determine value of parameter K, distance based learning is not clear which type of distance to use and which attribute to use to produce the best results, and computation cost is quite high because we need to compute distance of each query instance to all training samples. 
 
•	Decision trees are a non-parametric supervised learning method used for classification and regression. This model works by creating simple decision rules that are inferred from the data features which are then used to predict the value of the target variable. I chose this model to see if it could predict the target variable of loan status by analyzing the other variables. The strengths of Decision trees are: they require less effort for data preparation during pre-processing, they do not require normalization of data, does not require scaling of data, missing values in the data do not affect the process of building a decision tree, and is very intuitive and easy to explain. The weaknesses of a decision tree are: a small change in the data can cause a large change in the structure of the decision tree which can cause instability, sometimes its calculations can become more complex when compared to other algorithms, it takes longer to train the model, and it is bad at applying regression and predicting continuous values.

•	Random forest is a method that creates many decision trees and via majority rule will make the classification. The way this works is each individual tree in the random forest will make a class prediction and then the class with the most votes will become the model’s prediction. Since my model has many variables, I thought the random forest model would be better than a decision tree which is why I decided to test it. The strengths of Random Forest are: they have fast runtimes, and they are able to deal with unbalanced and missing data. The weaknesses of Random Forest are: it is bad at regression, and has problems with over-fitting if the data has noise. 

•	Gradient Boosting is a functional gradient algorithm that repeatedly selects a function that leads in the direction of a weak hypothesis or negative gradient so that it can minimize a loss function. This works by combining several weak learning models to produce a powerful predicting model. The reason I chose this model is because in my research it was one of the recommended models. The strengths of Gradient Boosting are: trains fast on large datasets, has support handling categorical features, and it can handle missing values on its own. The weaknesses of Gradient boosting are: it is prone to overfitting, models can be computationally expensive and take a long time to train, and it can be hard to interpret the final models.

Model Training and Tuning

Since I was the only person working on this project, I decided to not tune any hyperparameters. The reason is when I tried to change the hyperparameters the performance differences were unnoticeable due to the models fluctuating performances every time I ran them. Of the default models I used for this project, none of the models overfit nor did they underfit. 

Results

I measured the accuracy of my models by finding the mean of the outputs of the cross-validation score. This metric is good because it gives you a good representation of the accuracy even if there are outliers in the data. The results for all my models are below.

![image](https://user-images.githubusercontent.com/64668781/236995160-14ee4261-41c9-4530-98f4-b4060fa6fa9e.png)

In terms of the best training time, the decision tree classifier and the K-Neighbors classifier did best.  

![image](https://user-images.githubusercontent.com/64668781/236995195-f6015cf6-5b71-49de-8c46-4fd46e22c49b.png)

Therefore, the Gradient Boosting classifier did best in terms of accuracy, while the decision tree classifier did best in terms of training time.   

Discussion   

The model that clearly performed the best was the Decision tree model because it had over 70% accuracy and its time to train was less than .5 a second. When compared to Gradient boosting and random forest, which had 3 to 9 % more accuracy, it had the highest accuracy for its training time. If the goal of the model is accuracy, then Gradient boosting would be the best model, but when factoring training time Decision trees are best due to high accuracy and having the lowest training time. I think the Decision tree model was the best because of its simple design to solve classification problems which is why it had similar accuracy to more complex models such as gradient boosting and random forest, but its training time was at least half as such as the other two due to its simplicity. If I were to continue this project, I would experiment with the different hyperparameters each model has to test if I could improve the accuracy of my models. When I compared the results of my models, I found that I was around 3 to 5 percent less accurate than the best models for loan prediction. In addition, my results seemed to have average accuracy when compared to models that were not the best. If I planned to deploy this project into production, then I would do further testing with different hyperparameters in the training set to see if I can further improve the accuracy of my models. In addition, I would test the models with new datasets that are more balanced for the different parameters to see if the models maintain their accuracy or to identify any possible biases in my models. Last, if this model went into production, then I would make its decisions more interpretable by having it state the reasons why the applicant was denied. The method I would take to accomplish this task is to have the model identify the weight of each parameter. After that I would have the model analyze the applicant’s information and have it identify the parameter with the most weight that fails to meet the average entry of an approved candidate. While this system will not identify all the reasons an applicant was denied, it will give the applicant and bank an idea of what the applicant can try to change to become approved. For example, if a person was denied due to not presenting a credit score, then the bank can identify this and inform the applicant of their missing credit score which could potentially get them approval if they present it. 
  
References  

Chatterjee, Debdatta. “Loan Prediction Problem Dataset.” Kaggle, 12 Mar. 2019, www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?select=train_u6lujuX_CVtuZ9i.csv. 

Saini, Anshul. “Gradient Boosting Algorithm: A Complete Guide for Beginners.” Analytics Vidhya, 14 Oct. 2021, www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/. 

Simplilearn. “Loan Approval Prediction Using Machine Learning | Machine Learning Projects 2022 | Simplilearn.” YouTube, 20 Sept. 2022, www.youtube.com/watch?v=x2NrPeHSPU0&ab_channel=Simplilearn. 

Stecanella, Bruno. “Support Vector Machines (SVM) Algorithm Explained.” MonkeyLearn Blog, 22 June 2017, monkeylearn.com/blog/introduction-to-support-vector-machines-svm/. 



