# MachineLearningProject
Loan prediction model


Project Requirements 
1.Topic Research: Research past work related to your project
  a. Describe your project objective
  My project objective is to create a Loan prediction model that takes multiple variables and determines whether an individual will be granted a loan.
  
  b. What are the potential applications of your project?
  Potential applications of my project are banks using the model to determine the eligibility of a loan candidate or small companies that give personal loans can also use this model for the same reason.
  
  c. What are the known challenges of building or implementing a model for your topic?
  Some known challenges for building or implementing my model are data preparation and data cleaning. To clarify, it is important to have good data preparation so that the model can accurately weigh all the parameters. In addition, data cleaning is extremely important because missing data in the training set can give the model a bias toward a parameter that will cause the model to not give accurate predictions.
  
  d. What types of datasets have been used in the past?
  The types of datasets that have been used in the past for loan prediction models are Categorical Datasets. Categorical Datasets represent features or characteristics of a person or an object. In this case it would be the individual applying for a loan.
  
  e. What types of methods have been applied to related research in the past?
  In the past the most common methods that were applied to loan prediction models were SVM, XGBoost, and random forest.
  
  f. What is the state of the art (SOTA) method for your ML task?
  SVM and XGBoost are the best methods for my task. These two methods alternate on being the best depending on the dataset used. 
  
  g. What metrics are used for measuring model success in this task?
  The metrics used to measure model success for this task are accuracy, precision, and recall. 
  
2. Dataset: Address the following about your dataset
  a. Describe the dataset for your project.
  My dataset has a size of 614 entries that consist of 13 parameters. These parameters include the loan ID, gender of the applicant, marital status, number of dependents, level of education, whether the individual is self-employed, income of the applicant, income of the co-applicant, the loan amount, the loan amount term, credit history of applicant, property area, and the loan status. 
  
  b. What are some challenges of the dataset you are working with?
  Some of the challenges of the dataset are determining the weights of the different variables within the model and determining whether it is a good idea to replace the missing data with the average. 
  
  c. How was your dataset prepared (i.e. how was data collected)? How was it labeled (if at all)? (If you cannot find this information, just mention so)
  I could not find this information.
  
  d. Are there any potential biases embedded in your dataset that may lead to problems if ever used in production?
  One potential bias that could be embedded in my dataset that could become a problem is that the Gender parameter is not fairly represented in the data. Therefore, the model could become biased for or against women depending on the approval rate for women.   

3. Data Analysis: Perform an analysis of your dataset. (Remember this must not be done on the test set).
  a. Provide statistics about your dataset to give a rough overview of your data in numbers
  There are 614 entries with unique loan ids. For gender, 80% are male applicants, 18% are female, and 2% are other. For marital status, 64.8% of applicants are married, while 35.2% are not married. For dependents, 56.1% have no dependents, 16.6% have one dependent, 16.4% have two dependents, while 8.3% have three or more dependents. For self-employed, 81.4% are not self-employed, while 18.6% are self-employed. For education, 78% are graduates, while 22% are not graduates.  
  
  b. Explain what insight each statistic provides about your dataset with regards to the ML task
  The amount of unique loan ids represents the size of my training and validation set. Marital status can influence an applicant’s ability to repay the loan, which is an important factor in determining approval for a loan. The number of dependents influences the amount of money an applicant will have to repay the loan. The level of education can determine how much income an applicant can make, which is important in determining if an applicant can repay the loan. Self-employment can show how stable a person’s income is for paying the loan. Applicant and co-applicant income are very important because they determine the amount of money an applicant has access to repay the loan. Loan amount and loan amount term are used to calculate how much an applicant has to pay a month to repay their loan. 

4. Data Cleaning: Address the following questions about data cleaning…
  a. Implement at least 2 data cleaning techniques to your dataset
  The data cleaning techniques I used were searching for missing values, replacing missing data with the most common entry of that parameter, and converting data types.
  
  b. Explain each data cleaning method and why you chose to apply it to your dataset
  The data cleaning method of searching for missing values is as the name suggests. Its goal is to identify how many data values are missing from each parameter. The reason I chose to use this on my dataset was to identify which parameters could be removed and which parameters needed to be fixed through data replacement. In continuation, data replacement of missing values is used to remove missing data from the dataset that can misdirect the weights of the parameters. Typically, in data replacement the average or most common entry is used to replace a null entry. The last data cleaning technique I used was converting data types which is essentially changing strings to int etc. I used this on my data set to convert string variables into int64 numbers which are easier for the model to understand and analyze. 

5. Data Processing: Transform your dataset in ways that would be useful for your project objective.
  a. Implement at least 3 data transformation methods to your dataset
  I plan to use data discretization, generalization, and revising. 
  
  b. Explain each data processing method and why you chose to apply it to your dataset
  Data discretization is used to turn continuous variables into categorical variables by splitting them into discrete ranges. I chose to use data discretization to test on more models (like regression) that could not be used due to continuous variables like applicant income. Data generalization is transforming a set of values into a more generic form. I chose to use generalization to make it easier for the model to train by having it compute similar values. Data revision is a technique that involves changing the format of an existing dataset to make it more useful for analysis purposes. I chose this technique so that I can combine the similar fields of applicant income and co-applicant income into a new field of total household income, this will reduce the number of fields my model will can to analyze without removing any data.

6.Model Implementation: Implement ML models for your task
  a. Implement at least 3 significantly different ML models for your task (You don’t need to implement models from scratch. Using existing python libraries is allowed)
  I plan to use SVM, K-Nearest Neighbor, Decision tree, Random Forest, and Gradient Boosting.
  
  b. Explain how each model works and why you chose to use it for your project (Explain each model in detail with accompanying visuals if needed)
  SVM is a supervised machine learning model that uses classification algorithms for two-group classification problems. The way it works is that it creates a hyperplane that best separates the two groups it is trying to classify and then uses it to make its predictions. The reason I chose to test this model is because my loan prediction model is a two-group classification problem. To clarify, it must predict whether the loan is approved or not.
  K-Nearest neighbor is a nonparametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. This model works off the assumption that similar points can be found near one another. I chose to use this model since it is typically a good classification model, and a classification model is what I need.
  Decision trees are a non-parametric supervised learning method used for classification and regression. This model works by creating simple decision rules that are inferred from the data features which are then used to predict the value of the target variable. I chose this model to see if it could predict the target variable of loan status by analyzing the other variables.
  Random forest is a method that creates many decision trees and via majority rule will make the classification. The way this works is each individual tree in the random forest will make a class prediction and then the class with the most votes will become the model’s prediction. Since my model has many variables, I thought the random forest model would be better than a decision tree which is why I decided to test it. 
  Gradient Boosting is a functional gradient algorithm that repeatedly selects a function that leads in the direction of a weak hypothesis or negative gradient so that it can minimize a loss function. This works by combining several weak learning models to produce a powerful predicting model. The reason I chose this model is because in my research it was one of the recommended models. 
  
  c. Explain the strengths and weaknesses of each model you selected
  The strengths of SVM are: SVM works relatively well when there is a clear margin of separation between classes, more effective in high dimensional spaces, is effective in cases where the number of dimensions is greater than the number of samples and is relatively memory efficient. The weaknesses of SVM are: SVM is not suitable for large data sets, and poor performance when the data has significant noise.
  The strengths of k-nearest neighbor are: robust to noisy training data, and effective if the training data is large. The weaknesses of k-nearest neighbor are: need to determine value of parameter K, distance based learning is not clear which type of distance to use and which attribute to use to produce the best results, and computation cost is quite high because we need to compute distance of each query instance to all training samples. 
  The strengths of Decision trees are: they require less effort for data preparation during pre-processing, they do not require normalization of data, does not require scaling of data, missing values in the data do not affect the process of building a decision tree, and is very intuitive and easy to explain. The weaknesses of a decision tree are: a small change in the data can cause a large change in the structure of the decision tree which can cause instability, sometimes its calculations can become more complex when compared to other algorithms, it takes longer to train the model, and it is bad at applying regression and predicting continuous values. 
  The strengths of Random Forest are: they have fast runtimes, and they are able to deal with unbalanced and missing data. The weaknesses of Random Forest are: it is bad at regression, and has problems with over-fitting if the data has noise. 
  The strengths of Gradient Boosting are: trains fast on large datasets, has support handling categorical features, and it can handle missing values on its own. The weaknesses of Gradient boosting are: it is prone to overfitting, models can be computationally expensive and take a long time to train, and it can be hard to interpret the final models.

7. Model Training and Tuning: Train and tune your models as you train them
  a. Did any of your models ever overfit? What did you do to address this?
  I did not check for overfitting in the models. I plan to check this by comparing the accuracy scores of the training, validation, and test set. 
  
  b. Did any of your models underfit? What did you do to address this?
  I did not check for underfitting in the models. I plan to check this by comparing the accuracy scores of the training, validation, and test set. 
  
  c. For each model, which hyperparameters did you tune and what effect did those changes have on the model performance?
  I did not change any of the hyperparameters, I am planning on doing that a couple of days before the final submission. 

8. Results: Training your models, perform a final test and estimate your model performance (Give the results with tables or graphs rather than answering the questions below in words)
  a. How do you measure the accuracy model? Why is this metric(s) good?
  I measured the accuracy model by finding the mean of the outputs of the cross-validation score. This metric is good because it gives you a good representation of the accuracy even if there are outliers in the data. 
  
  b. What is the training time of your best models?
  I have not checked the training time for my models. I will implement that next week.
  
  c. What is the size (memory) of your best model?
  I have not checked the size of my models. I will implement that next week.
  
  d. What model performed the best in each of the above criteria?
  For accuracy, Gradient Boosting classifier did best. The two other categories are unknown.
  
  e. Include images of sample outputs of your model
  ![image](https://user-images.githubusercontent.com/64668781/235278369-56328038-4bda-4e00-9be7-ba616032c3e2.png)

9. Discussion: After training, tuning, and testing your models, do a post analysis of your experiments and models you have created
  a. Was there a single model that clearly performed the best?
  Based on accuracy, the gradient boosting classifier did best. 
  
  b. Why do you think the models that performed the best were the most successful?
  I think that gradient boosting and Random Forest did best because they were made to solve classification problems quickly.
  
  c. What data transformations or hyperparameter settings lead to the most successful improvements in model performance?
  This is unknown to me because I have not implemented any of these changes. I was just focused on getting the models to work. 
  
  d. Were the above changes that lead to improvements also observed in any of research related to your project
  Again, I have not made any changes to the hyperparameters and only used generalization in data transformation which was done consistently in my research.
  
  e. If you were to continue this project, describe more modeling, training, or advanced validation techniques
  I don't know what I would add to this because it is not finished yet.
  
  f. Note any other interesting observations or findings from your experiments
  I found everything to be ordinary and got similar results to the research I did. 
  
  g. What potential problems or additional work in improving or testing your model do you foresee if you planned to deploy it to production?
  At this current moment I would say I need to further test different training hyperparameters to see if they impact the accuracy of the model. Also, I would test it with a new dataset to see if it maintains its accuracy.  
  
  h. If you were to deploy your model in production, how would you make your models' decisions more interpretable?
  I would have it predict the approval of the applicant and state why they were denied if they were denied. 
  
10. References: Include references to research, tutorials, or other resources you used throughout your project
I will add this when I am done with the project. 
