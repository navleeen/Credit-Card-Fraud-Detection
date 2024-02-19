# Credit-Card-Fraud-Detection

## Objective:

The objective of the Credit Card Fraud Detection Problem is to build a model from past credit card transactions with the knowledge of the data that turned out to be a fraud. Then this model is used to identify whether a new transaction is fraudulent or not. So aim here is to detect 100% of the fraudulent transactions while minimizing the incorrect fraud classifications.

### Observations

The dataset is highly unbalanced, the positive class (frauds) account only for 0.17% of all transactions. This dataset presents transactions that occurred in two days, where we have around only 490 fraud transactions out of around 2 lakhs 80 thousand transactions.

The dataset consists of numerical values from the 28 ‘Principal Component Analysis (PCA)’ transformed features, namely V1 to V28. Also, no metadata provided about these PCA transformed features, so pre-analysis or feature study could not be possible.

The only features which have not been transformed with PCA are 'Time' and 'Amount'.

There is no missing value in the dataset.

Owing to such imbalance in data, an algorithm that does not do any feature analysis and predicts all the transactions as genuine will also achieve an accuracy of 99.828%. Therefore, accuracy is not a correct measure of efficiency in our case. We need some other standard of correctness while classifying transactions as fraud or non-fraud.

The ‘Time’ feature does not indicate the actual time of the transaction and is more of a list of the data in chronological order. So we assume that the ‘Time’ feature has little or no significance in classifying a fraud transaction. Therefore, we eliminate this column from further analysis.

### Challenges

Imbalanced Dataset: The data set is highly skewed, consisting of 490 frauds in a total of 2,84,000 observations. This resulted in only 0.17% of fraud cases. This skewed set is justified by the low number of fraudulent transactions.

Determining the appropriate evaluation parameters: There are two very common measures for the fraud detection techniques: false-positive and false-negative rates. These two measures have an opposite relationship, one decrease and other one increases. Accuracy is not a suitable metrics for credit card fraud detection technique, since the dataset is highly imbalanced. Therefore with very high accuracy, all fraudulent transactions can be misclassified. The error cost of misclassifying fraudulent instances is higher than the error cost of misclassifying legitimate instances, it is important to study not only the precision (correct classified instances) but also the sensibility(correct classified fraudulent instances) of each case.

If we use this dataset as the base for our predictive models and analysis, our model will probably overfit since it will "assume" that most transactions are not a fraud. But we don't want our model to assume, we want our model to detect patterns that give signs of fraud!

### EDA

In the EDA part, ran a few initial comparisons between the three columns - Time, Amount, and Class. I found the distribution of each feature, and relationship amounts each feature.

#### Preprocess the data

Before building the model, I preprocessed the data like scaling, balancing and then splitting the data.

1. Scaling the data - It is a good idea to scale the data so that the feature with lesser significance might not end up dominating the objective function due to its larger range. Features V1 to V28, already transformed with PCA, however amount feature is not scaled. So amount feature is scaled by taking its log.

2. Class Imbalance Solutions - 2.1 Under Sampling - Random undersampling deletes examples from the majority class and can result in losing information invaluable to a model.

2.2 Over Sampling - Random oversampling duplicates examples from the minority class in the training dataset and can result in overfitting for some models.

2.3 SMOTE - In this technique, instead of simply duplicating data from the minority class, we synthesize new data from the minority class.

3. Splitting the data - I have split the data into 70:30 ration. 70% of data is used as training data and the rest 30% is used as testing data. shuffle=True,stratify=y. Stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify. For example, our dataset contains around 0.17% of fraud transaction and 98.83% genuine transaction, so stratify=y will make sure that your random split has the same ratio.

Logistic Regression is a statistical model that tries to minimize the cost of how wrong a prediction is. Random Forests is an ensemble of decision trees that collectively predict if a transaction is fraudulent or not.

### Building the Model

I build the various models with a different algorithm after implementing the various class imbalance distribution algorithm to check what works well and what doesn't.

I build the models using

Logistic Regression

Once the model is built I evaluated each machine learning models and found which one is best. Then I used the best model to predict the unseen transaction to find whether it is fraud or genuine.

### Performance Measurement of Models

#### Accuracy: 
The measure of correct predictions made by the model – that is, the ratio of fraud transactions classified as fraud and non-fraud classified as non-fraud to the total transactions in the test data.

Recall / Sensitivity: Sensitivity, or True Positive Rate, or Recall, is the ratio of correctly identified fraud cases to total fraud cases.

#### Specificity: 
Specificity, or True Negative Rate, is the ratio of correctly identified non-fraud cases to total non-fraud cases.

Specificity = TN / (TN + FP)

#### Precision: 
Precision is the ratio of correctly predicted fraud cases to total predicted fraud cases.

#### F1 Score: 
F1 score a combination of recall and precision into one metric. F1 score is the weighted average of precision and recall, taking BOTH false positives and false negatives into account. Usually much more useful than accuracy, especially with uneven classes.

#### Receiver Operating Characteristics (ROC) Curve: 
It is an evaluation metric that helps identify the strength of the model to distinguish between two outcomes. It defines if a model can create a clear boundary between the positive and the negative class.

It is a plot between Sensitivity and ( 1 - Specificity ), which intuitively is a plot between True Positive Rate and False Positive Rate. It depicts if a model can clearly identify each class or not

Higher the area under the curve, better the model and it's the ability to separate the positive and negative class.

#### Result: 
All of the scores for Random Forest balanced with OverSampling and SMOTE models are very promising. Both models have a high Recall and ROC, which is exactly what I am looking for.
