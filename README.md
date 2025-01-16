# Overview
Customer Segmentation is one the most important applications of unsupervised learning. Using clustering techniques, companies can identify the several segments of customers allowing them to target the potential user base. In this machine learning project, we will make use of K-means clustering which is the essential algorithm for clustering unlabeled dataset. Before ahead in this project, learn what actually customer segmentation is.

![image](https://github.com/user-attachments/assets/df2d1c2c-6c3c-44f8-81e9-409a1dc64ee1)

## What is Customer Segmentation
Customer Segmentation is the process of division of customer base into several groups of individuals that share a similarity in different ways that are relevant to marketing such as gender, age, interests, and miscellaneous spending habits.

Companies that deploy customer segmentation are under the notion that every customer has different requirements and require a specific marketing effort to address them appropriately. Companies aim to gain a deeper approach of the customer they are targeting. Therefore, their aim has to be specific and should be tailored to address the requirements of each and every individual customer. Furthermore, through the data collected, companies can gain a deeper understanding of customer preferences as well as the requirements for discovering valuable segments that would reap them maximum profit. This way, they can strategize their marketing techniques more efficiently and minimize the possibility of risk to their investment.

The technique of customer segmentation is dependent on several key differentiators that divide customers into groups to be targeted. Data related to demographics, geography, economic status as well as behavioral patterns play a crucial role in determining the company direction towards addressing the various segments
 ## Datasets
 The dataset leveraged in this project captures approximately 8950 active credit card users’ behavioral
 activity over a period of six months. It provides analysis at the level of the individual customer through 18
 different behavioral variables. This is obtained from Kaggle[6] and contains valuable information on
 customer spending patterns and payment trends. Although one issue that needs to be addressed
 concerning this dataset is the dealing of missing values and scaling of the data for cluster purposes.
 The dataset has the following key variables, briefly summarized below:
 CUST_ID: Unique identifier for each credit card holder (Categorical).
 BALANCE:Total balance remaining in the account, available for making purchases.
 BALANCE_FREQUENCY:Ascorebetween 0 and 1 indicating how often the balance is updated (1 = frequent, 0 = infrequent).
 PURCHASES:Total amount of purchases made by the account holder.
 ONEOFF_PURCHASES:Maximumamount spent in a single transaction.
 INSTALLMENTS_PURCHASES:Total amount spent on installment-based purchases.
 CASH_ADVANCE:Amountof cash advanced by the user.
 PURCHASES_FREQUENCY:Ascorebetween 0 and 1 reflecting how often purchases are made (1 = frequent, 0 = infrequent).
 ONEOFF_PURCHASES_FREQUENCY:Frequency of one-off purchases (1 = frequent, 0 = infrequent).
 PURCHASES_INSTALLMENTS_FREQUENCY:Frequency of installment-based purchases (1 = frequent, 0 = infrequent).
 CASH_ADVANCE_FREQUENCY:Frequency of cash advances taken by the user.
 CASH_ADVANCE_TRX:Numberoftransactions involving cash advances.
 PURCHASES_TRX:Number of purchase transactions made.
 CREDIT_LIMIT: Maximum credit limit available to the cardholder.
 PAYMENTS:Total amount of payments made by the user.
 MINIMUM_PAYMENTS:Minimumamount of payments made.
 PRC_FULL_PAYMENT:Percentage of full payments made by the user.
 TENURE:Duration of credit card usage by the user

 # Algorithms Used in this Project
 ## 1. K-Means Clustering
 K-Means is a popular clustering algorithm that partitions data into k clusters by minimizing the distance
 between data points and the cluster centroids. It's a centroid-based clustering method, meaning it seeks to
 minimize intra-cluster variance by iteratively adjusting the positions of cluster centers.
 Mathematics Behind K-Means:
 1. Initialization: Start by selecting kkk initial cluster centroids randomly.
 2. Assignment Step: Assign each data point to the nearest centroid based on Euclidean distance.
 3. Update Step: Calculate the new centroid of each cluster as the mean of all data points assigned to
 it.
 4. Convergence: Repeat the assignment and update steps until the centroids no longer change
 significantly or a maximum number of iterations is reached.
 The algorithm minimizes the Within-Cluster Sum of Squares (WCSS), which is defined as:
 𝑊𝐶𝑆𝑆 = ∑𝑘
 𝑖=1
 ∑𝑥∈𝑐 𝑖
 ∣∣𝑥 − µ 𝑖
 ∣| 2
 𝐶
 where isthe i-th cluster and is the centroid of .
 𝑖
 µ
 𝑖
 ## 2. DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
 𝐶
 𝑖
DBSCAN is a density-based clustering algorithm that groups points closely packed together (with many
 neighbors) and marks points in low-density regions as noise or outliers. It doesn’t require specifying the
 number of clusters in advance, unlike K-Means.
 Mathematics Behind DBSCAN:
 1. Core Points: A point is a core point if it has at least a minimum number of points (minPts) within
 a given radius (ϵ).
 2. Directly Reachable: A point A is directly reachable from point B if A is within the ϵ-distance of
 Band Bis a core point.
 3. Density-Connected: Two points are density-connected if they are both reachable from a common
 core point.
 4. Cluster Formation: DBSCAN clusters are formed by connecting density-reachable points,
 allowing it to find arbitrarily shaped clusters.
 ## 3. GMM(Gaussian Mixture Model)
 Gaussian Mixture Model (GMM) is a probabilistic model that assumes the data is generated from a
 mixture of multiple Gaussian distributions with unknown parameters. Each Gaussian represents a cluster,
 allowing GMM to model clusters that may overlap and have different shapes.
 Mathematics Behind GMM:
 1. Gaussian Components: Each cluster is represented by a Gaussian (normal) distribution with
 parameters (mean μ and covariance Σ).
 2. Expectation-Maximization (EM): GMM uses the EM algorithm to estimate parameters.
 ○ Expectation Step (E-step): Calculate the probability of each data point belonging to
 each Gaussian component.
 ○ Maximization Step (M-step): Update the parameters (mean, covariance, and mixing
 coefficients) of each Gaussian component based on these probabilities.
 3. Likelihood Maximization: GMM maximizes the likelihood of the data under the mixture model,
 assigning each point a probability of belonging to each cluster

 # Exploratory Data Analysis (EDA)
  ## 1. Correlation Heatmap
 ![image](https://github.com/user-attachments/assets/97563ee8-21a2-48a2-952c-d270c788b421)
## 2. Boxplot Analysis
![image](https://github.com/user-attachments/assets/94833f10-3f01-4331-89b3-03eac76fffff)
## 3.Log-Transformed Box Plot
![image](https://github.com/user-attachments/assets/68c89db1-bbf7-4257-b0d6-a84e5384b26a)
## 4.  Pairplot Analysis
![image](https://github.com/user-attachments/assets/32a40590-54d4-4fa8-878d-c4b365709dde)

# Methodology
![image](https://github.com/user-attachments/assets/faab6793-98b8-4723-a97a-601fbb162e50)

# Visualizing the Clusters
## K-Means Clustering Visualization
 To visualize the clusters from K-Means, we applied PCA to reduce the data to 2D and plotted
 the clusters:
 ![image](https://github.com/user-attachments/assets/1a81ce58-4168-4ac7-9829-0475b2c9ffe1)

 ## GMMClustering Visualization
 Similarly, GMM clustering results were visualized after dimensionality reduction using PCA:
 ![image](https://github.com/user-attachments/assets/992db0a1-1643-4fc0-acee-3a39a069d43c)

 ## DBSCANClustering Visualization
For DBSCAN, we visualized the clusters along with the noise points detected
![image](https://github.com/user-attachments/assets/7feeecc0-9be9-487f-897f-ecd0209bbf64)

 # Best Clustering Method
 Based on the evaluation metrics, K-Means and DBSCAN emerged as the best clustering methods for this
 dataset, with comparable Silhouette Scores and similar performance across other metrics.
  ### Silhouette Scores:
 ● K-MeansSilhouette Score: 0.6397
 ● GMMSilhouetteScore: 0.6304
 ● DBSCANSilhouette Score: 0.6397
 
 The choice between K-Means and DBSCAN depends on the specific requirements of the clustering task:
 ● K-Meansprovides hard clustering, where each data point belongs to a single, definitive cluster. It
 is ideal for applications requiring clear, distinct clusters and straightforward interpretation.
 ● DBSCANisuseful when the data contains noise or irregular cluster shapes, making it robust to
 outliers and suitable for identifying clusters with varying densities.
 Here we choose K-Means as the preferred clustering method for this project due to its simplicity and
 effectiveness in producing well-defined clusters
 
 # Random Forest Model Results and Deployment
 ## Save New labeled Clustered Dataset For Random Forest:
 df_cust.to_csv("New_Clustered_Customer_Data.csv")
 Train Random Forest Classifier:
 rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
 rf_model.fit(X_train, y_train)
 rf_y_pred = rf_model.predict(X_test)
 
### Random Forest Classifier Results: The Random Forest Classifier was evaluated using a confusion
 matrix and classification metrics:
 Confusion Matrix:
 [[218
 2
 1 13]
 [ 3 635 7 17]
 [ 3 18 87 0]
 [ 11 14 0 761]]
 Classification Report:
0
 0.93
 0.93
 0.93
 234
 1
 2
 3
 accuracy
 macro avg
 weighted avg
 ● KeyMetrics:
 0.95
 0.92
 0.96
 0.94
 0.95
 ○ Accuracy: 95%
 0.96
 0.81
 0.97
 0.92
 0.95
 0.95
 0.86
 0.97
 0.95
 0.93
 0.95
 ○ Precision: Ranges from 0.92 to 0.96 across classes.
 662
 108
 786
 1790
 1790
 1790
 ○ Recall: Highest for class 3 (0.97), but slightly lower for class 2 (0.81).
 ○ F1-Score: Averages 0.93, indicating a good balance between precision and recall.
 ## Model Deployment:
 ### ● Savingthe Model:
 The trained Random Forest model was saved using pickle for deployment:
 filename_rf = 'random_forest_model.sav'
 pickle.dump(rf_model, open(filename_rf, 'wb'))
 ### ● LoadingandTesting the Model:
 After saving the model, it was loaded again to test its accuracy on unseen data:
 loaded_model = pickle.load(open(filename_rf, 'rb'))
 result = loaded_model.score(X_test, y_test)
 print(result, '% Accuracy')
 Model Accuracy: 95.03%

# Input Screen

![image](https://github.com/user-attachments/assets/9baa22b7-07ae-43a6-ba92-b6ccd6e791fd)

# Project Title

Customer Segmentation Using Clustering


## Authors

- Dinesh Chand Foujdar

## Package Requirements
To run this project, you need the following packages and libraries installed:

- matplotlib
- streamlit
- scikit-learn
- pandas
- plotly
- seaborn

Ensure you have these installed in your environment. You can install them using the `requirements.txt` file provided.





## Setup Instructions
### Deployment on Hugging Face
The project is deployed on Hugging Face Spaces using the following configuration:
---
- title: Customer Segmentation Using Clustering
- emoji: 👀
- colorFrom: red
- colorTo: purple
- sdk: streamlit
- sdk_version: 1.40.0
- app_file: app.py
- pinned: false
- license: mit
- short_description: Divides the customer base into small groups.
---
##  Steps to Deploy on Hugging Face
- Create a New Space:
- Choose Streamlit as the SDK:
- Upload your project files (app.py,random_forest_model.sav, New_Clustered_Customer_data.csv requirements.txt etc.) to the Space:
- Configure the Space:
- Check requirements.txt:
- Deploy:
Once all files are uploaded and the configuration is set, the app will automatically deploy. You can access it through the Hugging Face Space URL.


## 
This `README.md` file now includes instructions for deploying your project on Hugging Face and makes sure that all dependencies are correctly installed. The `app.py` file is updated to include a black title using HTML and CSS.
