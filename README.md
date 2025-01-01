# Clustering and Classification Project

## Overview
This project analyzes a dataset on crimes against women in India (2001-2021) using both clustering and classification techniques. The objective is to uncover patterns in the data through unsupervised learning and use these patterns to classify new instances with supervised learning. This approach helps in understanding the trends and predicting potential crime scenarios.

## Project Understanding
The analysis aims to:
- Identify meaningful clusters within the dataset to group similar states or crime types.
- Develop classification models to predict categories for new data points.

### Goals
1. Perform clustering to understand the inherent structure of the dataset.
2. Use the clustering results to enhance classification models.
3. Provide actionable insights from the data analysis.

## Data Understanding
The dataset contains information on crimes against women in India, with details such as:
- **Crime Types**: Different categories of crimes.
- **States/Regions**: Geographic distribution.
- **Year**: Temporal trends from 2001 to 2021.
- **Other Features**: Additional attributes relevant to the analysis.

### Dataset Highlights
- **Size**: 2500 rows with multiple features.
- **Challenges**:
  - Missing values in certain fields.
  - Potential biases due to underreporting or inconsistent definitions.
  - High-dimensional data requiring preprocessing and feature selection.

## Data Processing
Steps taken to prepare the data for analysis:
1. **Data Cleaning**
   - Handled missing values using imputation or removal techniques.
   - Standardized feature definitions for consistency.
2. **Feature Engineering**
   - Created new features based on domain knowledge.
   - Encoded categorical variables using methods like one-hot encoding.
3. **Normalization/Scaling**
   - Applied StandardScaler or MinMaxScaler to normalize numeric features for clustering and classification.

## Exploratory Data Analysis (EDA)
1. **Descriptive Statistics**
   - Summary statistics to understand the distribution of features.
2. **Visualization**
   - Used Seaborn and Matplotlib to plot:
     - Crime trends over the years.
     - Geographic distribution of crime rates.
     - Correlation heatmaps for feature relationships.
3. **Insights**
   - Identified states with consistently high crime rates.
   - Highlighted trends in specific crime types over time.

## Modeling
### Clustering
1. **Techniques Used**
   - **K-Means**: For identifying groups with similar crime profiles.
   - **DBSCAN**: For detecting density-based clusters.
2. **Visualization**
   - Plotted clusters using PCA for dimensionality reduction.

### Classification
1. **Algorithms Implemented**
   - Decision Trees, Random Forest, K-Nearest Neighbors, SVM, Naive Bayes.
2. **Hyperparameter Tuning**
   - Performed using BayesSearchCV for optimal model performance.
3. **Feature Selection**
   - Applied Recursive Feature Elimination (RFE) to enhance model accuracy.

## Evaluation
1. **Clustering Evaluation**
   - Used metrics like Silhouette Score and Davies-Bouldin Index to validate clusters.
2. **Classification Metrics**
   - Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
3. **Results**
   - Clustering revealed key patterns and helped create features for classification.
   - Classification models achieved high accuracy with the Random Forest performing best.

## Conclusion
- The combination of clustering and classification provided valuable insights into crimes against women in India.
- Clustering highlighted regions and crime types needing attention.
- Classification models demonstrated potential for predicting future trends.

### Recommendations
1. Expand the dataset with more recent data for up-to-date analysis.
2. Address potential biases in the dataset for more accurate modeling.
3. Use the findings to assist policymakers in resource allocation and crime prevention strategies.

