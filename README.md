# Heart Satalog 

## Table of Contents
- [Dataset](#Dataset)
- [Project Overview](#overview)
- [Data Visualization](#data-visualization)
- [MLP and CNN Models](#mlp-and-cnn-Models)
- [neural network weight optimization](#Neural-Network-weight-optimization)
- [Swarm Optimization](#swarm-optimization)
- [Conclusion](#conclusion)
- [Accuracy Summary](#accuracy)

## Dataset
https://www.kaggle.com/datasets/mexwell/heart-disease-dataset

## Overview

This repository contains a comprehensive analysis and machine learning modeling and some optimization algorithms for a heart disease dataset. The project is divided into several parts:

1. **Data Visualization**: So we can understand more about the data we are dealing with.
2. **MLP and CNN Models**: Implementation and comparison of Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) models.
3. **Neural Network weight optimization**: Application of various optimization algorithmsand comparing them: Genetic Algorithm (GA), Simulated Annealing (SA), Gradient Descent (GD), and Hill Climbing to improve the models (which we will need to run the randomized_optimization notebook).
4. **Swarm Optimization**: Utilizing Swarm Optimization to update the weights of a neural network with 1 hidden layer.


  ## Data Visualization

Simple visualization on the dataset to gain more knowledge on what we are working with.
### Key Visualizations

- **Target Distribution**: Distribution of the target variable indicating the presence of heart disease.
- **Age vs. Max Heart Rate**: Scatter plot showing the relationship between age and maximum heart rate, colored by the presence of heart disease.
- **Correlation Heatmap**: Heatmap revealing correlations between the features.
- **Sex and Heart Disease**: Comparison of heart disease prevalence between males and females.
**Note**: each graph have been explained breifly in the Notebook.
  
## MLP and CNN Models

We implemented and compared the performance of MLP and CNN models on the heart disease dataset. and found out that even that the MLP got slightly higher accuracy but its not actually better because when we check the graph in the end of the notebook we can see that the validation and the training got a gap between them which means its overfitting therefore CNN is better in our case. Therefore, we got to understand that Higher accuracy doesn't necessarily means better result!


## Neural Network weight optimization
Various optimization techniques were applied to improve the neural network models:

- **Genetic Algorithm (GA)**
- **Simulated Annealing (SA)**
- **Gradient Descent (GD)**
- **Hill Climbing**
  
With 3 optimization problems:
- **One Max**
- **Flip Flop**
- **Knapsack**
  which have been used in our randomized_optimization Notebook

| Method                    | Accuracy | Sensitivity | Specificity | AUC  |
|---------------------------|----------|-------------|-------------|------|
| Gradient Descent (GD)     | 84%      | 88%         | 79%         | 83%  |
| Genetic Algorithm (GA)    | 54%      | 17%         | 99%         | 58%  |
| Simulated Annealing (SA)  | 84%      | 92%         | 75%         | 83%  |
| Hill Climbing (HC)        | 84%      | 89%         | 79%         | 84%  |


The optimization shows that Gradient Descent (GD) and Randomized Hill Climbing (RHC) performs almost the same with high accuracy and balanced specificity and sensitivity. While in Simulated annealing (SA) got the highest sensitivity, indicating its ability to correctly identify the positive cases, which is important for applications where false negatives are costly. However, Simulated annealing 's (SA) specificity is slightly lower, suggesting a higher rate of false positives. On the other hand, we got Genetic Algorithm (GA) shows the poorest and worst performance among all with low accuracy and sensitivity, despite its high specificity, indicating it is better at identifying true negatives but fails to capture the true positives effectively even the AUC for it is considered as low. Keep in mind that GD, SA, and RHC all show reasonably high AUC %, which demonstrates their effectiveness in distinguishing between the classes.
Overall, while GD and RHC are considered both good choices for our application, GA is the least effective one, especially in tasks requiring high sensitivity like our heart dataset which makes it unsuitable for us. Which takes us to SA, which stands out to be the best Algorithm we got out of all of them for our application.



**Note**: in order to runt this we have to run the randomized_optimziation.ipynb that could take a long time to finish computing sometimes (~2 hours) depending on hardware specs.
then open up the neural_network_weight_optimization, please note that the problems may take a long time to finish computing sometimes (~30) depending on hardware specs.



## Swarm Optimization

Swarm Optimization was used to update the weights of a neural network with one hidden layer. This approach aimed to find the optimal weights that minimize the error in predictions.
In this file we’ll be training a neural network using particle swarm optimization and For this we’ll be using the standard pyswarms. for optimizing the network’s weights and biases. This aims to demonstrate how the API is capable of handling custom-defined functions. we will be trying to classify the 3 iris species in the Iris Dataset.



  
## Conclusion

This project demonstrates the effectiveness of various ML models and optimization techniques in predicting heart disease. The data visualization helped in understanding the dataset, and tried some optimization techniques to show how each effects the model performance.



## Accuracy

| Model/Technique              | Accuracy |
|------------------------------|----------|
| MLP Model                    | 90%      |
| CNN Model                    | 89%      |
| Genetic Algorithm (GA)       | 54%     |
| Simulated Annealing (SA)     | 84%      |
| Gradient Descent (GD)        | 84%     |
| Hill Climbing                | 84%     |
| Swarm Optimization           | 86%      |

## Note: Please dont forget to change the file path!
