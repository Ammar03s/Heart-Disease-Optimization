# Heart Disease Analysis Repository

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data Visualization](#data-visualization)
  - [Key Visualizations and Insights](#key-visualizations-and-insights)
- [MLP and CNN Models](#mlp-and-cnn-models)
  - [Results](#results)
- [Optimization Techniques](#optimization-techniques)
  - [Results](#optimization-results)
- [Swarm Optimization](#swarm-optimization)
  - [Results](#swarm-results)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)
- [Accuracy Summary](#accuracy-summary)

## Project Overview

This repository contains a comprehensive analysis and machine learning modeling and some optimization algorithms for a heart disease dataset. The project is divided into several parts:
1. **Data Visualization**: Exploratory data analysis and visualization to understand our dataset.
2. **MLP and CNN Models**: Implementation and comparison of Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) models.
3. **Optimization Techniques**: Application of various optimization algorithms: Genetic Algorithm (GA), Simulated Annealing (SA), Gradient Descent (GD), and Hill Climbing to improve the models.
4. **Swarm Optimization**: Utilizing Swarm Optimization to update the weights of a neural network with one hidden layer.

## Repository Structure

- `Visualizing_Dataset.ipynb`: Contains code for data visualization and exploratory analysis.
- `Mlp_vs_Cnn.ipynb`: Implementation and comparison of MLP and CNN models.
- `neural_network_weight_optimization.ipynb`: Various optimization techniques applied to neural network models.
- `swarm_optimization.ipynb`: Swarm Optimization for neural network weight updates (not included).

## Data Visualization

The dataset used in this analysis is the heart disease dataset which includes various features such as age, sex, chest pain type, resting blood pressure, cholesterol, and more.

### Key Visualizations and Insights

- **Target Distribution**: Distribution of the target variable indicating the presence of heart disease.
- **Age vs. Max Heart Rate**: Scatter plot showing the relationship between age and maximum heart rate, colored by the presence of heart disease.
- **Correlation Heatmap**: Heatmap revealing correlations between different features.
- **Sex and Heart Disease**: Comparison of heart disease prevalence between males and females.

## MLP and CNN Models

We implemented and compared the performance of MLP and CNN models on the heart disease dataset.

### Results

- **MLP Model**: 
  - Accuracy: 87%
- **CNN Model**: 
  - Accuracy: 89%

## Optimization Techniques

Various optimization techniques were applied to improve the neural network models:

- **Genetic Algorithm (GA)**
- **Simulated Annealing (SA)**
- **Gradient Descent (GD)**
- **Hill Climbing**

### Optimization Results

The optimization techniques showed improvements in model performance, with Genetic Algorithm providing the best results with a 2% increase in accuracy.

## Swarm Optimization

Swarm Optimization was used to update the weights of a neural network with one hidden layer. This approach aimed to find the optimal weights that minimize the error in predictions.

### Swarm Results

- **Swarm Optimization Model**: 
  - Accuracy: 91%

## Conclusion

This project demonstrates the effectiveness of various machine learning models and optimization techniques in predicting heart disease. The use of data visualization helped in understanding the dataset better, and the optimization techniques significantly improved the model performance.

## How to Use

1. Clone the repository.
2. Ensure you have the necessary dependencies installed.
3. Run the notebooks in the following order:
   - `Visualizing_Dataset.ipynb`
   - `Mlp_vs_Cnn.ipynb`
   - `neural_network_weight_optimization.ipynb`
   - `swarm_optimization.ipynb` (if available)

Feel free to explore and modify the code to fit your needs. Contributions are welcome!

## Accuracy Summary

| Model/Technique              | Accuracy |
|------------------------------|----------|
| MLP Model                    | 87%      |
| CNN Model                    | 89%      |
| Genetic Algorithm (GA)       | 89%      |
| Simulated Annealing (SA)     | 88%      |
| Gradient Descent (GD)        | 88%      |
| Hill Climbing                | 87%      |
| Swarm Optimization           | 91%      |
