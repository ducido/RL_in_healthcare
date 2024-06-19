# Actor-Critic Model for Healthcare Billing Prediction

## Overview
This repo applies the Actor-Critic architecture to estimate billing amounts in the healthcare domain. Unlike Deep Q learning method, this model handles continuous data effectively, making it particularly suitable for financial predictions such as billing.

## Dataset
The model is trained on a comprehensive healthcare dataset available on Kaggle. The dataset includes 27 features which represent different states within the healthcare context, with the billing amount treated as the action variable.

**Access the dataset here:** [Healthcare Dataset on Kaggle](https://www.kaggle.com/datasets/prasad22/healthcare-dataset/data)

## Actor-Critic Architecture
The Actor-Critic method combines the benefits of both policy-based and value-based approaches for reinforcement learning. This architecture is chosen for its proficiency in dealing with continuous action spaces, making it ideal for predicting billing amounts which are continuous in nature.


![Deep Actor-Critic Reinforcement Learning](https://github.com/ducido/RL_in_healthcare/assets/122498122/15ba9422-bdf5-4fd4-bd4a-2153bc3b2a2a)



## Installation
tensorflow==2.11
pandas
numpy
...

## Disclaimer
This repo does not focus on the results, just implemetation
-----------------------------------------------------------
*Big thanks to TonManhKien for beautiful code*
