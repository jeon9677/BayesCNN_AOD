# Python codes for 'A Bayesian Convolutional Neural Network-based Generalized Linear Model'

## Package version
* numpy version: 1.19.5
* pandas version: 1.2.1
* scipy version: 1.6.0
* statsmodels version: 0.12.1
* keras version: 2.4.3
* pyreader: 0.4.9


## Codes for BayesCNN    
### BayesCNN/update_all50_days.py for training BayesCNN 
* BayesCNN (https://proceedings.mlr.press/v48/gal16.pdf) 
* Example command statement 
```diff
python update_all50_days.py 
```
  
### BayesCNN/update_prediction_50days.py for prediction
* This code generate prediction samples from predictive distribution using MC dropout
* You have the flexibility to adjust the quantity of MC dropout samples indicated as 'range(number of MC dropout)' 
* In this code, the number of MC dropout is 100  
* Example command statement for generating predictive distribution 
```diff
python update_prediction_50days.py 
```

