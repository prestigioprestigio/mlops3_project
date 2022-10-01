# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a Logistic Regression model from the sci-kit learn library  
Default parameters are used.  
No hypertuning.  

## Intended Use
Given census data, the purpose of the model is to predict whether a person makes above or below 50K USD yearly.  
  
## Training and Evaluation Data
The data is first cleaned from redundant space characters in the headers.
Afterwards, 10% of the data, named `census_ref_eval.csv` is hidden away for evaluation. The reamining 90% are stored in `census_ref.csv`.  
`census_ref.csv` is then split into a training and testing set, with a ration of 80/20, respectively.
All data needs to go through a preprocessing stage first, where categorical features are one-hot encoded, label binarized, and numerical features scaled.


## Metrics
The model's performance on the test set is as follows:  
Precision: 0.74  
Recall: 0.6  
Fbeta: 0.66  
Accuracy: 0.86  

## Ethical Considerations
Some countires are underrepresented in the data, which leads to less variability, and could introduce bias.  
It is then recommended to increase those countries' samples to try to mitigate that.
In the caveats section, some countires are almost half as likely to get misclassified, compared to others. Further investigation is required to understand the source of the bias.

## Caveats and Recommendations
As can be seen from the metrics, the model is generally accurate, with a few exceptions for the following feature/category pairs:  
Performance on slice relationship: Husband is 0.75  
Performance on slice relationship: Wife is 0.734  
Performance on slice native-country: Yugoslavia is 0.5  
Performance on slice native-country: Cambodia is 0.5  
Performance on slice native-country: Thailand is 0.5  
 
 
It is then recommended to use that model, but when it comes to the aforementioned weakpoints, perhaps a manual analysis would be better.  
