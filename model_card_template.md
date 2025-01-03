# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a RandomForestClassifier trained to predict whether a person's income exceeds $50,000 annually based on demographic and work-related attributes. 
It uses a combination of categorical and numerical features and was trained with the following hyperparameters:
- `n_estimators`: 100
- `max_depth`: 10
- `random_state`: 42

## Intended Use
The model is intended to assist in demographic analysis and income prediction tasks for educational or exploratory purposes. 
It should not be used for important decisions, like hiring or loans, without further testing.

## Training Data
The model was trained on census data consisting of demographic, work-related, and geographic features. 
The training dataset includes 80% of the overall data split. Features used include:
- Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country.
- Numerical: age, hours-per-week, education-num.

## Evaluation Data
The test dataset consists of 20% of the original census data. It is balanced to include all unique categories in the categorical features and the full range of numerical values.

## Metrics
The model was evaluated using precision, recall, and F1 score. 
Overall metrics on the test set are:
- Precision: 0.7962
- Recall: 0.5372
- F1 Score: 0.6416

## Ethical Considerations
The model might be biased if some demographic groups are underrepresented in the data.
It should not be used for important decisions like hiring or loans without more testing to ensure fairness.
The data reflects existing societal biases, such as differences in workclass or native-country.

## Caveats and Recommendations
The model depends on the quality and accuracy of the census data.
It shouldn’t be used for critical decisions without further testing and fairness checks.
Fine-tuning the model and improving the features could boost performance.
Suggestions:
Test the model on real-world data before using it.
Retrain the model regularly to keep it up-to-date with demographic changes.
