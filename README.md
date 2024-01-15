# Titanic Disaster Dataset

  

## Installation

To run the project, you'll need to have the following Python packages installed. You can install them using the following command:
  
```bash

pip install pandas matplotlib pycaret autoviz sweetviz shap FuzzyTM blosc2 wordcloud popular cuml

```

## About Dataset 

### Context

The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew. 
<p>
In this challenge, we build a predictive model that answers the question: <i>“what sorts of people were more likely to survive?”</i> using passenger data (ie name, age, gender, socio-economic class, etc).  While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. The data has been split into two groups: training set (train.csv) and test set (test.csv). The training set has the outcome of each passenger recorded and it should be used to build the machine learning models. In the test set, outcome of passengers is not provided so it serves as unseen data.
</p>

-  **Subject Area:** Disaster Investigation

-  **Associated Tasks:** Binary Classification

-  **Feature Type:** Categorical, Integer, Float

-  **Number of Records:** 1309

-  **Number of Features:** 11

  

### Features

| Variable | Definition                          | Key                                     |
|----------|-------------------------------------|-----------------------------------------|
| survival | Survival                            | 0 = No, 1 = Yes                         |
| pclass   | Ticket class                        | 1 = 1st, 2 = 2nd, 3 = 3rd               |
| sex      | Sex                                 |                                         |
| Age      | Age in years                        |                                         |
| sibsp    | # of siblings / spouses aboard     |                                         |
| parch    | # of parents / children aboard     |                                         |
| ticket   | Ticket number                       |                                         |
| fare     | Passenger fare                      |                                         |
| cabin    | Cabin number                        |                                         |
| embarked | Port of Embarkation                 | C = Cherbourg, Q = Queenstown, S = Southampton |
  

#### Variable Notes

**pclass:** A proxy for socio-economic status (SES)

- 1st = Upper

- 2nd = Middle

- 3rd = Lower

  

**age:** Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5 <br>

  

**sibsp:** The dataset defines family relations in this way...

- Sibling = brother, sister, stepbrother, stepsister

- Spouse = husband, wife (mistresses and fiancés were ignored)

  

**parch:** The dataset defines family relations in this way...

- Parent = mother, father

- Child = daughter, son, stepdaughter, stepson

- Some children traveled only with a nanny, therefore parch=0 for them.

 ## Data Exploration and Model Automation
Our main goal here is to dig into the data, get cozy with it, and make friends with some handy tools like Sweetviz, Autoviz, and PyCaret. We're on a mission to understand our dataset better and implement techniques to automate visualization with cool Python libraries like Sweetviz and Autoviz, while PyCaret takes care of the heavy lifting in building models.

### Sweetviz
Sweetviz is a Python library for Exploratory Data Analysis (EDA) that generates beautiful, high-density visualizations to quickly analyze and compare datasets. It provides insights into the distribution of features, relationships between variables, and more.
```bash
########## Usage Example ##########
import  sweetviz  as  sv

# Assuming you have a DataFrame called 'data'
report  =  sv.analyze(data)
report.show_html('sweetviz_report.html')
```

### Autoviz

Autoviz is another EDA tool that automatically visualizes the dataset with a single line of code. It's designed to handle large datasets and generates various plots to help us understand the data distribution, relationships, and outliers.
```bash
########## Usage Example ##########
from  autoviz.AutoViz_Class  import  AutoViz_Class

AV  =  AutoViz_Class()
report  =  AV.AutoViz('your_file_name.extension')
```

### PyCaret

PyCaret is a low-code machine learning library that automates the end-to-end machine learning workflow. It provides tools for automating feature engineering, model selection, hyperparameter tuning, and model deployment. PyCaret simplifies the machine learning process, making it easy for both beginners and experienced data scientists to build and deploy models efficiently.
```bash
########## Usage Example ##########
from  pycaret.classification  import  *

# Assuming you have a DataFrame called 'data', variable 'x' where the name of the target is stored and a DataFrame called 'test_data' which gives new unseen data.

# Set up the PyCaret environment for machine learning, handling data preprocessing, feature engineering, and other initial configurations to kickstart the model-building process
clf_setup  =  setup(data, target=x)

# Compare the performance of multiple machine learning models
best_model  =  compare_models()

# Initiate the training process for a specified machine learning model
xgb  =  create_model('xgboost')

# Hyperparameter Tuning of a trained machine learning model
tuned_model  =  tune_model(xgb)

# Generate predictions on unseen data using a pre-trained machine learning model
predictions  =  predict_model(data, data=test_data)

# Lock in the best-performing machine learning model
final_tuned_model  =  finalize_model(tuned_model)

# Save the trained machine learning model as a .pkl (pickle) file
save_model(final_tuned_model, 'saved_model')

# Load the .pkl (pickle) file
loaded_model = load_model('saved_model')

########## Additional Methods ##########

# Retrieve the current PyCaret configuration.
get_config('X_train')

# Assess the performance of a trained model using various plots and metrics.
evaluate_model(final_tuned_model)

# Provide interpretation plots for the trained model, such as feature importance plots. This function only supports tree based models for binary classification.
interpret_model(final_tuned_model)

# Visualize different aspects of the model, such as the learning curve or the feature interaction.
plot_model(final_tuned_model, plot='learning')

# Create an ensemble of models to potentially boost overall performance.
ensemble_model(final_tuned_model) 

# Blend multiple models to create an ensemble.
blend_models([tuned_model, 'rf', 'et'])

# Stack multiple models to create a composite model.
stack_models([tuned_model, 'rf', 'et'])

# Deploy the trained model on cloud platforms like AWS, Azure, or Google Cloud.
deploy_model(final_tuned_model, model_name='my_deployed_model', platform='aws', authentication={'bucket': 's3-bucket-name'})
```

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the [LICENSE](https://github.com/soumyadeepghoshGG/TItanic---Automated-Report-Gen-and-Training/blob/main/License.txt) file for details.

  

## Contact

For questions or issues, please contact me (Soumyadeep Ghosh) via mail: soumyadeepghosh57@gmail.com