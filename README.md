# Identifying hate crime in online media 
## Members:
### Ahsan Ejaz
### Jashanpreet Kaur Gill
### Justin Collier


# Work Distribution:

## Ahsan Ejaz
### * ...

## Jashanpreet Kaur Gill
* Added tests for model precision score
* Added tests for ROC AUC score
* Created CI/CD pipeline specific to Jashan branch
* Integrated precision score and ROC AUC score tests into the CI/CD pipeline
* Started working on the project report to be submitted.

## Justin Collier:
### * ...

 
## Changelog
### July, 24. 2024 - Ahsan
### Initial Commit / Repo Invite
+ Initialized the project repo & readme file

### July, 27. 2024 - Ahsan
### Create CI-CD.yml based upon the workflow
+ created an initial version of our workflow. The logic behind is that I want to deploy CI CD PIpeline using github actions in such a way that when i push a new code or run a new model instead of a logistic regression model it runs smoothly. If i run a bad model or push a bad code it doesn't allow me to commit.

### July, 27. 2024 - Justin
### Flask App: 'Twitter' Frontend, Hard Coded Censoring, Simple Page Navigation
+ Implemented the Flask library to handle our simple front-end/back-end interactions.
+ Created a Twitter-esque UI to handle our raw text input/output
+ Hard coded a simple chat-filter as a placeholder that censors pre-defined keywords from the user's tweet.

### July, 27. 2024 - Justin
### Flask App: Censored Word Log, Front-End CSS Polish
+ Added tracking and logging for censored words as they are detected.
+ Added a display section under the filtered tweet that shows what words were censored and their word count.
+ Modified styles.css to give the web page a more twitter-like appearance

### July, 27. 2024 - Ahsan
###  tested logistic regression model to classify hate speech

### July, 30. 2024 - Ahsan
### Update ci-cd.yml

### July, 30. 2024 - Justin
### Hugging Face - Hate Speech Model
+ Added the transformers library and used it to implement the pre-trained unitary/toxic-bert model from Hugging Face.
+ Altered the hate_speech_detection function to use this model for classification.
+ Finalized the MVP for the project & its features

### August, 1. 2024 - Ahsan
### dded a bad model of linear regression

### August, 5. 2024 - Jashanpreet Kaur Gill
+ Added precision score test Implemented a test to check the precision score of the model

### August, 5. 2024 - Jashanpreet Kaur Gill
+ Added ROC AUC score test - Implemented a test to check the ROC AUC score of the model

### August, 5. 2024 - Jashanpreet Kaur Gill
+ Created CI/CD pipeline specific to Jashan branch - Developed a CI/CD pipeline specific to the Jashan branch to ensure the integration of new tests

### August, 5. 2024 - Jashanpreet Kaur Gill
+ Integrated precision score and ROC AUC score tests into the CI/CD pipeline - Updated the CI/CD pipeline to include the newly added precision score and ROC AUC score tests

# Data Sources/References: 
## * unitary/toxic-bert model from huggingface
### Source: https://huggingface.co/unitary/toxic-bert
