# Fake-News-Detection

This project explores the use of supervised learning models to detect fake news with a 7796×4 dataset (source: Data Flair). The dataset can be viewed here: https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view

## News.csv format:
Column 1 | Column 2 | Column 3 | Column 4 
   ID    |   Title  |   Text   |   Label ("Real"/"Fake")

## Outputs
### Accuracy of 5-fold Cross Validated Logisitic Regression
![logit_fakenews_barplot](./outputs/logit_fakenews_barplot.png)
![logit_fakenews_output](./outputs/logit_fakenews_output.png)

### Accuracy of 5-fold Cross Validated Passive Aggressive Classifer
![pac_fakenews_barplot](./outputs/pac_fakenews_barplot.png)
![pac_fakenews_output](./outputs/pac_fakenews_output.png)

### Train Test Split on Logit and PAC
![logit_fakenews_single_test_output](./outputs/logit_fakenews_single_test_output.png)
![pac_fakenews_single_test_output](./outputs/pac_fakenews_single_test_output.png)

## Improvements
Some potential next steps include:
* improve data quality for learning 
* test additional models for performance
* introduce additional attributes/factor columns 
