# Poverty-Prediction

This project was done as a part of [CSC-522 Automated Learning and Design Analysis.](https://www.engineeringonline.ncsu.edu/course/csc-522-automated-learning-and-data-analysis-2/)
The project was done by 

* Harika Malapaka 
* Krithika Sekhar
* Jagadeesh Saravanan
* Ragavendran Balakrishnan


## Project Overview
Many social programs have a hard time making sure the right people are given enough aid. It’s especially tricky when a program focuses on the poorest segment of the population. The world’s poorest typically can’t provide the necessary income and expense records to prove that they qualify.

In Latin America, one popular method uses an algorithm to verify income qualification. It’s called the Proxy Means Test (or PMT). With PMT, agencies use a model that considers a family’s observable household attributes like the material of their walls and ceiling, or the assets found in the home to classify them and predict their level of need.

While this is an improvement, accuracy remains a problem as the region’s population grows and poverty declines.

To improve on PMT, the IDB (the largest source of development financing for Latin America and the Caribbean) has turned to the Kaggle community. They believe that new methods beyond traditional econometrics, based on a dataset of Costa Rican household characteristics, might help improve PMT’s performance.

Beyond Costa Rica, many countries face this same problem of inaccurately assessing social need. If Kagglers can generate an improvement, the new algorithm could be implemented in other countries around the world.


## Data
The data for this project was taken from kaggle competition of topic [Costa Rican Household Poverty Level Prediction
](https://www.kaggle.com/c/costa-rican-household-poverty-prediction). The Data consists of two files that are present in the folder data.

* [train.csv](data/train.csv)
* [test.csv](data/test.csv)

## Executing the Scripts

Executing the project requires executing the scripts in the below order

| Order | Script name | Description |
|-------|-------------|-------------|
| 1     | clean_dimensions.py | This script reads the raw training and testing data and reduces the unnecessary dimensions for further steps.|
| 2     | preprocessing.py    | This script encodes object values into integers and normalize the entire data so that the data can be fit into machine learning models |
| 3     | Supervised.py | This script reads the normalized data and computes the PCA for the data. The PCA along with the normalized data is used in various supervised models and the metrics are evaluated. |
| 4     | Unsupervised.py | This script reads the normalized data and does k-means clustering |
| 5     | Semi-Supervised.py | This script reads the normalized data with both training and test and gives the distribution of the target variable using the the best model from supervised.|

Once all the scripts are run, we can compare the efficiencies of each model and find the best one out.

After each run, there are some intermediate csvs that are generated, for use by other scripts.

Also, Pngs of some reports gets generated, which are saved in the figs folder.
