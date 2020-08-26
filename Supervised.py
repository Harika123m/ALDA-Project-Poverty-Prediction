import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
from sklearn import tree
from keras.models import Sequential
from keras.layers import Dense

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tensorflow import set_random_seed
from sklearn.metrics import classification_report


# Applying random seed of 7 for getting the same results every time
np.random.seed(7)
set_random_seed(7)

# We directly use the normalized data for supervised learning
df = pd.read_csv('TrainNormalized.csv')

df_y = df['Target']
df_x = df.drop(['Target'], axis=1)

num_pcas = list(range(5, 41, 5))

classification_metrics = {

    'accuracy': {

        'stratified': {
            'decision_tree': [],
            'naive_bayes': [],
            'SVM': [],
            'neural_nets': [],
            'Random Forest': [],
            'Adaboost': [],
            'Logistic': [],
            'KNN': []
        },

        'cross-validation': {
            'decision_tree': [],
            'naive_bayes': [],
            'SVM': [],
            'neural_nets': [],
            'Random Forest': [],
            'Adaboost': [],
            'Logistic': [],
            'KNN': []
        }


    },

    'precision': {

        'stratified': {
            'decision_tree': [],
            'naive_bayes': [],
            'SVM': [],
            'neural_nets': [],
            'Random Forest': [],
            'Adaboost': [],
            'Logistic': [],
            'KNN': []
        },

        'cross-validation': {
            'decision_tree': [],
            'naive_bayes': [],
            'SVM': [],
            'neural_nets': [],
            'Random Forest': [],
            'Adaboost': [],
            'Logistic': [],
            'KNN': []
        }


    },

    'recall': {

        'stratified': {
            'decision_tree': [],
            'naive_bayes': [],
            'SVM': [],
            'neural_nets': [],
            'Random Forest': [],
            'Adaboost': [],
            'Logistic': [],
            'KNN': []
        },

        'cross-validation': {
            'decision_tree': [],
            'naive_bayes': [],
            'SVM': [],
            'neural_nets': [],
            'Random Forest': [],
            'Adaboost': [],
            'Logistic': [],
            'KNN': []
        }


    },

    'f1': {

        'stratified': {
            'decision_tree': [],
            'naive_bayes': [],
            'SVM': [],
            'neural_nets': [],
            'Random Forest': [],
            'Adaboost': [],
            'Logistic': [],
            'KNN': []
        },

        'cross-validation': {
            'decision_tree': [],
            'naive_bayes': [],
            'SVM': [],
            'neural_nets': [],
            'Random Forest': [],
            'Adaboost': [],
            'Logistic': [],
            'KNN': []
        }

    }
}

strategies = [i for i in classification_metrics.keys()]
classification_type = [i for i in classification_metrics[strategies[0]].keys()]
classifiers = [i for i in classification_metrics[strategies[0]][classification_type[0]].keys()]


def evaluate_classification_metrics(y_test, predicted, metric):
    if metric == 'accuracy':
        return accuracy_score(y_test, predicted)
    elif metric == 'precision':
        return precision_score(y_test, predicted, average='micro')
    elif metric == 'recall':
        return recall_score(y_test, predicted, average='micro')
    elif metric == 'f1':
        return f1_score(y_test, predicted, average='micro')


def evaluate_cross_validation_score(classifier, train, labels, metric):
    if metric != 'accuracy':
        metric += '_micro'
    return np.mean(cross_val_score(classifier, train, labels, cv=5, scoring=metric))


def classify_with_decision_tree(x_train, y_train, x_test,  y_test, train, labels):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    for _type in classification_metrics:
        classification_metrics[_type]['stratified']['decision_tree'].append(
            evaluate_classification_metrics(y_test, predicted, _type)
        )

    clf = tree.DecisionTreeClassifier()
    for _type in classification_metrics:
        classification_metrics[_type]['cross-validation']['decision_tree'].append(
            evaluate_cross_validation_score(clf, train, labels, _type)
        )
    print(classification_report(y_test, predicted))


def classify_with_naive_bayes(x_train, y_train, x_test,  y_test, train, labels):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    predicted = gnb.predict(x_test)
    for _type in classification_metrics:
        classification_metrics[_type]['stratified']['naive_bayes'].append(
            evaluate_classification_metrics(y_test, predicted, _type)
        )

    gnb = GaussianNB()
    for _type in classification_metrics:
        classification_metrics[_type]['cross-validation']['naive_bayes'].append(
            evaluate_cross_validation_score(gnb, train, labels, _type)
        )


def classify_with_svm(x_train, y_train, x_test,  y_test, train, labels):
    parameters = {'gamma': [0.1, 0.3, 0.4, 0.5, 1], 'C': [0.5, 1, 2, 3]}

    grid = GridSearchCV(SVC(), parameters, cv=5, scoring='accuracy')
    grid.fit(x_train, y_train)
    print(grid.best_params_)
    clf = SVC(C=grid.best_estimator_.C, kernel='rbf', gamma=grid.best_estimator_.gamma)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    for _type in classification_metrics:
        classification_metrics[_type]['stratified']['SVM'].append(
            evaluate_classification_metrics(y_test, predicted, _type)
        )

    svm = SVC(gamma='auto')
    for _type in classification_metrics:
        classification_metrics[_type]['cross-validation']['SVM'].append(
            evaluate_cross_validation_score(svm, train, labels, _type)
        )


def classify_with_neural_nets(x_train, y_train, x_test,  y_test, train, labels):
    hidden_neurons = [10, 30]
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values

    model = Sequential()
    for _units in hidden_neurons:
        model.add(Dense(units=_units, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, verbose=0)
    _score = model.evaluate(x_test, y_test, verbose=0)
    classification_metrics['accuracy']['stratified']['neural_nets'].append(_score[1])


def classify_with_random_forest(x_train, y_train, x_test, y_test, train, labels):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    for _type in classification_metrics:
        classification_metrics[_type]['stratified']['Random Forest'].append(
            evaluate_classification_metrics(y_test, predicted, _type)
        )
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    for _type in classification_metrics:
        classification_metrics[_type]['cross-validation']['Random Forest'].append(
            evaluate_cross_validation_score(clf, train, labels, _type)
        )


def classify_with_adaboost(x_train, y_train, x_test, y_test, train, labels):
    clf = AdaBoostClassifier(n_estimators=50, random_state=0)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    for _type in classification_metrics:
        classification_metrics[_type]['stratified']['Adaboost'].append(
            evaluate_classification_metrics(y_test, predicted, _type)
        )
    clf = AdaBoostClassifier(n_estimators=50, random_state=0)
    for _type in classification_metrics:
        classification_metrics[_type]['cross-validation']['Adaboost'].append(
            evaluate_cross_validation_score(clf, train, labels, _type)
        )


def classify_with_logistic_regression(x_train, y_train, x_test, y_test, train, labels):
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial')
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    for _type in classification_metrics:
        classification_metrics[_type]['stratified']['Logistic'].append(
            evaluate_classification_metrics(y_test, predicted, _type)
        )
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial')
    for _type in classification_metrics:
        classification_metrics[_type]['cross-validation']['Logistic'].append(
            evaluate_cross_validation_score(clf, train, labels, _type)
        )


def classify_with_knn(x_train, y_train, x_test, y_test, train, labels):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    for _type in classification_metrics:
        classification_metrics[_type]['stratified']['KNN'].append(
            evaluate_classification_metrics(y_test, predicted, _type)
        )
    clf = KNeighborsClassifier(n_neighbors=3)
    for _type in classification_metrics:
        classification_metrics[_type]['cross-validation']['KNN'].append(
            evaluate_cross_validation_score(clf, train, labels, _type)
        )


def classify_data(df_x, df_y):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, stratify=df_y, random_state=42)
    classify_with_decision_tree(x_train, y_train, x_test, y_test, df_x, df_y)
    classify_with_naive_bayes(x_train, y_train, x_test, y_test, df_x, df_y)
    classify_with_svm(x_train, y_train, x_test, y_test, df_x, df_y)
    classify_with_neural_nets(x_train, y_train, x_test, y_test, df_x, df_y)
    classify_with_random_forest(x_train, y_train, x_test, y_test, df_x, df_y)
    classify_with_adaboost(x_train, y_train, x_test, y_test, df_x, df_y)
    classify_with_logistic_regression(x_train, y_train, x_test, y_test, df_x, df_y)
    classify_with_knn(x_train, y_train, x_test, y_test, df_x, df_y)


def main():

    classify_data(df_x, df_y)
    for i in num_pcas:
        pca = PCA(n_components=i)
        train = pd.DataFrame(pca.fit_transform(df_x))
        classify_data(train, df_y)

    for _classification_type in classification_type:
        for classifier in classifiers:
            print('{} {}'.format(classifier, _classification_type))
            for _type in strategies:
                _val = classification_metrics[_type][_classification_type][classifier]
                print("{} : [{}] : (pca, accuracy_score)".format(_type, list(zip(['Normalized']+num_pcas, _val))))
            print("\n")


if __name__ == '__main__':
    main()
