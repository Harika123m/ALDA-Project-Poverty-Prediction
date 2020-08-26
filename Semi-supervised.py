import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from seaborn import countplot


def main():
    data_train = pd.read_csv('TrainNormalized.csv')
    data_test = pd.read_csv('TestNormalized.csv')

    print('length of train data', len(data_train))
    print('length of test data', len(data_test))

    # Use mode :Pseudo labeling using  random forests for semi-supervised learning
    # we use Random Forests since it was one of the model which yielded the best results 
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=23, min_samples_split=5)
    X_train = data_train.drop(['Target'], axis=1)
    y_train = data_train['Target']

    print(data_train['Target'].value_counts())

    clf.fit(X_train, y_train)
    pred = clf.predict(data_test)

    pred = pd.DataFrame(pred)
    print(pred[0].value_counts())

    data_test['Target'] = pred

    big_data = pd.concat([data_train, data_test])

    print(big_data.shape)

    big_data_feat = big_data.drop(['Target'], axis=1)
    big_data_label = big_data['Target']

    # fit a Random forst model using data from both train and etst set.
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=23, min_samples_split=5)

    clf.fit(big_data_feat, big_data_label)
    data_test_target = data_test['Target']
    data_test = data_test.drop(['Target'], axis=1)
    test_predictions = pd.DataFrame(clf.predict(data_test))
    target_distribution = test_predictions[0].value_counts()

    # Let's evaluate the accuracy to see how consistent the semi-supervised model is
    accuracy_score_result = accuracy_score(data_test_target, test_predictions)
    print('The Accuracy score using Semi supervised model {}'.format(accuracy_score_result))
    if accuracy_score_result > 0.95:
        print('The Accuracy score is consistent')

    test_predictions.columns = ['Target']
    ax = countplot(x='Target', data=test_predictions)
    fig = ax.get_figure()
    fig.savefig('fig/semi_supervised_distribution.png')


if __name__ == '__main__':
    main()
