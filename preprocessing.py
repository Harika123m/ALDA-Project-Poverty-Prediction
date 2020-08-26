import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# setting warnings off
pd.options.mode.chained_assignment = None


def main():
    df_train = pd.read_csv('TrainHouseHold.csv')
    df_test = pd.read_csv('TestHouseHold.csv')
    columns_with_labels = df_train.select_dtypes('object').columns.tolist()
    for col in columns_with_labels:
        data = pd.concat([df_train[col], df_test[col]])
        lb_make = LabelEncoder()
        lb_make.fit(data)
        df_train[col] = lb_make.transform(df_train[col])
        df_test[col] = lb_make.transform(df_test[col])

    df_y = df_train['Target']
    df_train = df_train.drop(['Target'], axis=1)
    df_test = df_test.drop(['Target'], axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()

    print('train :', df_train.shape)
    print('test :', df_test.shape)
    min_max_scaler.fit(df_train)
    df_tr = min_max_scaler.transform(df_train)
    df_te = min_max_scaler.transform(df_test)

    train_data = pd.DataFrame(df_tr, columns=df_train.columns)
    train_data['Target'] = df_y
    test_data=pd.DataFrame(df_te, columns=df_test.columns)
    test_data.to_csv('TestNormalized.csv', index=False)
    train_data.to_csv('TrainNormalized.csv', index=False)


if __name__ == '__main__':
    main()
