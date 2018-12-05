import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def plot_corr(df, size=8):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)


if __name__ == '__main__':
    data = pd.read_csv('day.csv', index_col='instant')
    print(data.head(2))
    plot_corr(data)
    plt.show()
    data = data.drop(['registered', 'dteday'], axis=1)

    X, X_test, Y, Y_test = train_test_split(
        data.drop(['cnt'], axis=1), data['cnt'],
        test_size=0.1, random_state=11)

    scaler = StandardScaler()
    scaler.fit(X, Y)
    X_scaled = scaler.transform(X)
    X_scaled_test = scaler.transform(X_test)

    regrsgd = linear_model.SGDRegressor(random_state=11)
    regrsgd.fit(X_scaled, Y)
    ans = regrsgd.predict(X_scaled_test)

    print('mse score (with categorical): ', metrics.mean_squared_error(Y_test, ans))

    X_scaled_without_cat = X_scaled[:, 7:]
    X_scaled_without_cat_test = X_scaled_test[:, 7:]

    regrsgd.fit(X_scaled_without_cat, Y)
    ans = regrsgd.predict(X_scaled_without_cat_test)

    print('mse score (without categorical): ', metrics.mean_squared_error(Y_test, ans))

    one_hot_encoder = OneHotEncoder(sparse=False, categorical_features=range(0, 7))

    data_one_hot_encoded = one_hot_encoder.fit_transform(data.drop(['cnt'], axis=1))

    X_one_hot, X_one_hot_test, Y, Y_test = train_test_split(
        data_one_hot_encoded, data['cnt'],
        test_size=0.1, random_state=11)

    scaler.fit(X_one_hot, Y)
    X_one_hot_scaled = scaler.transform(X_one_hot)
    X_one_hot_scaled_test = scaler.transform(X_one_hot_test)

    regrsgd.fit(X_one_hot_scaled, Y)
    ans = regrsgd.predict(X_one_hot_scaled_test)

    print('mse score (with one hot encoded categorical): ', metrics.mean_squared_error(Y_test, ans))

    data_one_hot_encoded_without_atemp = one_hot_encoder.fit_transform(data.drop(['cnt', 'atemp'], axis=1))

    X_one_hot, X_one_hot_test, Y, Y_test = train_test_split(
        data_one_hot_encoded_without_atemp, data['cnt'],
        test_size=0.1, random_state=11)

    scaler.fit(X_one_hot, Y)
    X_one_hot_scaled = scaler.transform(X_one_hot)
    X_one_hot_scaled_test = scaler.transform(X_one_hot_test)

    regrsgd.fit(X_one_hot_scaled, Y)
    ans = regrsgd.predict(X_one_hot_scaled_test)

    print('mse score (with one hot encoded categorical and without atemp): ', metrics.mean_squared_error(Y_test, ans))
