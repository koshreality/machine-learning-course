import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics, tree
import pydotplus
from io import StringIO


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

    regrtree = tree.DecisionTreeRegressor(random_state=11)
    regrtree.fit(X, Y)
    ans = regrtree.predict(X_test)

    print('mse score: ', metrics.mean_squared_error(Y_test, ans))

    dotfile = StringIO()
    tree.export_graphviz(regrtree, out_file=dotfile, max_depth=5, feature_names=list(data.drop(['cnt'], axis=1).columns.values))
    graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
    graph.write_png("first.png")

    parameters = {"criterion": ["mse", "mae"],
                  "min_samples_split": [2, 5, 10, 15],
                  "max_depth": [None, 10],
                  "min_samples_leaf": [1, 2, 4, 8],
                  "max_leaf_nodes": [None, 50],
                  "max_features": [None, 8, 9, 10],
                  }
    cv = model_selection.GridSearchCV(regrtree, parameters, scoring='neg_mean_squared_error', cv=4)
    cv.fit(X, Y)

    print("best params: " + str(cv.best_params_))
    print("best mse score: " + str(-cv.best_score_))

    regrtree.set_params(**cv.best_params_)
    regrtree.fit(X, Y)

    dotfile = StringIO()
    tree.export_graphviz(regrtree, out_file=dotfile, max_depth=5, feature_names=list(data.drop(['cnt'], axis=1).columns.values))
    graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
    graph.write_png("second.png")
