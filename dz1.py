import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = open('semeion.data', 'r')

    imgs = []
    digits = []
    for line in data:
        imgs.append([int(i[0]) for i in line.split()[0:256]])
        for i in range(10):
            if line.split()[256+i] == '1':
                digits.append(i)
                break

    data.close()

    X, X_test, Y, Y_test = train_test_split(imgs, digits, test_size=0.4, random_state=11)

    clf = KNeighborsClassifier(n_neighbors=2, n_jobs=4).fit(X, Y)
    f1s = f1_score(y_true=Y_test, y_pred=clf.predict(X_test), average='micro')
    print("f1_micro score: " + str(f1s))

    cv = GridSearchCV(clf,
                      {'n_neighbors': list(range(4, 15)),
                       'metric': ['euclidean', 'manhattan'],
                       'weights': ['uniform', 'distance']},
                      scoring='f1_micro', cv=3, n_jobs=4)
    cv.fit(X, Y)

    print("best params: " + str(cv.best_params_))
    print("best score: " + str(cv.best_score_))

    img2d = np.zeros([16, 16])
    for i in range(16):
        img2d[i, :] = imgs[666][(i * 16):(i * 16 + 16)]
    plt.imshow(img2d, cmap='gray')
    plt.show()
