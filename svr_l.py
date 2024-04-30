from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import pprint


def predict(x, y):
    svr = SVR(kernel="rbf")
    multioutput_regressor = MultiOutputRegressor(svr)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=1)

    multioutput_regressor.fit(xtrain, ytrain)

    predicted = multioutput_regressor.predict(xtest)
    print("SVR Model\n")
    print("actual")
    pprint.pprint(ytest)
    print("predicted")
    for e in predicted:
        pprint.pprint(list(e))

    print(f"MSE: {mean_squared_error(ytest, predicted)}")
    print(f"R2 : {r2_score(ytest, predicted)}")

    return








