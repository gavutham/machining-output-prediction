from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def predict(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=1)

    rf_model = RandomForestRegressor(n_estimators=100)


    rf_model.fit(xtrain, ytrain)

    predicted = rf_model.predict(xtest)

    print("Random Forest Model\n")
    print("actual")
    print(ytest)
    print("predicted")
    for e in predicted:
        print(list(e))

    print(f"MSE: {mean_squared_error(ytest, predicted)}")
    print(f"R2 : {r2_score(ytest, predicted)}")
