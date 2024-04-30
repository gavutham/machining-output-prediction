from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def predict(x, y):
    dt_model = DecisionTreeRegressor(random_state=42)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=1)

    dt_model.fit(xtrain, ytrain)
    predicted = dt_model.predict(xtest)

    print("Decision Tree Model\n")
    print("actual")
    print(ytest)
    print("predicted")
    for e in predicted:
        print(list(e))

    print(f"MSE: {mean_squared_error(ytest, predicted)}")
    print(f"R2 : {r2_score(ytest, predicted)}")
