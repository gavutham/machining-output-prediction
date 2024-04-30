
from svr_l import predict
# from rf import predict
import pandas
# from decision_tree import predict

data = pandas.read_csv("data.csv", delimiter=" ")

x = data.iloc[:, 0:3]
y = data.iloc[:, 4:]

predict(x, y)

