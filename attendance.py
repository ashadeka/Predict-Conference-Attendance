import pandas as pd
import numpy.random as rnd
from sklearn.ensemble import RandomForestRegressor 
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np
from pylab import *

rnd.seed(0)

df = pd.read_csv("final_members.csv", index_col = [0])

# The machine learning 
predicting = df[['adjusted fee', 
                            'average_distance','Desirability', 'Attendance']]
selection = rnd.binomial(1, 0.6, size=len(predicting)).astype(bool)
training = predicting[selection]
testing = predicting[~selection]
rfr = RandomForestRegressor()
predictors = ['adjusted fee','average_distance', 'Desirability']
rfr.fit(training[predictors], training['Attendance'])
test_prediction = rfr.predict(testing[predictors])
trained_prediction = rfr.predict(training[predictors])


train_errors  = (trained_prediction-training["Attendance"])/training["Attendance"]
train_accuracy = train_errors.abs().mean()
print("training error: %.3f"%train_accuracy)


errors  = (test_prediction-testing["Attendance"])/testing["Attendance"]
test_accuracy = errors.abs().mean()
print("testing error: %.3f"%test_accuracy)

df1 = pd.DataFrame(rfr.feature_importances_,predictors)
df1.columns = ["Feature Importance"]
print(df1)
atlanta = rfr.predict([425,df.ix["Atlanta"]["average_distance"],df.ix["Atlanta"]["Desirability"]])
print("Atlanta Attendance:", int(round(atlanta[0])))

olm = lm.LinearRegression()

plt.style.use("ggplot")
plt.figure()
plt.title("Actual Attendance vs Predicted Attendance")
plt.xlabel("Actual Attendance")
plt.ylabel("Predicted Attendance")
pr = list(test_prediction)
testing["predicted"] = pr
plt.scatter(testing["predicted"],testing["Attendance"])
m,b = np.polyfit(testing["predicted"],testing["Attendance"],1)

yp = polyval([m,b], testing["Attendance"])
plt.plot(testing["Attendance"],yp)
plt.savefig("test.png")

"""
attendance = testing["Attendance"].values
olm.fit(list(test_prediction), list(attendance))
plt.plot(testing["predicted"], olm.predict(testing[["Attendance"]].values))

"""
plt.figure()
train_pr = list(trained_prediction)
training["predicted"] = train_pr
plt.scatter(training["predicted"],training["Attendance"])
n,a = np.polyfit(training["predicted"],training["Attendance"],1)

ya = polyval([n,a], training["Attendance"])
plt.plot(training["Attendance"],ya)

testing.to_csv("testing.csv")
#PREDICTED VALUES VS ACTUAL ATTENDANCE for testing df

