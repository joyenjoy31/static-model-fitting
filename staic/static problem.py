import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel("sample data (1).xlsx")

df["ORDERDATE(x)"] = pd.to_datetime(df["ORDERDATE(x)"])
df["X"] = np.arange(1, len(df) + 1)  

df["Y"] = pd.to_numeric(df["QUANTITYORDERED(y)"], errors="coerce")

df = df.dropna(subset=["X", "Y"])

x = df["X"].values
y = df["Y"].values
n = len(df)


sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x**2)
sum_y2 = np.sum(y**2)
sum_xy = np.sum(x * y)

print("Predictors:")
print("Σx =", sum_x)
print("Σy =", sum_y)
print("Σx² =", sum_x2)
print("Σy² =", sum_y2)
print("Σxy =", sum_xy)

cov_xy = np.cov(x, y, bias=True)[0][1]
print("\nCovariance:", cov_xy)


std_x = np.std(x)
std_y = np.std(y)

print("\nStandard Deviation:")
print("SD(X) =", std_x)
print("SD(Y) =", std_y)


b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
a = np.mean(y) - b * np.mean(x)

print("\nRegression Equation:")
print(f"Y = {a:.2f} + {b:.2f}X")


plt.figure()
plt.scatter(x, y)
plt.plot(x, a + b*x)
plt.xlabel("Month Index (X)")
plt.ylabel("Quantity Ordered (Y)")
plt.title("Sales Regression Analysis")
plt.show()

residuals = y - (a + b*x)
outlier = np.argmax(np.abs(residuals))

print("\nOutlier Impact:")
print("Worst month index:", x[outlier])
print("Deviation:", residuals[outlier])

x_dec_2005 = x[-1] + 1
y_pred_dec = a + b * x_dec_2005

print("\nPredicted Quantity Order for Dec 2005:", round(y_pred_dec, 2))

df["Predicted_Y"] = a + b * df["X"]
df["Error"] = df["Y"] - df["Predicted_Y"]

print("\nMonthly Analysis (First 5 Rows):")
print(df[["ORDERDATE(x)", "Y", "Predicted_Y", "Error"]].head())
