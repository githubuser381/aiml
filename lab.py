1a
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/headbrain.csv')
df.head()

X = df['Head Size(cm^3)'].values
Y = df['Brain Weight(grams)'].values

n = len(X)
mean_X = np.mean(X)
mean_Y = np.mean(Y)

print(f"Mean of X: {mean_X}")
print(f"Mean of Y: {mean_Y}")

numer = 0
denom = 0

for i in range(n):
    numer += (X[i] - mean_X) * (Y[i] - mean_Y)
    denom += (X[i] - mean_X) ** 2
    
b1 = numer / denom
b0 = mean_Y - (b1 * mean_X)

print(f"b1: {b1}")
print(f"b0: {b0}")

max_x = np.max(X) + 100
min_x = np.min(X) - 100

x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

plt.plot(x, y, color="red", label="Regression Line")
plt.scatter(X, Y, color="blue", label="Scatter Plot")
plt.xlabel("Head")
plt.ylabel("Brain")
plt.legend()
plt.show()

ss_tot = 0
ss_res = 0

for i in range(n):
    y_pred = b0 + b1 * X[i]
    ss_res += (Y[i] - y_pred) ** 2
    ss_tot += (Y[i] - mean_Y) ** 2

r2 = 1 - (ss_res / ss_tot)
print(f"R2 score: {r2}")

1b
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/housing_prices_SLR.csv')
df.head()

plt.scatter(df['AREA'], df['PRICE'], c="blue")
plt.show()
plt.scatter(df['AREA'], df['PRICE'], c=np.random.random(df.shape[0]))
plt.show()

X = df[['AREA']].values
y = df['PRICE'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test: {y_test.shape}")
X[:5]
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print(f"Intercept: {lr_model.intercept_}")
print(f"Coefficientf: {lr_model.coef_}")
from sklearn.metrics import r2_score
r2_score(y_train, lr_model.predict(X_train))
r2_score(y_test, lr_model.predict(X_test))
plt.scatter(X_train[:, 0], y_train, c='r')
plt.scatter(X_test[:, 0], y_test, c='b')
plt.plot(X_train[:, 0], lr_model.predict(X_train), c='g')
plt.show()

2a
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('datasets/student.csv')
df.head()
math = df['Math'].values
read = df['Reading'].values
write = df['Writing'].values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(math, read, write, c='r')
plt.show()

m = len(math)
x0 = np.ones(m)
X = np.array([x0, math, read]).T

B = np.array([0, 0, 0])
Y = np.array(write)
alpha = 0.0001

def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2) / (2 * m)
    return J

initial_cost = cost_function(X, Y, B)
print(f"Initial Cost: {initial_cost}")

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        h = X.dot(B)
        loss = h - Y
        gradient = X.T.dot(loss) / m
        B = B - alpha * gradient
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
    
    return B, cost_history

newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)
print(f"New coefficients: {newB}")
print(f"Final cost: {cost_history[-1]}")

def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Y_pred = X.dot(newB)

print(f"RMSE: {rmse(Y, Y_pred)}")
print(f"R2 Score: {r2_score(Y, Y_pred)}")


2b
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('datasets/housing_prices.csv')
df.head()
df = df.drop(['CODE'], axis=1)
df.head()
x = df.iloc[:, :3].values
y = df.iloc[:, 3].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
from sklearn.linear_model import LinearRegression
mlr_model = LinearRegression()
mlr_model.fit(x_train, y_train)
print(f"Training score: {mlr_model.score(x_train, y_train)}")
print(f"Testing score: {mlr_model.score(x_test, y_test)}")

3a
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('breast_cancer.csv')
df.head()
df = df.iloc[:, :-1]
df.head()
x = df.iloc[:, 2:].values
y = df['diagnosis'].values
x[:2]
y[:5]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train, y_train)
predictions = dt_classifier.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(f"Training accuracy: {accuracy_score(y_train, dt_classifier.predict(x_train))}")
print(f"Testing accuracy: {accuracy_score(y_test, dt_classifier.predict(x_test))}")
print("Training confusion matrix: \n", confusion_matrix(y_train, dt_classifier.predict(x_train)))
print("Testing confusion matrix: \n", confusion_matrix(y_test, dt_classifier.predict(x_test)))
print("Classification report: \n", classification_report(y_test, dt_classifier.predict(x_test)))

3b
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('breast_cancer.csv')
df.head()
df = df.iloc[:, :-1]
df.head()
x = df.iloc[:, 2:].values
y = df['diagnosis'].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
predictions = nb_model.predict(x_test)
from sklearn.metrics import confusion_matrix
print("Training confusion matrix: \n", confusion_matrix(y_train, nb_model.predict(x_train)))
print("Testing confusion matrix: \n", confusion_matrix(y_test, nb_model.predict(x_test)))
print(f"Train score: {nb_model.score(x_train, y_train)}")
print(f"Test score: {nb_model.score(x_test, y_test)}")


4a
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('datasets/ch1ex1.csv')
df.head()
points = df.values
points[:5]
from sklearn.cluster import KMeans

model=KMeans
model = KMeans(n_clusters=3, n_init=10)
model.fit(points)
labels = model.predict(points)

xs = points[:, 0]
ys = points[:, 1]
centroids = model.cluster_centers_
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

plt.scatter(xs, ys, c=labels)
plt.scatter(centroids_x, centroids_y, marker='X', s=200)
plt.show()

4b
import pandas as pd
df = pd.read_csv('datasets/seeds-less-rows.csv')
df.head()
varieties = list(df.pop('grain_variety'))
samples = df.values
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

mergings = linkage(samples, method='complete')

dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=8,
)
plt.show()
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 6, criterion='distance')
df_new = pd.DataFrame({'labels': labels, 'varieties': varieties})
ct = pd.crosstab(df_new['labels'], df_new['varieties'])
ct


5a
import numpy as np
from functools import reduce
plt.scatter(xs, ys, c=labels)
plt.show()

def perceptron(weight, bias, x):
    model = np.add(np.dot(x, weight), bias)
    print('model: {}'.format(model))
    logit = 1/(1+np.exp(-model))
    print('Type: {}'.format(logit))
    return np.round(logit)

def compute(logictype, weightdict, dataset):
    weights = np.array([ weightdict[logictype][w] for w in weightdict[logictype].keys()])
    output = np.array([ perceptron(weights, weightdict['bias'][logictype], val) for val in dataset])
    print(logictype)
    return logictype, output

def main():
    logic = {
        'logic_and' : {
            'w0': -0.1,
            'w1': 0.2,
            'w2': 0.2
        },
        'logic_or': {
            'w0': -0.1,
            'w1': 0.7,
            'w2': 0.7
        },
        'logic_not': {
            'w0': 0.5,
            'w1': -0.7
        },
        'logic_nand': {
            'w0': 0.6,
            'w1': -0.8,
            'w2': -0.8
        },
        'logic_nor': {
            'w0': 0.5,
            'w1': -0.7,
            'w2': -0.7
        },
        'logic_xor': {
            'w0': -5,
            'w1': 20,
            'w2': 10
        },
        'logic_xnor': {
            'w0': -5,
            'w1': 20,
            'w2': 10
        },
        'bias': {
            'logic_and': -0.2,
            'logic_or': -0.1,
            'logic_not': 0.1,
            'logic_xor': 1,
            'logic_xnor': 1,
            'logic_nand': 0.3,
            'logic_nor': 0.1
        }
    }
    dataset = np.array([
        [1,0,0],
        [1,0,1],
        [1,1,0],
        [1,1,1]
    ])

    logic_and = compute('logic_and', logic, dataset)
#     logic_or = compute('logic_or', logic, dataset)
#     logic_not = compute('logic_not', logic, [[1,0],[1,1]])
    logic_nand = compute('logic_nand', logic, dataset)
#     logic_nor = compute('logic_nor', logic, dataset)
    # logic_xor = compute('logic_xor', logic, dataset)
    # logic_xnor = compute('logic_xnor', logic, dataset)

    def template(dataset, name, data):
        # act = name[6:]
        print("Logic Function: {}".format(name[6:].upper()))
        print("X0\tX1\tX2\tY")
        toPrint = ["{1}\t{2}\t{3}\t{0}".format(output, *datas) for datas, output in zip(dataset, data)]
        for i in toPrint:
            print(i)

    gates = [logic_and, logic_nand]

    for i in gates:
        template(dataset, *i)
5b
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv.imread('people.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
