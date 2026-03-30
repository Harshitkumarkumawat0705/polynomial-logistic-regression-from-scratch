import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


# Set seed
np.random.seed(42)

# Generate a classification dataset
x, y = make_classification(
    n_samples=200,
    n_features=2,          # Only 2 useful features for visualization
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,         # Low separation makes it harder
    flip_y=0.1,            # Add noise (10% label flipping)
    random_state=42
)

def polynimial_features(x):
    x1=x[:,[0]]
    x2=x[:,[1]]
    return np.hstack([
        x1,
        x2,
        x1*x1,
        x2*x2,
        x1*x2,
        x1**3,
        x2**3
    ])

mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
x = (x - mean) / std

x_poly=polynimial_features(x)
y = np.reshape(y, (-1, 1))

def decision_boundary(w,x,b):
    return np.dot(x,w)+b
def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g
def loss_function(y,w,g,lamdha):
    esp=1e-8
    m=y.shape[0]
    L= -np.mean(y*np.log(g+esp)+(1-y)*np.log(1-g+esp))
    regularization=((lamdha/(2*m))*np.sum(w**2))
    return L+regularization
def DW(g, y, x,lamdha,w):
    m = x.shape[0]
    return (1/m) * np.dot(x.T, (g - y))+((lamdha/m)*w)
def DB(g, y):
    return np.mean(g - y)
w=np.zeros((x_poly.shape[1],1))
b=0
alpha=0.01
lamdha=100
all_loss=[]
for i in range(1000):
    z=decision_boundary(w,x_poly,b)
    Y=sigmoid(z)
    all_loss.append(loss_function(y,w,Y,lamdha))
    w=w-alpha*DW(Y,y,x_poly,lamdha,w)
    b=b-alpha*DB(Y,y) 

# Plot
print(w, b)
# Create a grid
x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)
# Stack grid points
grid_points = np.c_[xx1.ravel(), xx2.ravel()]

# Convert to polynomial features
grid_poly = polynimial_features(grid_points)
z_grid = decision_boundary(w, grid_poly, b)
prob_grid = sigmoid(z_grid)
prob_grid = prob_grid.reshape(xx1.shape)
plt.figure(figsize=(8, 6))

# Plot data points
y=y.ravel()
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1],
            color="red", label="Class 0", alpha=0.6)

plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1],
            color="blue", label="Class 1", alpha=0.6)

# Plot decision boundary (P = 0.5)
plt.contour(xx1, xx2, prob_grid, levels=[0.5], colors="black")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary (Polynomial Logistic Regression)")
plt.legend()
plt.grid(True)
plt.show()