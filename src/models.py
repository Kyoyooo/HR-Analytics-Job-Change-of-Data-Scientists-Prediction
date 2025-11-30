import numpy as np

class LogisticRegressionNumPy:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []
        
    def _sigmoid(self, z):
        # Clip z để tránh tràn số (overflow) với hàm exp
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.num_iterations):
            # 1. Linear model
            linear_pred = np.dot(X, self.weights) + self.bias
            # 2. Activation
            y_pred = self._sigmoid(linear_pred)
            
            # 3. Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 4. Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # 5. Calculate Loss (Binary Cross Entropy)
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = - (1/n_samples) * np.sum(y * np.log(y_pred_clipped) + (1-y) * np.log(1-y_pred_clipped))
            self.losses.append(loss)
            
            if i % 1000 == 0:
                print(f"Iteration {i}: Loss {loss:.4f}")
                
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        y_probs = self.predict_proba(X)
        return np.where(y_probs > threshold, 1.0, 0.0)

# Các hàm đánh giá (Metrics)
def accuracy_score_numpy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1