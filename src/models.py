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
        self.weights = np.zeros(n_features) # Khởi tạo trọng số w = 0
        self.bias = 0 # Khởi tạo bias b = 0
        
        for i in range(self.num_iterations):
            # Tính z = w*x + b
            linear_pred = np.dot(X, self.weights) + self.bias
            
            # Tính y_hat = sigmoid(z) -> Xác suất dự đoán
            y_pred = self._sigmoid(linear_pred)
            
            # Tính đạo hàm (hướng dốc của đồi)
            # dw: Đạo hàm theo trọng số w
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            # db: Đạo hàm theo bias b
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update weights: Đi xuống đồi
            # Di chuyển ngược hướng đạo hàm để giảm lỗi
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Đo độ sai lệch
            epsilon = 1e-15 # Số cực nhỏ để tránh log(0)
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            # Dùng công thức Loss: -mean(y*log(y_hat) + (1-y)*log(1-y_hat))
            loss = - (1/n_samples) * np.sum(y * np.log(y_pred_clipped) + (1-y) * np.log(1-y_pred_clipped))
            self.losses.append(loss)
            
            if i % 1000 == 0:
                print(f"Iteration {i}: Loss {loss:.4f}")
                
    def predict_proba(self, X):
        # Chỉ chạy Forward Pass: Tính z -> Sigmoid(z)
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):
        # Lấy xác suất
        y_probs = self.predict_proba(X)
        # Nếu > 0.5 thì là lớp 1, ngược lại là lớp 0
        return np.where(y_probs > threshold, 1.0, 0.0)

# Các hàm đánh giá (Metrics)
def accuracy_score_numpy(y_true, y_pred):
    # Tính trung bình cộng dựa trên mảng True/False từ điều kiện y_true == y_pred -> Tỷ lệ đoán đúng
    return np.mean(y_true == y_pred)

def confusion_matrix_metrics(y_true, y_pred):
    # TP (True Positive): Thực tế là 1 VÀ Dự đoán là 1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # TN (True Negative): Thực tế là 0 VÀ Dự đoán là 0
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    # FP (False Positive): Thực tế là 0 nhưng dự đoán nhầm là 1
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    # FN (False Negative): Thực tế là 1 nhưng dự đoán nhầm là 0
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Precision = TP / (TP + FP) -> Độ chính xác khi dự đoán là 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall = TP / (TP + FN) -> Độ nhạy, khả năng tìm ra tất cả số 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1-Score: Trung bình điều hòa của Precision và Recall
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall) 
    else:
        f1 = 0
    
    return precision, recall, f1