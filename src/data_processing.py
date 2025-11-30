import numpy as np

def load_csv_numpy(filepath, dtypes=None):
    """
    Đọc file CSV, trả về header và data (structured array)
    """
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
    
    # dtype=None để tự động detect kiểu, encoding utf-8
    data = np.genfromtxt(filepath, delimiter=',', names=True, dtype=dtypes, encoding='utf-8')
    return header, data

def validate_data_integrity(data):
    """
    Kiểm tra tính hợp lệ của dữ liệu
    """
    report = []
    
    # Kiểm tra thuộc tính "training_hours" có dương hay không
    th = data['training_hours']
    invalid_th = np.sum(th < 0)
    if invalid_th > 0:
        report.append(f"Thuộc tính {invalid_th} có giá trị Training Hours âm")
    
    # Kiểm tra thuộc tính "city_development_index" có nằm trong khoảng 0 - 1 hay không 
    cdi = data['city_development_index']
    invalid_cdi = np.sum((cdi < 0) | (cdi > 1))
    if invalid_cdi > 0:
        report.append(f"Thuộc tính {invalid_cdi} có giá trị CDI không hợp lệ (ngoài khoảng 0-1)")
        
    if not report:
        print("Dữ liệu sạch, không có giá trị vô lý")
    else:
        for line in report: 
            print(line)

def fill_missing_categorical(data, col_name, strategy='mode'):
    """
    Điền giá trị thiếu
    - strategy='mode': Điền vào giá trị phổ biến nhất
    - strategy='unknown': Gán trực tiếp giá trị 'Unknown' vào các vị trí thiếu
    """
    col_data = data[col_name]
    is_missing = (col_data == '') | (col_data == 'nan')
    
    if np.sum(is_missing) == 0: 
        return data
    
    if strategy == 'unknown':
        data[col_name][is_missing] = 'Unknown'
    else:
        # Lọc ra các giá trị không bị thiếu, đếm tần suất của chúng và lấy gía trị xuất hiện nhiều nhất để điền vào chỗ  bị thiếu  
        valid_vals = col_data[~is_missing]
        unique, counts = np.unique(valid_vals, return_counts=True)
        if len(counts) > 0:
            mode_val = unique[np.argmax(counts)]
            data[col_name][is_missing] = mode_val
    return data

def ordinal_encode_experience(data):
    """
    Encode đặc trưng Experience sang số
    """
    col = data['experience']
    new_col = np.zeros(len(col))
    for i, val in enumerate(col):
        if val == '<1': 
            new_col[i] = 0
        elif val == '>20': 
            new_col[i] = 21
        elif val == '': 
            new_col[i] = 0
        else:
            try: new_col[i] = float(val)
            except: new_col[i] = 0
    return new_col

def label_encode(data, col_name):
    """Label Encoding cơ bản"""
    col = data[col_name]
    unique_vals, indices = np.unique(col, return_inverse=True)
    return indices

def create_new_features(data, exp_numeric):
    """
    Tạo các đặc trưng phái sinh:
    1. Interaction: CDI * Experience
    2. Ratio: Training Hours / (Experience + 1) 
    """
    cdi = data['city_development_index']
    th = data['training_hours']
    
    # Feature 1: Interaction (CDI * Exp)
    cdi_exp = cdi * exp_numeric
    
    # Feature 2: Training Intensity
    # Cường độ học so với số năm kinh nghiệm
    intensity = th / (exp_numeric + 1.0)
    
    return cdi_exp, intensity

def remove_outliers_iqr(col_data, threshold=1.5):
    """
    Loại bỏ ngoại lai sử dụng IQR
    """
    q75, q25 = np.percentile(col_data, [75 ,25])
    iqr = q75 - q25
    lower = q25 - threshold * iqr
    upper = q75 + threshold * iqr
    return (col_data >= lower) & (col_data <= upper)

def min_max_normalize(array):
    """
    Min-Max Scaling: [0, 1]
    """
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0: 
        return array
    return (array - min_val) / (max_val - min_val)

def log_transformation(array):
    """
    Log Transformation: Giúp xử lý dữ liệu bị lệch
    """
    # Đảm bảo không có giá trị âm
    if np.min(array) < 0:
        array = array - np.min(array)
    return np.log1p(array)

def decimal_scaling(array):
    """
    Decimal Scaling: x = x / 10^j
    """
    max_abs = np.max(np.abs(array))
    if max_abs == 0: 
        return array
    j = np.ceil(np.log10(max_abs))
    return array / (10**j)

def standard_scale(array):
    """
    Z-score Standardization: (x - mean) / std
    """
    mean = np.mean(array)
    std = np.std(array)
    return (array - mean) / (std + 1e-9)

def robust_scale(array):
    """
    Robust Scaling: (x - Median) / IQR.
    """
    median = np.median(array)
    q75, q25 = np.percentile(array, [75, 25])
    iqr = q75 - q25
    
    if iqr == 0: 
        return array - median # Tránh chia cho 0
    
    return (array - median) / iqr

def perform_pca(X, n_components=2):
    """
    Thực hiện PCA
    """
    # 1. Center data
    X_centered = X - np.mean(X, axis=0)
    
    # 2. Covariance Matrix
    n_samples = X_centered.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    
    # 3. Eigen Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Chọn k thành phần
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    eigenvectors_subset = eigenvectors[:, :n_components]
    
    # 5. Transform 
    X_reduced = np.dot(X_centered, eigenvectors_subset)
    
    if np.sum(eigenvalues) > 0:
        explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
    else:
        explained_variance = np.zeros(n_components)
    
    return X_reduced, explained_variance

def calculate_t_test_ind(group1, group2):
    """
    Tính T-statistic cho 2 mẫu độc lập
    """
    n1 = len(group1)
    n2 = len(group2)
    
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    
    se = np.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / (se + 1e-9)
    
    return t_stat, mean1, mean2

def save_processed_csv(X, y, filepath):
    """
    Lưu file CSV
    """
    if y is not None:
        combined = np.column_stack((X, y))
    else:
        combined = X
    np.savetxt(filepath, combined, delimiter=",", fmt='%.6f')
    print(f"Đã lưu file tại: {filepath}")