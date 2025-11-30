import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_target_distribution(y):
    """
    Vẽ biểu đồ tròn thể hiện tỷ lệ Target
    """
    counts = np.bincount(y.astype(int))
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=['Not Looking (0)', 'Looking (1)'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
    plt.title('Tỷ lệ ứng viên muốn thay đổi công việc')
    plt.show()

def plot_categorical_vs_target(data, col_name, target_col='target'):
    """
    Vẽ biểu đồ cột so sánh các đặc trưng thuộc loại định tính với Target
    """
    unique_vals = np.unique(data[col_name])
    looking = []
    not_looking = []
    
    targets = data[target_col]
    
    for val in unique_vals:
        mask = (data[col_name] == val)
        subset_target = targets[mask]
        looking.append(np.sum(subset_target == 1))
        not_looking.append(np.sum(subset_target == 0))
        
    x = np.arange(len(unique_vals))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, not_looking, width, label='Not Looking (0)')
    ax.bar(x + width/2, looking, width, label='Looking (1)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(unique_vals, rotation=45)
    ax.set_xlabel(col_name)
    ax.set_title(f'Mối quan hệ giữa {col_name} và target')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_loss_curve(losses):
    """Vẽ đường Loss khi training"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Hàm mất mát qua các vòng lặp (Gradient Descent)')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()