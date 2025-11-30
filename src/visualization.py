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

def analyze_experience_impact(exp_col, target):    
    # Nhóm 1: Có kinh nghiệm liên quan
    has_exp_mask = (exp_col == 'Has relevent experience')
    rate_has_exp = np.mean(target[has_exp_mask]) * 100
    count_has = np.sum(has_exp_mask)
    
    # Nhóm 2: Không có kinh nghiệm liên quan
    no_exp_mask = (exp_col == 'No relevent experience')
    rate_no_exp = np.mean(target[no_exp_mask]) * 100
    count_no = np.sum(no_exp_mask)
    
    # Trực quan hóa
    labels = ['Has Relevant Experience\n(Có kinh nghiệm)', 'No Relevant Experience\n(Chưa có kinh nghiệm)']
    rates = [rate_has_exp, rate_no_exp]
    
    plt.figure(figsize=(10, 6))
    # Sử dụng màu sắc để nhấn mạnh: Xanh (Thấp/An toàn) - Cam (Cao/Rủi ro)
    colors = ['#2ecc71', '#e67e22']
    
    ax = sns.barplot(x=labels, y=rates, palette=colors)
    
    plt.title('Tỷ lệ muốn thay đổi công việc theo đặc trưng relevent_experience', fontsize=14, fontweight='bold')
    plt.ylabel('Tỷ lệ Target = 1 (%)', fontsize=12)
    plt.ylim(0, 40) 
    
    # Thêm nhãn giá trị
    for i, rate in enumerate(rates):
        ax.text(i, rate + 1, f'{rate:.2f}%', ha='center', fontsize=12, fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()
    
    print(f"Tỷ lệ của nhóm có kinh nghiệm: {rate_has_exp:.2f}% (Số lượng: {count_has})")
    print(f"Tỷ lệ của nhóm chưa có kinh nghiệm: {rate_no_exp:.2f}% (Số lượng: {count_no})")

def analyze_last_new_job(last_job, target):    
    # Lọc bỏ giá trị thiếu hoặc trống
    mask_valid = (last_job != '') & (last_job != 'nan')
    last_job = last_job[mask_valid]
    target = target[mask_valid]
    
    # Tính toán tỷ lệ nghỉ việc theo nhóm
    ordered_labels = ['never', '1', '2', '3', '4', '>4']
    
    churn_rates = []
    counts = []
    
    for label in ordered_labels:
        mask = (last_job == label)
        if np.sum(mask) > 0:
            rate = np.mean(target[mask]) * 100
            churn_rates.append(rate)
            counts.append(np.sum(mask))
        else:
            churn_rates.append(0)
            counts.append(0)
            
    # Trực quan hóa (Bar Chart)
    plt.figure(figsize=(10, 6))
    
    # Tạo palette màu nhấn mạnh sự khác biệt
    # Màu đậm hơn cho tỷ lệ cao hơn
    ax = sns.barplot(x=ordered_labels, y=churn_rates, palette="rocket")
    
    plt.title('Tỷ lệ muốn thay đổi công việc hiện tại theo đặc trưng last_new_job', fontsize=14, fontweight='bold')
    plt.xlabel('Số năm từ lần thay đổi công việc cuối trước đó', fontsize=12)
    plt.ylabel('Tỷ lệ muốn thay đổi công việc (%)', fontsize=12)
    plt.ylim(0, 35) 
    
    # Thêm nhãn giá trị lên cột
    for i, rate in enumerate(churn_rates):
        ax.text(i, rate + 0.5, f'{rate:.1f}%', ha='center', fontweight='bold', fontsize=11)
        
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()
    
    # In kết quả
    for label, rate, count in zip(ordered_labels, churn_rates, counts):
        print(f"Nhóm {label}: {rate:.2f}% (Số lượng: {count})")

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