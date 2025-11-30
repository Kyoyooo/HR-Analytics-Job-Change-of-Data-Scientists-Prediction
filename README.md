# HR Analytics: Job Change Prediction of Data Scientists
### *A Pure NumPy Implementation (No Pandas, No Scikit-learn)*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

Đồ án xây dựng quy trình end-to-end (từ khám phá, xử lý dữ liệu đến mô hình hóa) để dự đoán xác suất một nhà Khoa học dữ liệu sẽ thay đổi công việc. Điểm đặc biệt của đồ án là **không sử dụng Pandas hay Scikit-learn** cho các tác vụ tính toán, mà tự triển khai toàn bộ thuật toán bằng **NumPy** để tối ưu hóa.

---

## Mục lục
1. [Giới thiệu](#-giới-thiệu)
2. [Tổng quan về bộ dữ liệu (Dataset)](#-dataset)
3. [Phương pháp tiếp cận (Method)](#-phương-pháp--thuật-toán)
4. [Cài đặt & thiết lập chương trình (Installation & Setup)](#-installation--setup)
5. [Hướng dẫn sử dụng (Usage)](#-usage)
6. [Kết quả](#-results)
7. [Cấu trúc đồ án](#-project-structure)
8. [Thách thức & Giải pháp](#-challenges--solutions)
9. [Hướng phát triển tiếp theo (Future Improvements)](#-future-improvements)
10. [Đóng góp (Contributors)](#-contributors)
11. [Bản quyền (License)](#-thông-tin-tác-giả)

---

## Giới thiệu

* **Mô tả bài toán:**
Một công ty hoạt động trong lĩnh vực Dữ liệu lớn (Big Data) và Khoa học dữ liệu (Data Science) muốn tuyển dụng các nhà khoa học dữ liệu từ những người đã hoàn thành xuất sắc các khóa học do chính công ty tổ chức. Có rất nhiều người đăng ký tham gia khóa đào tạo của họ. Vì vậy công ty muốn xác định xem ứng viên nào thực sự muốn làm việc cho họ sau khi đào tạo xong hoặc đang tìm kiếm cơ hội việc làm mới.

* **Động lực & Ứng dụng:**
   * **Giảm chi phí:** Việc này giúp giảm thiểu chi phí và thời gian, cũng như nâng cao chất lượng đào tạo, hỗ trợ việc lên kế hoạch khóa học và phân loại ứng viên.
   * **Chiến lược nhân sự:** Hiểu được các yếu tố (kinh nghiệm, thành phố, giờ học,...) ảnh hưởng đến quyết định nghỉ việc như thế nào, giúp cho HR có thể đưa ra chính sách giữ chân nhân tài tốt hơn.

* **Mục tiêu:**
Xây dựng mô hình phân lớp nhị phân (Binary Classification):
   * **Input:** Thông tin nhân khẩu học, kinh nghiệm, quá trình đào tạo,...
   * **Output:** Xác suất ứng viên tìm kiếm việc làm mới (Target: 1 = Có, 0 = Không).

---

## Dataset

* **Nguồn dữ liệu:** [HR Analytics: Job Change of Data Scientists (Kaggle)](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)
* **Kích thước:** 19158 mẫu cho tập Train (14 đặc trưng) và 2129 mẫu cho tập Test (13 đặc trưng vì loại bỏ đi Target).
* **Đặc điểm:** Dữ liệu hỗn hợp (có cả loại định tính và định lượng), cần xử lý trường hợp giá trị thiếu (Missing Values) và mất cân bằng nhãn (Imbalanced Dataset).

**Các đặc trưng chính:**
* `enrollee_id`: Mã số duy nhất đại diện cho mỗi ứng viên
* `city`: Mã thành phố nơi ứng viên đó sinh sống
* `city_developement_index`: Chỉ số phát triển của thành phố nơi ứng viên đó sinh sống
* `gender`: Giới tính của ứng viên
* `relevent_experience`: Cho biết ứng viên đã có kinh nghiệm trong lĩnh vực liên quan hay chưa
* `enrolled_university`: Loại hình khoá học đại học mà ứng viên đó đang theo học (nếu có)
* `education_level`: Trình độ học vấn của ứng viên
* `major_discipline`: Chuyên ngành học của ứng viên
* `experience`: Tổng số năm kinh nghiệm từ trước đến nay (tính theo năm)
* `company_size`: Quy mô nhân sự ở công ty hiện tại của ứng viên
* `company_type`: Loại hình công ty hiện tại của ứng viên
* `last_new_job`: Số năm kể từ lần cuối cùng ứng viên thay đổi công việc
* `training_hours`: Số giờ hoàn thành việc đào tạo
* `target`: Nếu là 0 – Không tìm kiếm công việc mới, nếu là 1 – Đang tìm kiếm công việc mới  

---

## Phương pháp tiếp cận (Method)

### 1. Khám phá dữ liệu (Data Exploration) 
Sử dụng thư viện **Matplotlib** và **Seaborn** để trực quan hoá dữ liệu, đưa ra 1 số góc nhìn thú vị về dữ liệu qua việc trả lời 1 số câu hỏi như sau:
* Tỷ lệ người muốn thay đổi công việc là bao nhiêu?
* Trình độ học vấn (Education Level) ảnh hưởng thế nào đến khả năng muốn thay đổi công việc?
* Liệu những ứng viên có kinh nghiệm liên quan (Has relevant experience) thì trung thành hơn hay dễ thay đổi việc hơn so với người mới (No relevant experience)?
* Liệu số năm kể từ lần cuối cùng ứng viên thay đổi công việc trước đó có mối liên hệ như thế nào đến khả năng muốn thay đổi công việc hiện tại?

### 2. Xây dựng quy trình xử lý dữ liệu (Preprocessing Pipeline)
* **Đọc và load dữ liệu** Viết hàm đọc CSV sử dụng `csv` module.
* **Kiểm tra tính hợp lệ:**
    * Kiểm tra thuộc tính "training_hours" có dương hay không.
    * Kiểm tra thuộc tính "city_development_index" có nằm trong khoảng 0 - 1 hay không.
* **Xử lý Missing Values:**
    * Với các giá trị định tính: Điền vào giá trị phổ biến nhất hoặc điền 'Unknown'.
    * Với các giá trị định lượng: Điền bằng Mean/Median.
* **Kiểm định giả thuyết thống kê:** Sử dụng kiểm định T-test độc lập để so sánh giá trị trung bình của đặc trưng training_hours giữa hai nhóm ứng viên: nhóm không đổi việc (Target=0) và nhóm muốn đổi việc (Target=1) để trả lời cho câu hỏi: "Việc ứng viên tham gia đào tạo nhiều giờ hơn có liên quan đến việc họ muốn bỏ việc hay không?"
* **Feature Engineering:**
    * Tạo đặc trưng Interaction với công thức `cdi_exp = city_developement_index * relevent_experience_numeric`. Ý nghĩa: Kết hợp Chỉ số phát triển thành phố và Kinh nghiệm.
    * Tạo đặc trưng Training Intensity với công thức `intensity = training_hours / (relevent_experience_numeric + 1.0)`. Ý nghĩa: Đo Cường độ học tập. 
* **Xử lý các giá trị ngoại lai (Outliers):** Sử dụng phương pháp IQR (Interquartile Range).
* **Chuẩn hoá (Normalization) cho từng đặc trưng:**
    * Kỹ thuật `min_max_normalize` được áp dụng cho đặc trưng `city_development_index` với cơ chế hoạt động là tập trung co giãn dữ liệu về đúng một khoảng cố định, thường là trong đoạn [0, 1]. Công thức: $$X_{new} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
    * Kỹ thuật `decimal_scaling` được áp dụng cho đặc trưng `experience` với cơ chế hoạt động là di chuyển dấu thập phân của số liệu sang trái cho đến khi giá trị tuyệt đối nhỏ hơn 1. Công thức: $$X_{new} = \frac{X}{10^j}$$ (với $j$ là số nguyên nhỏ nhất để $|X_{new}| < 1$).
* **Điều chuẩn khoảng giá trị phù hợp dữ liệu Non-Gaussian Distribution:** Áp dụng 2 kỹ thuật `log transformation` và `Robust Scaling` cho 2 đặc trưng `training_hours` và `intensity`
    * `log transformation` biến đổi phân phối của dữ liệu, giúp dữ liệu trở nên gần với phân phối chuẩn (Gaussian Distribution) hơn và xử lý các vấn đề về giá trị âm. Ý nghĩa là giảm độ lệch của dữ liệu và đưa dữ liệu về cùng scaling. 
    * `Robust Scaling` định vị tâm (Centering) bằng cách trừ mỗi giá trị của dữ liệu cho Median và chia cho IQR để thu hẹp hoặc mở rộng dữ liệu về cùng một đơn vị đo lường tương đối. Công thức: $$X_{new} = \frac{X - Median}{IQR}$$ (với $IQR = Q3 - Q1$).
    
* **Điều chuẩn dữ liệu (Standardization) hay Z-score để đạt trung bình 0 và phương sai 1:** Đối với đặc trưng `cdi_exp` và các đặc trưng thuộc loại định tính, áp dụng kỹ thuật `standard_scale` để  điều chuẩn dữ liệu hay Z-score để đạt trung bình $\mu$ = 0 và phương sai $\sigma^2$ = 1 với cơ chế hoạt động đưa dữ liệu về phân phối chuẩn tắc. Công thức: $$X_{new} = \frac{X - \mu}{\sigma}$$

* **Giảm chiều dữ liệu dùng kỹ thuật PCA (Principal Components Analysis)**
    * Center dữ liệu: $X_{centered} = X - \bar{X}$
    * Tính ma trận hiệp phương sai: $Cov = \frac{1}{n-1} X^T X$
    * Phân rã trị riêng (Eigendecomposition): $Cov \cdot v = \lambda \cdot v$
    * Sắp xếp và chọn k thành phần tương ứng với k hướng (eigenvectors) quan trọng nhất, chứa nhiều thông tin nhất (k giá trị eigenvalues lớn nhất), sau đó chuyển đổi dữ liệu qua chiều không gian mới (k chiều được chọn). 

### 3. Xây dựng mô hình Logistic Regression
Mô hình được xây dựng thủ công với thuật toán **Gradient Descent**.

* **Hypothesis (Sigmoid):**
    $$h_\theta(x) = \frac{1}{1 + e^{-z}} \quad \text{với } z = \theta^T x + b$$
* **Loss Function (Binary Cross Entropy):**
    $$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$
* **Update Rule (Gradient Descent):**
    $$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

---

## Cài đặt & thiết lập chương trình (Installation & Setup)

1.  **Clone đồ án:**
    ```bash
    git clone https://github.com/Kyoyooo/HR-Analytics-Job-Change-of-Data-Scientists-Prediction.git
    cd HR-Analytics-Job-Change-of-Data-Scientists-Prediction
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```
    
---

## Hướng dẫn sử dụng (Usage)

Chạy lần lượt các notebook theo thứ tự:

1.  **Khám phá dữ liệu:**
    * Mở `notebooks/01_data_exploration.ipynb`
    * Xem các phân tích về dữ liệu. 

2.  **Tiền xử lý dữ liệu:**
    * Mở `notebooks/02_preprocessing.ipynb`
    * Chạy chương trình để tiền xử lý dữ liệu, xử lý missing values, áp dụng Feature engineering và thực hiện tính toán thống kê mô tả, kiểm định giả thiết thống kê.
    * Dữ liệu sau khi xử lý sẽ được lưu vào thư mục `data/processed/`.

3.  **Huấn luyện mô hình & Đánh giá:**
    * Mở `notebooks/03_modeling.ipynb`
    * Huấn luyện mô hình Logistic Regression.
    * Ghi kết quả ra file `sample_submission_final.csv` trong thư mục `data/processed/`.

---

## Kết quả

### Các độ đo đánh giá mô hình
* **Accuracy:** ~77.2%
* **Precision:** ~55.41%
* **Recall:** ~23.6% (Dữ liệu mất cân bằng)
* **F1-Score:** ~0.331 

### Trực quan hóa
* **Loss Curve:** Hàm mất mát giảm dần và hội tụ tốt sau 5000 vòng lặp. Kết quả như sau:
    * Iteration 0: Loss 0.6931
    * Iteration 1000: Loss 0.5008
    * Iteration 2000: Loss 0.4954
    * Iteration 3000: Loss 0.4943
    * Iteration 4000: Loss 0.4940

* **Phân phối Target:** Tỷ lệ mất cân bằng ~ 3:1 giữa nhóm 0 và 1.

---

## Cấu trúc đồ án

```text
HR-Analytics-Job-Change-of-Data-Scientists-Prediction/
├── data/                          # Thư mục chứa dữ liệu
│   ├── raw/                       # Dữ liệu gốc
│   └── processed/                 # Dữ liệu sau khi xử lý 
├── notebooks/                     # Thư mục chứa các notebook 
│   ├── 01_data_exploration.ipynb  # Khám phá và trực quan hoá dữ liệu
│   ├── 02_preprocessing.ipynb     # Xử lý dữ liệu
│   └── 03_modeling.ipynb          # Xây dựng Logistic Regression từ đầu 
├── src/                           # Thư mục chứa các file python để xử lý các hàm con 
│   ├── __init__.py
│   ├── data_processing.py         # Thư viện chứa các hàm xử lý dữ liệu NumPy
│   ├── visualization.py           # Thư viện chứa các hàm vẽ biểu đồ
│   └── models.py                  # Class LogisticRegressionNumPy
├── README.md 
└── requirements.txt # Chứa các gói cài đặt để thiết lập chương trình
```

## Thách thức & Giải pháp
Trong quá trình thực hiện đồ án chỉ với thư viện **NumPy**, tác giả đã gặp một số thách thức:

* Xử lý dữ liệu hỗn hợp : Hàm `np.genfromtxt` thường gặp lỗi khi đoán kiểu dữ liệu cho các cột chứa ký tự lạ ('>4', '<1'). Giải pháp đưa ra là khai báo dữ liệu cho mỗi đặc trưng trước khi tiến hành đọc dữ liệu.
* Xử lý Vectorization khi khó khăn trong việc tối ưu tốc độ tính toán Gradient Descent. Giải pháp đưa ra là sử dụng phương pháp Broadcasting và Dot Product (`np.dot`, `np.sum`) thay vì `for loop`.
* Hàm `log` đôi lúc gặp lỗi với giá trị 0 hoặc số âm và hàm `exp` có lúc bị tràn số (overflow). Giải pháp đưa ra là sử dụng `np.clip` cho hàm `Sigmoid` và cộng thêm `epsilon` (1e-9) vào các mẫu số.

## Future Improvements
* Cải thiện độ đo Recall bằng kỹ thuật SMOTE (Synthetic Minority Over-sampling Technique) để xử lý mất cân bằng dữ liệu.
* Thử nghiệm các mô hình phi tuyến tính như Neural Networks (xây dựng từ đầu bằng NumPy với Backpropagation).
* Tối ưu hóa Hyperparameters (Learning rate, Regularization) bằng Grid Search thủ công.

## Đóng góp (Contributors)
* Thông tin tác giả: Võ Trần Duy Hoàng (MSSV: 23120266)
* Phương thức liên lạc (Contact):
    * Email: 23120266@student.hcmus.edu.vn
    * Github: https://github.com/Kyoyooo

## Giấy phép (Licenses)
Đồ án được phân phối dưới giấy phép MIT License.
