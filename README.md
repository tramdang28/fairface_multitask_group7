# Đồ án: PHÂN LOẠI CHỦNG TỘC, GIỚI TÍNH, ĐỘ TUỔI QUA ẢNH KHUÔN MẶT

## 1. Giới thiệu

- Đồ án tập trung vào bài toán phân loại chủng tộc, giới tính và độ tuổi từ ảnh khuôn mặt bằng các kỹ thuật học sâu trên bộ dữ liệu FairFace. Đây là một bài toán phân loại đa thuộc tính sử dụng kiến trúc Học đa nhiệm (Multi-task Learning), trong đó mỗi ảnh đầu vào được sử dụng để dự đoán đồng thời 3 nhãn đầu ra:
    - Chủng tộc (Race): 7 lớp (White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino_Hispanic).
    - Giới tính (Gender): 2 lớp (Male, Female).
    - Độ tuổi (Age): 9 nhóm tuổi.

Mỗi mô hình sử dụng một backbone chung để trích xuất đặc trưng và ba head dự đoán độc lập cho ba nhiệm vụ trên.

**Mục tiêu**: Mục tiêu của đồ án là xây dựng, huấn luyện, đánh giá và triển khai thử nghiệm các mô hình học sâu cho bài toán phân loại đồng thời chủng tộc, giới tính và độ tuổi từ ảnh khuôn mặt. Ngoài việc so sánh hiệu năng giữa nhiều kiến trúc khác nhau, đồ án còn phát triển một ứng dụng web minh họa bằng Streamlit nhằm kiểm chứng khả năng ứng dụng thực tế của mô hình được lựa chọn.

Bốn kiến trúc được triển khai và đánh giá bao gồm:

| **Kiến trúc Backbone**|                    **Mô tả**                      | 
|-----------------------|---------------------------------------------------| 
| Custom CNN            | Mạng CNN tự xây dựng làm mô hình cơ sở            | 
| MobileNetV3-Large     | Kiến trúc nhẹ, tối ưu tốc độ và số tham số        | 
| EfficientNet-B0       | Cân bằng giữa độ chính xác và chi phí tính toán   | 
| ViT-Small             | Vision Transformer cho bài toán thị giác máy tính |


Kết quả của các mô hình được tổng hợp và so sánh thông qua các chỉ số đánh giá cũng như lịch sử huấn luyện, làm cơ sở để phân tích ưu điểm và hạn chế của từng kiến trúc.
---

# 2. Cấu trúc thư mục

```text
project/
│
├── README.md                # Hướng dẫn chạy và tổng quan kết quả đồ án
├── requirements.txt         # Khai báo các thư viện Python cần cài đặt
│
├── notebooks/              # Mã nguồn triển khai trên Google Colab (.ipynb)
│   ├── 01_CustomCNN.ipynb
│   ├── 02_MobileNetV3_Large.ipynb
│   ├── 03_EfficientNet_B0.ipynb
│   ├── 04_ViT_Small.ipynb
│   └── FairFace_Main.ipynb     # Notebook tổng hợp, so sánh và đánh giá cuối cùng
│
├── runs/    #Thư mục chứa log của TensorBoard (# Chứa file tfevents của từng mạng)
│   └── tensorboard
│       ├── Custom_CNN/
│       ├── MobileNetV3_Large/
│       ├── EfficientNet_B0/
│       └── ViT_Small/
│
├── results/     # Lưu trữ số liệu thực nghiệm định dạng dữ liệu bảng (.csv)
│   │
│   ├── CustomCNN/                     
│   │   ├── tuning_results.csv         
│   │   ├── best_model_metrics.csv     
│   │   └── training_history.csv       
│   │
│   ├── MobileNetV3_Large/             
│   │   ├── tuning_results.csv         
│   │   ├── best_model_metrics.csv     
│   │   └── training_history.csv       
│   │
│   ├── EfficientNet_B0/               
│   │   ├── tuning_results.csv         
│   │   ├── best_model_metrics.csv     
│   │   └── training_history.csv       
│   │
│   ├── ViT_Small/                     
│   │   ├── tuning_results.csv         
│   │   ├── best_model_metrics.csv     
│   │   └── training_history.csv    
│   │    
│   └── Main/                 
│       ├── CustomCNN_metrics.csv          
│       ├── CustomCNN_train_history.csv    
│       ├── EfficientNet_B0_metrics.csv          
│       ├── EfficientNet_B0_train_history.csv    
│       ├── MobileNetV3_Large_metrics.csv          
│       ├── MobileNetV3_Large_train_history.csv    
│       ├── ViT_Small_metrics.csv          
│       ├── ViT_Small_train_history.csv             
│       └── model_comparison_multitask.csv       
│
├── figures/
│   │
│   ├── demo 
│   │ ├── streamlit_demo.png 
│   │ ├── result_1.png 
│   │ ├── result_2.png 
│   │ └── sample.png 
│   │
│   ├── EDA
│   │ ├── augmentation_examples.png 
│   │ ├── label_distribution.png  
│   │ └── sample_images_multitask.png 
│   │ 
│   ├── CustomCNN/                     
│   │   ├── accuracy_curve.png             
│   │   └── loss_curve.png         
│   │
│   ├── MobileNetV3_Large/             
│   │   ├── accuracy_curve.png             
│   │   └──  loss_curve.png           
│   │
│   ├── EfficientNet_B0/               
│   │   ├── accuracy_curve.png             
│   │   └──  loss_curve.png         
│   │
│   ├── ViT_Small/                     
│   │   ├── accuracy_curve.png             
│   │   └──  loss_curve.png        
│   │
│   └── Main/                    
│       ├── confusion_matrix_age.png    
│       ├── confusion_matrix_gender.png          
│       ├── confusion_matrix_race.png    
│       ├── learning_curves.png          
│       ├── model_comparison.png    
│       ├── performance_vs_cost.png          
│       └── radar_chart.png   
│      
├── models/
│   ├── CustomCNN_best.pth             
│   ├── CustomCNN_last.pth     
│   ├── EfficientNet_B0_best.pth       
│   ├── EfficientNet_B0_last.pth             
│   ├── MobileNetV3_Large_best.pth     
│   ├── MobileNetV3_Large_last.pth 
│   ├── ViT_Small_best.pth      
│   └── ViT_Small_last.pth             
│
├── fairface_demo/
│   ├── app.py            
│   ├── model_def.py 
│   └── ViT_Small_best.pth 
│
└── report/              # Tài liệu báo cáo 
     └──report.pdf 
     
```
---

# 3. Mô tả các thư mục

## notebooks/

- Thư mục `notebooks`: chứa toàn bộ mã nguồn triển khai trên Google Colab.
    - Mỗi notebook từ 01 đến 04 chứa chuỗi thí nghiệm tìm kiếm hyperparameter tốt nhất của từng kiến trúc. 
    - Mỗi notebook có thể chứa **nhiều thí nghiệm với các cấu hình khác nhau**. Sau khi hoàn tất, chỉ **thí nghiệm đạt kết quả tốt nhất** được sử dụng cho các bước tổng hợp tiếp theo.

- `FairFace_Main.ipynb`: Chạy và tổng hợp kết quả của **mô hình tốt nhất** từ bốn kiến trúc để phục vụ so sánh và đánh giá cuối cùng.

---

## results/

Lưu trữ kết quả thực nghiệm.

- Đối với từng kiến trúc (`CustomCNN`, `MobileNetV3_Large`, `EfficientNet_B0`, `ViT_Small`), thư mục tương ứng chỉ lưu **kết quả của thí nghiệm tốt nhất**, bao gồm:
    - `tuning_results.csv`: Kết quả thử nghiệm các cấu hình.
    - `best_model_metrics.csv`: Các chỉ số đánh giá của mô hình được chọn (mô hình tốt nhất).
    - `training_history.csv`: Lịch sử huấn luyện của mô hình tốt nhất.

Thư mục `results/Main/` dùng để tổng hợp kết quả cuối cùng, bao gồm:

- `model_comparison_multitask.csv`: Bảng tổng hợp đối sánh hiệu năng tổng thể và chi phí vận hành phần cứng giữa 4 kiến trúc mạng. Tệp tin này lưu trữ các chỉ số đo lường cốt lõi bao gồm:
    - Nhóm chỉ số hiệu năng tổng hợp: Độ chính xác trung bình (`Avg Acc (%)`), điểm F1-Score tổng thể toán học (`Avg Macro F1`), cùng hai chỉ số bổ trợ trung bình (`Avg Precision` và `Avg Recall`).

    - Nhóm chỉ số phân tách tác vụ: Điểm F1-Score Macro phân rã chi tiết cho từng nhiệm vụ song song là nhận diện chủng tộc (`Race F1`), giới tính (`Gender F1`), và phân đoạn độ tuổi (`Age F1`).

    - Nhóm chỉ số tài nguyên hệ thống: Tổng số lượng tham số thực tế của mô hình mạng đa nhiệm (`Params (M)`), tốc độ suy luận trích xuất trên một đơn vị ảnh mẫu (`Inf (ms/img)`), và tổng thời gian tiêu tốn cho chu kỳ huấn luyện (`Train (min)`).
* Các file .csv lưu lịch sử huấn luyện và các chỉ số đánh giá của từng mô hình.

Các tệp `train_history.csv` trong `results/Main/` là lịch sử huấn luyện của **mô hình tốt nhất** thuộc từng kiến trúc và được sử dụng để so sánh trực quan.

---

## runs/

Lưu trữ các tập tin tfevents phục vụ cho việc gọi và hiển thị giao diện giám sát trực quan TensorBoard. Các log này được sinh ra khi chạy file FairFace_Main.ipynb.

---

## figures/

Chứa các hình ảnh được xuất ra và các đồ thị trực quan hóa kết quả trong quá trình thực nghiệm, bao gồm:

- Các hình ảnh và biểu đồ khi khám phá dữ liệu (EDA).
- Learning curves của các mô hình.
- Các biểu đồ so sánh giữa các mô hình (Accuracy, F1-score, thời gian suy luận, ...).
- Các biểu đồ phân tích sâu như Ma trận nhầm lẫn (Confusion Matrix) cho cả 3 tác vụ và biểu đồ Radar đánh giá năng lực toàn diện.
- Các hình ảnh phục vụ triển khai ứng dụng, bao gồm ảnh khuôn mặt mẫu dùng để kiểm thử mô hình và ảnh chụp giao diện Streamlit minh họa quá trình suy luận thực tế.
---

## models/

Thư mục `models/` lưu trữ các file checkpoint (`.pth`) của từng mô hình, bao gồm:

- `*_best.pth`: Checkpoint có hiệu năng tốt nhất trên tập validation trong quá trình huấn luyện.
- `*_last.pth`: Checkpoint được lưu tại epoch cuối cùng của quá trình huấn luyện.

Các checkpoint này được tạo ra khi chạy `FairFace_Main.ipynb`, trong đó mỗi kiến trúc được huấn luyện lại với bộ siêu tham số tối ưu đã được lựa chọn từ giai đoạn thực nghiệm.

---

## fairface_demo/

Chứa ứng dụng web minh họa được xây dựng bằng Streamlit nhằm triển khai và kiểm tra khả năng suy luận của mô hình trong điều kiện thực tế, bao gồm:

- `app.py`: Giao diện ứng dụng Streamlit cho phép người dùng tải ảnh khuôn mặt và nhận kết quả dự đoán đồng thời cho ba tác vụ:
    - Chủng tộc (Race)
    - Giới tính (Gender)
    -  Nhóm tuổi (Age)

- `model_def.py`: Chứa định nghĩa kiến trúc Multi-Task Learning được sử dụng trong quá trình huấn luyện và suy luận, bao gồm backbone Vision Transformer (ViT-Small) và các nhánh phân loại cho từng tác vụ.

- `ViT_Small_best.pth`: File checkpoint của mô hình ViT-Small có hiệu năng tốt nhất, được sử dụng để nạp trọng số và thực hiện suy luận trong ứng dụng.

Ứng dụng được phát triển nhằm minh họa khả năng triển khai mô hình học sâu vào một hệ thống suy luận thực tế. Người dùng có thể tải ảnh khuôn mặt thông qua giao diện web và nhận kết quả dự đoán đồng thời cho ba tác vụ Race, Gender và Age mà không cần thực hiện lại quá trình huấn luyện mô hình.

---

## report/

Chứa file báo cáo định dạng PDF.

---
# 4. Khám phá dữ liệu (EDA)
- Trước khi huấn luyện, project thực hiện phân tích khám phá dữ liệu (Exploratory Data Analysis - EDA) gồm các bước:
    - Thống kê phân phối của ba thuộc tính Race, Gender và Age trên tập huấn luyện và tập kiểm định.
    - Trực quan hóa phân bố nhãn bằng biểu đồ cột.
    - Hiển thị các ảnh mẫu đại diện cho từng nhóm chủng tộc cùng với nhãn giới tính và độ tuổi tương ứng.
    - Kiểm tra sự tồn tại của giá trị thiếu (Missing Values) sau khi ánh xạ nhãn.
    - Minh họa các phép biến đổi dữ liệu (Data Augmentation).

---
# 5. Tiền xử lý dữ liệu
- Dữ liệu được tiền xử lý trước khi đưa vào mô hình theo các bước sau:
    - Tự động tải bộ dữ liệu FairFace từ Kaggle thông qua thư viện kagglehub.
    - Sử dụng toàn bộ tập Validation và lấy ngẫu nhiên 40% tập Train bằng Stratified Sampling theo thuộc tính Race nhằm giảm thời gian huấn luyện nhưng vẫn duy trì phân phối lớp ban đầu.
    - Chuyển đổi các nhãn dạng chuỗi (Race, Gender, Age) sang chỉ số số nguyên để phục vụ quá trình huấn luyện.
    - Thiết lập seed = 42 cho Python, NumPy và PyTorch nhằm tăng khả năng tái lập kết quả thực nghiệm.
    - Tính toán Class Weights cho từng tác vụ (Race, Gender, Age) bằng phương pháp balanced để giảm ảnh hưởng của mất cân bằng dữ liệu.

- Data Augmentation
    - Đối với tập huấn luyện, các phép biến đổi dữ liệu được áp dụng bao gồm:
        - Resize
        - RandomCrop
        - RandomHorizontalFlip
        - ColorJitter
        - RandomRotation
        - ToTensor
        - Normalize theo thống kê của ImageNet
    - Đối với tập kiểm định, chỉ áp dụng:
        - Resize
        - ToTensor
        - Normalize theo thống kê của ImageNet
- Sau khi tiền xử lý, dữ liệu được đóng gói thành các Dataset và DataLoader riêng cho từng kiến trúc, với kích thước ảnh (img_size) và kích thước lô (batch_size) được cấu hình tương ứng cho từng mô hình.

---
# 6. Yêu cầu hệ thống
- Python 3.10+
- PyTorch
- Google Colab hoặc máy có GPU CUDA

---
# 7. Cài đặt môi trường

Để chuẩn bị môi trường chạy đồ án, cài đặt các thư viện cần thiết được liệt kê trong requirements.txt bằng lệnh:

```bash
pip install -r requirements.txt
```

Khi sử dụng Google Colab, nhiều thư viện phổ biến như PyTorch đã được cài đặt sẵn. Tuy nhiên, để đảm bảo môi trường đầy đủ và thống nhất với dự án, vẫn nên cài đặt các phụ thuộc trong requirements.txt trước khi chạy notebook.

---

# 8. Cách sử dụng

Để tái đóng gói kết quả, phân tích đồ thị và xuất báo cáo một cách chính xác nhất, quy trình thực hiện cần tuân thủ nghiêm ngặt theo các bước tuần tự dưới đây:

1. Chuẩn bị bộ dữ liệu FairFace.
- Project sử dụng thư viện kagglehub để tự động tải bộ dữ liệu FairFace từ Kaggle. 

- Khi chạy notebook, dữ liệu sẽ được tải về và đường dẫn được cập nhật tự động trong biến CONFIG, do đó người dùng không cần tải thủ công bộ dữ liệu hoặc chỉnh sửa đường dẫn trong điều kiện sử dụng thông thường.

- Sau khi tải dữ liệu, chương trình sử dụng:
    - Toàn bộ tập validation. 
    - Ngẫu nhiên **40% tập huấn luyện gốc** bằng phương pháp **stratified sampling theo thuộc tính Race** nhằm giảm thời gian huấn luyện nhưng vẫn giữ phân phối lớp của dữ liệu ban đầu.

2. Huấn luyện Thử nghiệm từng Kiến trúc.
- Mở và chạy lần lượt:
    - 01_CustomCNN.ipynb 
    - 02_MobileNetV3_Large.ipynb 
    - 03_EfficientNet_B0.ipynb 
    - 04_ViT_Small.ipynb

- Mỗi notebook sẽ lặp qua danh sách các cấu hình thực nghiệm để tìm ra siêu tham số tối ưu.

3. Chạy Tổng hợp và So sánh Cuối cùng.
- Sau khi đã xác định và sinh ra đầy đủ kết quả của mô hình tốt nhất từ bốn kiến trúc ở Bước 2, hãy mở và tiến hành chạy notebook: FairFace_Main.ipynb.

- Notebook sẽ:
    - Khởi tạo và huấn luyện lại bốn mô hình (`CustomCNN`, `MobileNetV3_Large`, `EfficientNet_B0` và `ViT_Small`) với bộ siêu tham số tối ưu đã được lựa chọn.
    - Lưu checkpoint của từng mô hình vào thư mục `models/`, bao gồm: 
        - Mô hình có hiệu năng tốt nhất trên tập validation (`*_best.pth`).
        - Mô hình tại epoch cuối cùng (`*_last.pth`) .
    - Ghi nhận và lưu lịch sử huấn luyện cũng như các chỉ số đánh giá của từng mô hình vào thư mục `results/Main/`.
    - Tạo và lưu bảng so sánh hiệu năng tổng thể giữa các kiến trúc vào tệp model_comparison_multitask.csv.
    - Kích hoạt và sinh mã log TensorBoard lưu trữ vào thư mục runs/.
    - Tự động vẽ và xuất toàn bộ các biểu đồ phân tích sâu (Đường cong Loss, Accuracy, đồ thị so sánh, ma trận nhầm lẫn Confusion Matrix, Error Analysis) vào thư mục figures/.

4. Khởi chạy giao diện giám sát trực quan TensorBoard
- Trên máy cục bộ, để so sánh trực quan tiến trình hội tụ và các đường cong huấn luyện của cả 4 mô hình xuất sắc nhất, thực hiện lệnh sau trên Terminal: 
`tensorboard --logdir=runs/tensorboard`
    - Mở trình duyệt web và truy cập vào địa chỉ mạng cục bộ được cung cấp (thường là http://localhost:6006/) để xem biểu đồ.

- Ngoài ra, khi sử dụng Colab thì có thể chạy lệnh sau để dễ theo dõi log ngay trong notebook:
`%load_ext tensorboard`
`%tensorboard --logdir /content/drive/MyDrive/project/runs/tensorboard`
    - Có thể thay bằng đường dẫn tương ứng nếu thư mục project được lưu ở vị trí khác.

5. Triển khai và chạy ứng dụng demo Streamlit
Sau khi hoàn thành quá trình huấn luyện và thu được checkpoint của mô hình tốt nhất, có thể triển khai ứng dụng web minh họa bằng Streamlit để kiểm tra khả năng suy luận của mô hình trên các ảnh khuôn mặt mới.

- Di chuyển đến thư mục ứng dụng:
`cd fairface_demo`

- Nếu môi trường chưa được cài đặt các thư viện của dự án, hãy chạy lệnh sau để cài đặt các phụ thuộc từ requirements.txt:
`pip install -r ../requirements.txt`
    - Nếu đã cài đặt các thư viện theo requirements.txt ở Bước 7 thì có thể bỏ qua bước này.

- Khởi động ứng dụng:
`streamlit run app.py`

- Sau khi ứng dụng được khởi chạy thành công, trình duyệt sẽ tự động mở tại địa chỉ:
`http://localhost:8501`

- Tại giao diện ứng dụng:
    - Tải lên một ảnh khuôn mặt định dạng JPG, JPEG hoặc PNG.
    - Hệ thống sẽ tự động thực hiện tiền xử lý ảnh đầu vào.
    - Mô hình ViT-Small được huấn luyện trước sẽ được nạp từ file checkpoint ViT_Small_best.pth.
    - Ứng dụng sẽ trả về đồng thời ba kết quả dự đoán:
        - Chủng tộc (Race)
        - Giới tính (Gender)
        - Nhóm tuổi (Age)
    - Đồng thời hiển thị độ tin cậy (Confidence Score) và các dự đoán có xác suất cao nhất.

Lưu ý:

    - Ứng dụng Streamlit được thiết kế để chạy trên môi trường cục bộ (VS Code hoặc Terminal), tách biệt với môi trường huấn luyện Google Colab.
    - Trước khi chạy ứng dụng, cần đảm bảo file checkpoint ViT_Small_best.pth đã được đặt đúng vị trí trong thư mục fairface_demo/.

---

# 9. Kết quả đầu ra

- Sau khi chạy chương trình, các kết quả sẽ được lưu tại:

* `models/`: Trọng số của các mô hình tốt nhất.
* `results/`: Chỉ số đánh giá và lịch sử huấn luyện của các mô hình tốt nhất.
* `results/Main/`: Kết quả tổng hợp và lịch sử huấn luyện của bốn mô hình được chọn để so sánh.
* `runs/tensorboard`: Log TensorBoard của bốn mô hình tốt nhất.
* `figures/`: Các biểu đồ và hình ảnh trực quan phục vụ phân tích kết quả.
* `fairface_demo/app.py`: Ứng dụng Streamlit dùng để suy luận và trình diễn mô hình.
* `fairface_demo/model_def.py`: Định nghĩa kiến trúc Multi-Task Learning phục vụ huấn luyện và triển khai.

---

# 10. Demo ứng dụng Streamlit

- Bên cạnh quá trình huấn luyện và đánh giá mô hình, đồ án còn triển khai một ứng dụng web minh họa bằng Streamlit nhằm kiểm chứng khả năng ứng dụng thực tế của mô hình ViT-Small.

- Ứng dụng cho phép:
    - Tải lên ảnh khuôn mặt.
    - Tự động tiền xử lý dữ liệu đầu vào.
    - Thực hiện suy luận bằng mô hình đã huấn luyện.
    - Dự đoán đồng thời:
        - Chủng tộc (Race)
        - Giới tính (Gender)
        - Nhóm tuổi (Age)
    - Hiển thị xác suất dự đoán (Confidence Score).
    - Hiển thị Top-3 dự đoán Race có xác suất cao nhất.

- Người dùng có thể tương tác trực tiếp với mô hình thông qua giao diện web mà không cần chạy notebook huấn luyện.
