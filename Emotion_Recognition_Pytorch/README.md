
NHÓM 04: NHẬN DIỆN CẢM XÚC - MÔ HÌNH ĐỀ XUẤT
==

## Mục Lục

[TOC]

## Cấu trúc đồ án

Đồ án sẽ bao gồm 4 folder chính:
- FER2013_VGG19: chứa file đã lưu tại epoch tốt nhất của model.
- data: chứa file fer2013.csv dùng cho việc train mà được lấy từ kaggle.
- face_detection: chứa một file dat pretrain nhận diện mặt người được lấy từ thư viện dlib.
- test_images: chứa các ảnh mặt người dùng cho dự đoán cảm xúc.
- Link dataset: https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data.
- Khi tải dữ liệu từ link trên, ta sẽ có 1 file tên fer2013.tar.gz, giải nén file này sẽ được 1 file tên fer2013.csv và file này sẽ được đặt vào folder data để tạo ra file data.h5 cho việc huấn luyện mô hình.

## Cài đặt môi trường

Tạo thêm 2 folder là cropped_images và results để lưu các nén ảnh cho việc dự đoán chính xác hơn và lưu lại kết quả đã dự đoán được. Tại thư mục ```Emotion_Recognition_Pytorch``` chạy các lệnh bên dưới:
```
cd test_images
mkdir cropped_images
mkdir results
```

## Chuẩn bị dữ liệu

Đầu tiên chạy ```file build_dataset``` chúng ta sẽ có 1 file ```data.h5``` trong folder ```data```, đây là file chia dữ liệu chính thành các tập con bao gồm train/valid/test từ file csv và được dùng cho việc huấn luyện và đánh giá mô hình.
```
python build_dataset.py
```

## Huấn luyện mô hình
Chạy file ```main.py``` để huấn luyện mô hình và tại mỗi epoch tốt nhất kết quả sẽ được lưu vào folder ```FER2013_VGG19```.

```
python main.py --lr 0.01

```

## Ứng dụng mô hình dự đoán cảm xúc
Chạy file ```make_pred``` để nhận diện cảm xúc trên ảnh mặt người, đầu vào có thể là ảnh một người hoặc nhiều người.  Ví dụ dự đoán cảm xúc trên tệp ```5.jpg``` 

```
python make_pred.py --input_img 'test_images/5.jpg' --output_img 'test_images/results/5.jpg'
```

## Kết Luận
Mô hình cho kết quả cải thiện hơn 3\% so với mô hình gốc và có thời gian huấn luyện và khả năng hội tụ tốt hơn mô hình trước đó, đạt 69.96\%.

## Tài liệu tham khảo
[1] Challenges in representation learning: Facial expression recognition challenge, URL: https://www.kaggle.com/c/challenges- in- representationlearning-facial-expression-recognition-challenge.

[2] Tim Dettmers et al. (2021), “8-bit Optimizers via Block-wise Quantization”, arXiv preprint arXiv:2110.02861.

[3] JostineHo, Jostineho/Mememoji: A facial expression classification system that recognizes 6 basic emotions: Happy, sad, surprise, fear, anger and neutral. URL:
https://github.com/JostineHo/mememoji.

[4] Davis E King (2015), “Max-margin object detection”, arXiv preprint arXiv:1502.00046.

[5] Paulius Micikevicius et al. (2017), “Mixed precision training”, arXiv preprint arXiv:1710.03740.

[6] Adrian Rosebrock, “11. Case Study: Emotion Recognition”, in: Deep learning for computer vision with python, Adrian Rosebrock, PyImageSearch.com, 2019, 141–163.


## Thông tin liên hệ

|       **Tên**      | **Mã số sinh viên** |           **Email**           |
|:------------------:|:-------------------:|:-----------------------------:|
|    Trần Xuân Lộc   |       18127131      | 18127131@student.hcmus.edu.vn |
|    Trần Đại Chí    |       18127070      | 18127070@student.hcmus.edu.vn |
| Võ Trần Quang Tuấn |       18127248      | 18127248@student.hcmus.edu.vn |
