NHÓM 04: EMOTION RECOGNITION - NHẬN DIỆN CẢM XÚC
===

## Table of Contents

[TOC]

## Cài đặt môi trường
---
1. Khởi tạo môi trường ảo
Để cài đặt môi trường ảo ```mypython```, ta cần chạy các câu lệnh sau:
```bash
virtualenv mypython
```
2. Kích hoạt môi trường ảo
Để kích hoạt môi trường ảo, ta chạy các câu lệnh sau:
```bash
source mypython/Scripts/activate
```
**Lưu ý:** Đối với hệ điều hành linux sử dụng câu lệnh sau để kích hoạt môi trường:
```bash
source mypython/bin/activate
```
3. Cài đặt các thư viện cần thiết
Cài các thư viện đã được liệt kê sẵn trong file ```requirements.txt``` bằng câu lệnh bên dưới:

```bash
pip install -r requirements.txt
```

## Dữ liệu bài toán
Dữ liệu bài toán FER2013 lấy từ một cuộc thi trên Kaggle[1] bao gồm 23709 mẫu với thông tin nhãn, mảng các điểm ảnh xám được làm phẳng và mục đích sử dụng của mẫu đó. Tập dữ liệu chứa 7 loại cảm xúc khác nhau. Nhiệm vụ là dự đoán chính xác nhãn của cảm xúc từ dữ liệu các điểm ảnh đã cho trước đó.

Dữ liệu sau khi tải về giải nén ta theo cấu trúc cây đề cập trong báo cáo.
## Mô hình kế thừa
Cài đặt lại mô hình trong sách và thực hiện 4 thực nghiệm chính. Chi tiết thể hiện ở thư mục [Emotion_Recognition_Keras](https://).

## Mô hình đề xuất
Đưa ra mô hình đề xuất và kết quả thực nghiệm cho thấy sự cải thiện về độ chính xác và thời gian huấn luyện. Chi tiết thể hiện ở thư mục [Emotion_Recognition_Pytorch](https://).


## Tài liệu tham khảo

[1] Challenges in representation learning: Facial expression recognition challenge,
URL: https://www.kaggle.com/c/challenges- in- representationlearning-facial-expression-recognition-challenge.

[2] Adrian Rosebrock, “11. Case Study: Emotion Recognition”, in: Deep learning for computer vision with python, Adrian Rosebrock, PyImageSearch.com,
2019, 141–163.


## Thông tin liên hệ

|       **Tên**      | **Mã số sinh viên** |           **Email**           |
|:------------------:|:-------------------:|:-----------------------------:|
|    Trần Xuân Lộc   |       18127131      | 18127131@student.hcmus.edu.vn |
|    Trần Đại Chí    |       18127070      | 18127070@student.hcmus.edu.vn |
| Võ Trần Quang Tuấn |       18127248      | 18127248@student.hcmus.edu.vn |
