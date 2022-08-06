
NHÓM 04: NHẬN DIỆN CẢM XÚC - MÔ HÌNH KẾ THỪA
===

## Mục Lục

[TOC]

## Cấu trúc đồ án

- **AMOL**: thư mục chứa các hàm phụ trợ và kiến trúc mô hình.
- **config**: thư mục chứa cấu hình cơ bản của tập dữ liệu, các đường dẫn tới các thư mục chứa dữ liệu.
- **checkpoints**: thư mục chứa mô hình tốt nhất đã huấn luyện làm đầu vào cho các chương trình nhận diện.
- **fer2013**:
- ```build_dataset.py```: sử dụng để đọc và tách dữ liệu chính thành các tập con.
- ```emotion_detector.py```: sử dụng để phát hiện khuôn mặt trong video hoặc camera dựa trên mô hình đã huấn luyện.
- ```emotion_detector_image.py```: có chức năng tương tự như file trên nhưng sử dụng trên ảnh tĩnh.
- ```haarcascade_frontalface_default.xml```: file chứa thông tin đặc trưng để làm đầu vào cho bộ phát hiện khuôn mặt sử dụng thư viện opencv.
- ```train_recognizer.py```: dùng để huấn luyện mô hình.
- ```test_recognizer.py```: dùng để đánh giá mô hình.

## Chuẩn bị dữ liệu

Sử dụng file ```build_dataset.py``` để tạo ra các file ```.hdf5``` làm dữ liệu cho quá trình huấn luyện và đánh giá mô hình.
```bash
python build_dataset.py
```

## Huấn luyện mô hình
---
Phần này, trình bày các câu lệnh để huấn luyện mô hình, chi tiết thể hiện trong báo cáo.
1. Thực nghiệm 1
```bash
CUDA_VISIBLE_DEVICES=1 python train_recognizer.py --checkpoints checkpoints/exp1 -opt "SGD"

CUDA_VISIBLE_DEVICES=1 python train_recognizer.py --checkpoints checkpoints/exp1 -opt "SGD" -lr 1e-3 --start 20 --model checkpoints/exp1/epoch_20.hdf5

CUDA_VISIBLE_DEVICES=1 python train_recognizer.py --checkpoints checkpoints/exp1 -opt "SGD" -lr 1e-4 --start 40 --model checkpoints/exp1/epoch_40.hdf5
```

![Quá trình hội tụ hàm lỗi](https://github.com/https://github.com/stark4079/Emotion_Recognition/blob/main/Emotion_Recognition_Keras/fer2013/output/exp1/exp1_loss.png?raw=true)

![Quá trình hội tụ độ chính xác](https://github.com/https://github.com/stark4079/Emotion_Recognition/blob/main/Emotion_Recognition_Keras/fer2013/output/exp1/exp1_acc.png?raw=true)

2. Thực nghiệm 2
```bash
CUDA_VISIBLE_DEVICES=1 python train_recognizer.py --checkpoints checkpoints/exp2 -lr 1e-3 --epoch 30

CUDA_VISIBLE_DEVICES=1 python train_recognizer.py --checkpoints checkpoints/exp2 -lr 1e-4 --epoch 15 --start 30 --model checkpoints/exp2/epoch_30.hdf5
```
![Quá trình hội tụ hàm lỗi](https://github.com/https://github.com/stark4079/Emotion_Recognition/blob/main/Emotion_Recognition_Keras/fer2013/output/exp2/exp2_loss.png?raw=true)

![Quá trình hội tụ độ chính xác](https://github.com/https://github.com/stark4079/Emotion_Recognition/blob/main/Emotion_Recognition_Keras/fer2013/output/exp2/exp2_acc.png?raw=true)

3. Thực nghiệm 3
```bash
CUDA_VISIBLE_DEVICES=1 python train_recognizer.py --checkpoints checkpoints/exp3 -lr 1e-3 --epoch 40

CUDA_VISIBLE_DEVICES=1 python train_recognizer.py --checkpoints checkpoints/exp3 -lr 1e-4 --epoch 20 --start 40 --model checkpoints/exp3/epoch_40.hdf5

CUDA_VISIBLE_DEVICES=1 python train_recognizer.py --checkpoints checkpoints/exp3 -lr 1e-5 --epoch 15 --start 60 --model checkpoints/exp3/epoch_60.hdf5
```
![Quá trình hội tụ hàm lỗi](https://github.com/https://github.com/stark4079/Emotion_Recognition/blob/main/Emotion_Recognition_Keras/fer2013/output/exp3/exp3_loss.png?raw=true)

![Quá trình hội tụ độ chính xác](https://github.com/https://github.com/stark4079/Emotion_Recognition/blob/main/Emotion_Recognition_Keras/fer2013/output/exp3/exp3_acc.png?raw=true)

4. Thực nghiệm 4
Xây dựng lại dữ liệu đầu vào và tiến hành thực nghiệm.
```bash
python build_dataset.py

CUDA_VISIBLE_DEVICES=1 python train_recognizer.py --checkpoints checkpoints/exp4 -lr 1e-3 --epoch 40

CUDA_VISIBLE_DEVICES=1 python train_recognizer.py --checkpoints checkpoints/exp4 -lr 1e-4 --epoch 20 --start 40 --model checkpoints/exp4/epoch_40.hdf5

CUDA_VISIBLE_DEVICES=1python train_recognizer.py --checkpoints checkpoints/exp4 -lr 1e-5 --epoch 15 --start 60 --model checkpoints/exp4/epoch_60.hdf5
```
![Quá trình hội tụ hàm lỗi](https://github.com/https://github.com/stark4079/Emotion_Recognition/blob/main/Emotion_Recognition_Keras/fer2013/output/exp4/exp4_loss.png?raw=true)

![Quá trình hội tụ độ chính xác](https://github.com/https://github.com/stark4079/Emotion_Recognition/blob/main/Emotion_Recognition_Keras/fer2013/output/exp4/exp4_acc.png?raw=true)

## Đánh giá mô hình
Để đánh giá mô hình, ta chạy câu lệnh bên dưới để đọc mô hình đã huấn luyện tốt nhất.
```bash
python test_recognizer.py --model checkpoints/epoch.hdf5
```
Mô hình đạt độ chính xác **66.43\%** trên tập kiểm tra.

## Ứng dụng mô hình dự đoán cảm xúc
---
1. Sử dụng với camera
```bash
python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch.hdf5 
```
2. Sử dụng với ảnh tĩnh
```bash
python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch.hdf5 --image path/to/your/image.*{jpg, png,...}
```

3. Sử dụng với video
```bash
python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch.hdf5 --video path/to/your/video.mp4
```

## Kết luận
Chúng tôi đã thực nghiệm lại tất cả các thực nghiệm trên và kết quả khá tương đồng nhưng đối với kết quả sau cùng đạt 66.10\% so với kết quả trình bày trong sách[2] là 66.96/% thấp hơn 0.86\%.

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

