"""
baseline.py
Chạy mô hình CNN cơ bản trên CIFAR-10, lưu kết quả vào file 'baseline_results.txt'
"""

import time
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers, callbacks
import numpy as np

# 1. Load dữ liệu
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. Xây dựng mô hình cơ bản (không BN, không Dropout)
def build_baseline():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = build_baseline()
model.compile(optimizer=optimizers.SGD(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. Callbacks: EarlyStopping để xác định epoch hội tụ, ModelCheckpoint để lưu tốt nhất
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
checkpoint = callbacks.ModelCheckpoint('baseline_best.h5', monitor='val_accuracy', save_best_only=True, verbose=0)

# 4. Train và đo thời gian
print("Bắt đầu train baseline...")
start_time = time.time()
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=256,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop, checkpoint],
                    verbose=1)
train_time = time.time() - start_time

# 5. Lấy kết quả tốt nhất (từ early stopping)
best_epoch = early_stop.stopped_epoch - early_stop.patience + 1  # epoch thực tế
if best_epoch < 1:  # nếu không early stop
    best_epoch = len(history.history['val_accuracy'])
best_val_acc = max(history.history['val_accuracy'])
# Lấy training loss tại epoch tốt nhất (index)
best_idx = history.history['val_accuracy'].index(max(history.history['val_accuracy']))
train_loss_at_best = history.history['loss'][best_idx]

# 6. Ghi kết quả
with open('baseline_results.txt', 'w') as f:
    f.write("=== KẾT QUẢ BASELINE (không BN, không Dropout, SGD) ===\n")
    f.write(f"Training Loss (tại epoch tốt nhất): {train_loss_at_best:.4f}\n")
    f.write(f"Validation Accuracy tốt nhất: {best_val_acc:.4f}\n")
    f.write(f"Thời gian train: {train_time:.2f} giây\n")
    f.write(f"Số epoch hội tụ (epoch đạt accuracy cao nhất): {best_epoch+1}\n")

print("Đã lưu kết quả vào baseline_results.txt")
print(f"Val Accuracy: {best_val_acc:.4f}, Time: {train_time:.2f}s, Epoch: {best_epoch+1}")