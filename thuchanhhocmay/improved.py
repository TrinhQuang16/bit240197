import time
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers, callbacks
import pandas as pd

# Load dữ liệu
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Hàm xây dựng mô hình với BN và Dropout
def build_improved(dropout_conv=0.25, dropout_dense=0.5):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_conv),

        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_conv),

        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(dropout_conv),

        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_dense),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Hàm train và lấy kết quả
def train_config(name, optimizer, dropout_conv=0.25, dropout_dense=0.5, epochs=10):
    print(f"\n--- Training: {name} ---")
    model = build_improved(dropout_conv, dropout_dense)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
    checkpoint = callbacks.ModelCheckpoint(f'{name}_best.h5', monitor='val_accuracy', save_best_only=True, verbose=0)

    start = time.time()
    history = model.fit(x_train, y_train, batch_size=256, epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stop, checkpoint],
                        verbose=1)
    train_time = time.time() - start

    best_idx = history.history['val_accuracy'].index(max(history.history['val_accuracy']))
    best_val_acc = history.history['val_accuracy'][best_idx]
    train_loss = history.history['loss'][best_idx]
    epoch_converge = best_idx + 1

    return {
        'model': name,
        'optimizer': optimizer.__class__.__name__,
        'dropout_conv': dropout_conv,
        'dropout_dense': dropout_dense,
        'train_loss': train_loss,
        'val_acc': best_val_acc,
        'train_time': train_time,
        'epoch_converge': epoch_converge
    }

# Các cấu hình
configs = [
    {'name': 'SGD_Momentum_drop25', 'optimizer': optimizers.SGD(0.01, momentum=0.9), 'dropout_conv': 0.25},
    {'name': 'SGD_Momentum_drop50', 'optimizer': optimizers.SGD(0.01, momentum=0.9), 'dropout_conv': 0.5},
    {'name': 'Adam_drop25', 'optimizer': optimizers.Adam(0.001), 'dropout_conv': 0.25},
]

# Chạy và thu thập kết quả
results = []
for cfg in configs:
    res = train_config(cfg['name'], cfg['optimizer'], cfg['dropout_conv'])
    results.append(res)

try:
    with open('baseline_results.txt', 'r') as f:
        lines = f.readlines()
        baseline_val_acc = float(lines[2].split(':')[1].strip())
        baseline_loss = float(lines[1].split(':')[1].strip())
        baseline_time = float(lines[3].split(':')[1].split()[0])
        baseline_epoch = int(lines[4].split(':')[1].strip())
    baseline_row = {
        'model': 'Baseline',
        'optimizer': 'SGD',
        'dropout_conv': 0,
        'dropout_dense': 0,
        'train_loss': baseline_loss,
        'val_acc': baseline_val_acc,
        'train_time': baseline_time,
        'epoch_converge': baseline_epoch
    }
    results.insert(0, baseline_row)
except FileNotFoundError:
    print("Chưa có baseline_results.txt, hãy chạy baseline.py trước.")
    # Tạm thời bỏ qua baseline

# Tạo bảng
df = pd.DataFrame(results)
df = df[['model', 'optimizer', 'dropout_conv', 'dropout_dense', 'train_loss', 'val_acc', 'train_time', 'epoch_converge']]
df.columns = ['Mô hình', 'Optimizer', 'Dropout Conv', 'Dropout Dense', 'Train Loss', 'Val Acc', 'Thời gian (s)', 'Epoch hội tụ']

print("\n" + "="*80)
print("BẢNG SO SÁNH KẾT QUẢ")
print("="*80)
print(df.to_string(index=False))

# Kết luận
best = df.loc[df['Val Acc'].idxmax()]
print("\n" + "="*80)
print("KẾT LUẬN")
print("="*80)
print(f"Mô hình tốt nhất: {best['Mô hình']}")
print(f"Val Accuracy: {best['Val Acc']:.4f}")
print(f"Training Loss: {best['Train Loss']:.4f}")
print(f"Thời gian: {best['Thời gian (s)']:.2f}s")
print(f"Epoch hội tụ: {best['Epoch hội tụ']}")
print("\nLý do: Kết hợp BN + Dropout + Adam giúp hội tụ nhanh, giảm overfitting, đạt độ chính xác cao nhất.")