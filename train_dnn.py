import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist

# === STEP 1: 訓練模型 (使用 TensorFlow) ===

# 讀取資料
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 建立模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.1)

# TensorFlow 測試集準確率
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"[TensorFlow] Test accuracy: {test_acc:.4f}")

# === STEP 2: 儲存為 .json + .npz (符合 nn_predict.py 格式) ===

# 建構 JSON 架構
model_arch = []
weights = {}
dense_idx = 0

for i, layer in enumerate(model.layers):
    if isinstance(layer, Flatten):
        model_arch.append({
            "name": f"flatten_{i}",
            "type": "Flatten",
            "config": {},
            "weights": []
        })
    elif isinstance(layer, Dense):
        W, b = layer.get_weights()
        weights[f"dense_{dense_idx}_kernel"] = W
        weights[f"dense_{dense_idx}_bias"] = b
        model_arch.append({
            "name": f"dense_{dense_idx}",
            "type": "Dense",
            "config": {"activation": layer.activation.__name__},
            "weights": [f"dense_{dense_idx}_kernel", f"dense_{dense_idx}_bias"]
        })
        dense_idx += 1

# 儲存 json
with open("fashion_mnist.json", "w") as f:
    json.dump(model_arch, f)

# 儲存權重
np.savez("fashion_mnist.npz", **weights)

print("已儲存模型為 fashion_mnist.json 與 fashion_mnist.npz")

# === STEP 3: 使用 nn_predict.py 做 NumPy 推論測試 ===

from nn_predict import nn_inference

# 讀取剛剛儲存的模型
with open("fashion_mnist.json") as f:
    loaded_arch = json.load(f)
loaded_weights = dict(np.load("fashion_mnist.npz"))

# 預測前 1000 筆測試資料
x_test_input = x_test[:1000]
y_test_input = y_test[:1000]

# 用 nn_predict 推論
pred_logits = nn_inference(loaded_arch, loaded_weights, x_test_input)
y_pred = np.argmax(pred_logits, axis=1)

# 計算準確率
acc = np.mean(y_pred == y_test_input)
print(f"[NumPy 推論] Accuracy on 1000 test samples: {acc:.4f}")
