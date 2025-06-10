import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist

# 引入 nn_forward_h5
from nn_predict import nn_forward_h5

# === 載入資料 ===
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# === 模型設計與訓練 ===
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)

# 儲存為 .h5（可選）
model.save("fashion_mnist.h5")

# === 將架構與權重轉換為 JSON + NPZ ===
arch = []
weights = {}
layer_idx = 0

for layer in model.layers:
    if isinstance(layer, Flatten):
        arch.append({
            "name": f"flatten_{layer_idx}",
            "type": "Flatten",
            "config": {},
            "weights": []
        })
    elif isinstance(layer, Dense):
        w, b = layer.get_weights()
        weights[f"dense_{layer_idx}_kernel"] = w
        weights[f"dense_{layer_idx}_bias"] = b
        arch.append({
            "name": f"dense_{layer_idx}",
            "type": "Dense",
            "config": {"activation": layer.activation.__name__},
            "weights": [f"dense_{layer_idx}_kernel", f"dense_{layer_idx}_bias"]
        })
    layer_idx += 1

# 儲存
with open("fashion_mnist.json", "w") as f:
    json.dump(arch, f)

np.savez("fashion_mnist.npz", **weights)

# === 使用 nn_forward_h5 驗證推論 ===
# 載入剛剛存的 json 與 npz
with open("fashion_mnist.json") as f:
    model_arch = json.load(f)

npz = np.load("fashion_mnist.npz")
weights_dict = dict(npz)

# 取前 1000 筆做測試
x_test_subset = x_test[:1000]
y_test_subset = y_test[:1000]

# NumPy 推論
pred_logits = nn_forward_h5(model_arch, weights_dict, x_test_subset)
pred_label = np.argmax(pred_logits, axis=1)

acc = np.mean(pred_label == y_test_subset)
print(f"[NumPy 推論] Accuracy on 1000 test samples: {acc:.4f}")
