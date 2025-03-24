# AsArrayFlow

**AsArrayFlow** is a minimalist deep learning library built from scratch using **NumPy and CuPy** — designed to be simple, extensible, and educational. It mimics some of the design philosophy of frameworks like Keras or PyTorch, but without symbolic tensors or bloated abstraction layers.

---

## 🚀 Features

- 🧱 Fully modular layer system (`InputLayer`, `Dense`, etc.)
- 🔁 Dynamic graph building via Python calls
- 🧠 Lazy weight initialization and automatic shape inference
- 🔗 Supports skip connections and multi-input architectures
- ⚡ GPU acceleration via CuPy (optional)
- ✅ Pure Python, NumPy-first — no extra dependencies

---

## 📦 Installation

```bash
git clone https://github.com/alekselazar/AsArrayFlow.git
cd AsArrayFlow
pip install -e .
```

### CuPy on Windows
CuPy **is supported on Windows**, but you must install the correct wheel for your CUDA version. For example:

```bash
# If you have CUDA 11.8 installed:
pip install cupy-cuda118
```

> See [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for other CUDA versions.

If you're on Linux or WSL:
```bash
pip install cupy  # Prebuilt wheels available for most CUDA versions
```

---

## 🧠 Quick Example

```python
from core.layers import InputLayer, Dense
from core.model import Model
import numpy as np

# Define model
input_layer = InputLayer((None, 10))
hidden = Dense(16)(input_layer)
output = Dense(1)(hidden)
model = Model(input_layer, output)
model.compile()

# Predict
X = np.random.randn(32, 10)
y = model.predict(X)
print(y.shape)  # (32, 1)
```

---

## 📚 Roadmap

- [x] Core layer system
- [x] Model graph compilation and `predict()`
- [x] Lazy shape inference
- [ ] `fit()` method with loss and optimizers
- [ ] Built-in layers: `Concatenate`, `ReLU`, `Softmax`, etc.
- [ ] Convolutional and recurrent layers
- [ ] Dataset loading utils
- [ ] Model saving/loading

---

## 🛠 Project Structure

```
core/
├── layers.py        # Core layer classes
├── model.py         # Model execution logic
setup.py             # Install script
LICENSE              # MIT License
README.md            # You're reading it
```

---

## 🤝 Contributing

Want to contribute? Suggestions and PRs are welcome. This project is ideal if you're:
- Learning how deep learning frameworks work under the hood
- Curious about NumPy/CuPy-level GPU ops
- Wanting to tinker with custom models without TensorFlow/PyTorch overhead

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](./LICENSE) for details.
