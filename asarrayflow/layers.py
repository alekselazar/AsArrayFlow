import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    GPU_AVAILABLE = False
    cp = None

class Layer:
    """Base class for all layers."""
    def __init__(self):
        self.input = None
        self.output = None
        self.output_shape = None
        self.weights = None
        self.bias = None
        self._prev = None  # Store previous layer for gradient tracking
        self._computed = False
    
    def __call__(self, x):
        self._prev = x

        if self.weights is None and self.output_shape is None:
            self._initialize_weights(self._prev.output.shape)
            self.output_shape = self._compute_output_shape(self._prev.output.shape)
        else:
            raise TypeError('Layer already connected to another')

        return self
    
    def _initialize_weights(self, shape):
        raise NotImplementedError
    
    def _compute_outpute_shape(self, input_shape):
        raise NotImplementedError
    
    def forward(self, x):
        if len(x.shape) < 2:
            raise ValueError(f'Expected batch size in first dimention, got shape {x.shape}')
        
        batch_size = x.shape[0]
        self.input = x
        self.output = self.compute(x)

        if self.output.shape[0] != batch_size:
            self.reset()
            raise RuntimeError(f'Batck size mismatch on output, expected {batch_size}, got {self.output.shape[0]}')
        self._computed = True
        return self.output

    def compute(self, x):
        raise NotImplementedError
    
    def reset(self):
        self.input = None
        self.output = None
        self._computed = False

    def apply_gradients(self, optimizer):
        """Applies gradients using a given optimizer."""
        pass

class InputLayer(Layer):
    """Input layer that just passes data forward."""
    def __init__(self, input_shape, use_gpu=False):
        super().__init__()
        self.input_shape = (None, *input_shape)
        self.backend = cp if (use_gpu and GPU_AVAILABLE) else np

    def forward(self, x):
        self.output = x
        return self.output

    def backward(self, grad_output):
        if self._prev:
            self._prev.backward(grad_output)

    def _initialize_weights(self, shape):
        pass

    def _compute_outpute_shape(self, input_shape):
        return self.input_shape

class Dense(Layer):
    """Fully connected (Dense) layer."""
    def __init__(self, output_size, use_gpu=False):
        super().__init__()
        self.use_gpu = use_gpu
        self.backend = cp if (use_gpu and GPU_AVAILABLE) else np
        self.output_size = output_size
        self.weights = None
        self.biases = None
        self.grad_weights = None
        self.grad_biases = None

    def _initialize_weights(self, shape):
        if self.weights is not None:
            raise RuntimeError('Weights alredy initialized for this layer')
        input_size = shape[-1]
        self.weights = self.backend.random.randn(input_size, self.output_size) * 0.01
        self.biases = self.backend.zeros(self.output_size)
        self.grad_weights = self.backend.zeros_like(self.weights)
        self.grad_biases = self.backend.zeros_like(self.biases)

    def _compute_outpute_shape(self, input_shape):
        return (*input_shape[:-1], self.output_size)
    
    def compute(self, x):
        if isinstance(x, Layer):  # If input is a layer, use its output
            x = x.output
        self.input = x
        original_shape = x.shape

        if len(original_shape) > 2:
            x_reshaped = x.reshape(-1, original_shape[-1])
            output = self.backend.dot(x_reshaped, self.weights) + self.biases
            return output.reshape(*original_shape[:-1], self.output_size)
        else:
            output = self.backend.dot(x, self.weights) + self.biases
            return output

    def backward(self, grad_output):
        """Computes gradients for backpropagation and propagates to previous layer."""
        self.grad_weights = self.backend.dot(self.input.T, grad_output)  # dL/dW
        self.grad_biases = self.backend.sum(grad_output, axis=0)  # dL/db
        grad_input = self.backend.dot(grad_output, self.weights.T)  # dL/dX
        
        # Recursively call backward on the previous layer
        if self._prev:
            self._prev.backward(grad_input)
    
    def apply_gradients(self, optimizer):
        """Uses an optimizer to update weights and biases."""
        optimizer.update(self.weights, self.grad_weights)
        optimizer.update(self.biases, self.grad_biases)
