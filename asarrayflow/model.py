from layers import Layer, InputLayer
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    GPU_AVAILABLE = False
    cp = None

class Model:

    def __init__(self, inputs, outputs):
        if isinstance(inputs, (list, tuple)):
            for layer in inputs:
                if not isinstance(layer, InputLayer):
                    raise TypeError('Inputs should be instances of InputLayer class')
            self.inputs = inputs
        else:
            if not isinstance(inputs, InputLayer):
                raise TypeError('Inputs should be instances of InputLayer class')
            self.inputs = [inputs]
        if isinstance(outputs, (list, tuple)):
            for layer in outputs:
                if not isinstance(layer, Layer):
                    raise TypeError('Outputs should be NN Layers')
            self.outputs = outputs
        else:
            if not isinstance(outputs, Layer):
                raise TypeError('Outputs should be NN Layers')
            self.outputs = [outputs]
        self.execution_order = []
        self._compiled = False

    def compile(self):
        if self._compiled:
            raise RuntimeError('Model is compiled already.')

        layer_depth = {}

        def traverse(layer, depth=0):
            if layer in layer_depth:
                layer_depth[layer] = max(layer_depth[layer], depth)
            else:
                layer_depth[layer] = depth

            if hasattr(layer, "_prev") and layer._prev is not None:
                if isinstance(layer._prev, (list, tuple)):
                    for prev in layer._prev:
                        traverse(prev, depth+1)
                else:
                    traverse(layer._prev, depth+1)

        for output_layer in self.outputs:
            traverse(output_layer)

        self.execution_order = sorted(layer_depth, key=lambda l: -layer_depth[l])

        missing_inputs = [inp for inp in self.inputs if inp not in layer_depth]

        if missing_inputs:
            raise RuntimeError(f'Found disconnected inputs: {missing_inputs}')
        
        self._compiled = True

    def reset(self):
        for layer in self.execution_order:
            layer.reset()

    def predict(self, x):
        self.reset()

        if isinstance(x, dict):
            for input_layer in self.inputs:
                input_layer.forward(x[input_layer])
        
        else:
            if len(self.inputs) != 1:
                raise ValueError('Model expects multiple inputs, but got single input')
            self.inputs[0].forward(x)

        for layer in self.execution_order:
            if layer._computed:
                continue
            if layer._prev is not None:
                if isinstance(layer._prev, (list, tuple)):
                    inputs = [prev.output for prev in layer._prev]
                else:
                    inputs = layer._prev.output

            layer.forward(inputs)

        return [out.output for out in self.outputs] if len(self.outputs) > 1 else self.outputs[0].output

    def fit(self, x, y, loss, opimizer, epochs=1, batch_size=None, verbose=True):
        if x.shape[0] != y.shape[0]:
            raise ValueError('Input set size does not fit output set size')
        backend = cp if GPU_AVAILABLE else np
        data_size = x.shape[0]

        if batch_size is None:
            batch_size = data_size

        for epoch in range(epochs):

            indices = backend.arange(data_size)
            backend.random.shuffle(indices)

            for start in range(0, data_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                x_batch = x[batch_idx]
                y_batch = y[batch_idx]

                y_predict = self.predict(x_batch)

                s_loss, d_loss = loss(y_predict, y_batch)

                for output_layer in self.outputs:
                    output_layer.backward(d_loss)
                
                for layer in self.execution_order:
                    if hasattr(layer, 'apply_gradients'):
                        layer.apply_gradients(opimizer)

            if verbose:
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {s_loss:.4f}')
