"""Minimal TensorRT runner that exchanges data with PyTorch tensors.

One engine per perception mode, each built for a fixed input shape, so the
executed graph is exactly what was profiled.
"""
from __future__ import annotations

from pathlib import Path

import torch


class TRTModule:
    def __init__(self, engine_path: str | Path, device: str = "cuda:0"):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)

        # trt.nptype() touches np.bool, removed in numpy >= 1.24, so map types directly.
        to_torch = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                    trt.DataType.INT8: torch.int8, trt.DataType.INT32: torch.int32,
                    trt.DataType.BOOL: torch.bool}

        self.buffers, self.bindings, self.input_idx, self.output_idx = [], [], [], []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = to_torch[self.engine.get_binding_dtype(i)]
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self.buffers.append(buf)
            self.bindings.append(int(buf.data_ptr()))
            (self.input_idx if self.engine.binding_is_input(i) else self.output_idx).append(i)

        self.input_shape = tuple(self.buffers[self.input_idx[0]].shape)
        self.input_dtype = self.buffers[self.input_idx[0]].dtype
        # TensorRT allocates its scratch outside the torch caching allocator, so this
        # is the only figure that reflects what the engine actually costs on device.
        self.device_mem_mb = (self.engine.device_memory_size
                              + sum(b.numel() * b.element_size() for b in self.buffers)) / 2**20

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        inp = self.buffers[self.input_idx[0]]
        inp.copy_(x.to(dtype=self.input_dtype, non_blocking=True))
        self.ctx.execute_v2(self.bindings)
        return self.buffers[self.output_idx[0]]
