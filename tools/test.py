import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    "D:\\PythonProjects\\PoseGaze\\models\\hub\\checkpoints\\yolox_x_8xb8-300e_humanart-a39d44ed.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print(f"Providers: {sess.get_providers()}")
print(f"Input name: {sess.get_inputs()[0].name}")
print(f"Input shape: {sess.get_inputs()[0].shape}")

# Test inference
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
output = sess.run(None, {sess.get_inputs()[0].name: dummy_input})
print("✓ Inference successful")
