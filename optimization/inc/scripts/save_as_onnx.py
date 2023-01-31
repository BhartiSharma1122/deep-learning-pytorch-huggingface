import torch
from transformers import AutoModelForSequenceClassification,AutoConfig
from neural_compressor.experimental.export import torch_to_fp32_onnx, torch_to_int8_onnx

model_name_or_path="bert-base-uncased"
output_dir="test_zero"
sample_inputs = torch.randn(1, 3, 224, 224)
dynamic_axes = {}
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
int8_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
model.load_state_dict(torch.load(output_dir + "/model_best.pth")["state_dict"])

from neural_compressor.training import prepare_compression
compression_manager = prepare_compression(model, combs)

