import torch
import torch.nn as nn


input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)


# 加载模型
loaded_checkpoint = torch.load("08_3_checkpoint.pth")
model.load_state_dict(loaded_checkpoint["model_state"])
epoch = loaded_checkpoint["epoch"]

print("epoch=" + str(epoch))
print("model.state_dict()=" + str(model.state_dict()))

# 测试模型
model.eval()

print(model(torch.tensor([1.0])))