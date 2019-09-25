from encoding.models import get_segmentation_model
from torch.autograd import Variable
import torch.onnx

batch = 3
dummy_input = Variable(torch.randn(batch, 3, 480, 480))

model = get_segmentation_model('encnet', jpu=False, lateral=False)
print(model)

# エラーがおきる。
# 多分、encoding.nn.Encodingが悪さしてるような気がする
torch.onnx.export(
    model, dummy_input, "encnet.onnx",
    verbose = True
    )
