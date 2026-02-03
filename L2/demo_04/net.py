import torch
from transformers import BertModel

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

#加载预训练模型
pretrained = BertModel.from_pretrained(r"D:\transforms\models\model\bert-base-chinese\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea").to(DEVICE)
print(pretrained)

#定义下游任务（增量模型）
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #设计全连接网络，实现二分类任务
        self.fc = torch.nn.Linear(768,2)
    #使用模型处理数据（执行前向计算）
    def forward(self,input_ids,attention_mask,token_type_ids):
        #冻结Bert模型的参数，让其不参与训练
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        #增量模型参与训练
        out = self.fc(out.last_hidden_state[:,0])
        return out