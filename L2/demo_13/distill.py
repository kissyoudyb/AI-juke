import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW


# ========== 配置参数 ==========
class Config:
    # 模型设置
    teacher_model_name = "/root/autodl-tmp/Qwen/Qwen1.5-1.8B-Chat"
    student_model_name = "/root/autodl-tmp/Qwen/Qwen1.5-0.5B-Chat"

    # 训练参数
    batch_size = 1
    num_epochs = 30
    learning_rate = 1e-5  # 降低学习率
    max_seq_length = 512
    temperature = 3.0  # 降低温度值
    alpha = 0.7  # 蒸馏损失权重

    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grad_accum_steps = 4  # 梯度累积步数

    # 使用float32避免混合精度问题
    dtype = torch.float32


config = Config()


# ========== 数据加载 ==========
class DistillationDataset(Dataset):
    def __init__(self, tokenizer, sample_texts=None):
        self.tokenizer = tokenizer
        self.examples = []

        # 示例数据（实际需替换为真实数据集）
        sample_texts = [
            "人工智能的核心理念是",
            "大语言模型蒸馏的关键在于",
            "深度学习模型的压缩方法包括",
            "知识蒸馏如何提高小模型性能",
            "Transformer架构的核心组件是",
            "注意力机制的工作原理",
            "模型量化如何减少计算资源",
            "神经网络剪枝的基本方法",
            "模型蒸馏中的温度参数作用",
            "如何评估蒸馏后模型的质量",
            "软标签与硬标签的区别",
            "蒸馏损失函数的设计原则",
            "教师模型与学生模型的选择",
            "蒸馏训练中的学习率调度",
            "如何防止蒸馏过程中的过拟合"
        ]

        for text in sample_texts:
            encoding = tokenizer(
                text,
                max_length=config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            self.examples.append(encoding)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"].squeeze(),
            "attention_mask": self.examples[idx]["attention_mask"].squeeze()
        }


# ========== 模型初始化 ==========
def load_models():
    # 加载教师模型（冻结参数）
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        device_map="auto",
        torch_dtype=config.dtype
    ).eval()

    # 加载学生模型
    student = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        device_map="auto",
        torch_dtype=config.dtype
    ).train()

    return teacher, student


# ========== 蒸馏损失函数 ==========
class DistillationLoss:
    @staticmethod
    def calculate(
            teacher_logits,  # 教师模型logits [batch, seq_len, vocab]
            student_logits,  # 学生模型logits [batch, seq_len, vocab]
            attention_mask,  # 注意力掩码
            temperature=config.temperature,
            alpha=config.alpha
    ):
        # 1. 添加数值稳定性处理
        teacher_logits = torch.clamp(teacher_logits, min=-1e4, max=1e4)
        student_logits = torch.clamp(student_logits, min=-1e4, max=1e4)

        # 2. 软目标蒸馏损失
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)

        # 3. 添加掩码处理，避免填充位置影响损失
        mask = attention_mask.unsqueeze(-1).expand_as(soft_teacher)
        kl_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction="none",
            log_target=False
        )
        kl_loss = (kl_loss * mask).sum() / mask.sum()  # 平均每个token的损失
        kl_loss = kl_loss * (temperature ** 2)

        # 4. 学生自训练损失（交叉熵）
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = teacher_logits.argmax(-1)[..., 1:].contiguous()

        # 5. 使用掩码过滤填充位置
        shift_mask = attention_mask[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none"
        )
        ce_loss = (ce_loss * shift_mask.view(-1)).sum() / shift_mask.sum()

        # 6. 确保损失值有效
        if torch.isnan(kl_loss).any() or torch.isnan(ce_loss).any():
            kl_loss = torch.tensor(0.0, device=kl_loss.device)
            ce_loss = torch.tensor(0.0, device=ce_loss.device)
            print("NaN loss detected, resetting to zero")

        return alpha * kl_loss + (1 - alpha) * ce_loss


# ========== 训练流程 ==========
def train():
    # 初始化组件
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    teacher, student = load_models()

    # 确保学生模型在正确设备上
    student.to(config.device)

    # 数据集示例
    dataset = DistillationDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)

    # 优化器设置
    optimizer = AdamW(student.parameters(), lr=config.learning_rate, weight_decay=0.01)

    step_count = 0
    # 训练循环
    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            inputs = {k: v.to(config.device) for k, v in batch.items()}

            # 教师模型前向（不计算梯度）
            with torch.no_grad():
                teacher_outputs = teacher(**inputs)

            # 学生模型前向
            student_outputs = student(**inputs)

            # 添加注意力掩码到损失计算
            loss = DistillationLoss.calculate(
                teacher_outputs.logits,
                student_outputs.logits,
                inputs["attention_mask"]
            )

            # 检查损失是否为NaN
            if torch.isnan(loss):
                print("NaN loss detected, skipping backward pass")
                optimizer.zero_grad()
                continue

            # 反向传播（带梯度累积）
            (loss / config.grad_accum_steps).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)

                # 参数更新
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

                # 学习率调整（示例）
                warmup_steps = 500
                if step_count < warmup_steps:
                    lr = config.learning_rate * step_count / warmup_steps
                else:
                    lr = config.learning_rate * (warmup_steps ** 0.5) / (step_count ** 0.5)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # 打印训练信息
                if step_count % 10 == 0:
                    print(f"Epoch {epoch + 1} | Step {step_count} | Loss: {loss.item():.4f} | LR: {lr:.2e}")

                    # 添加梯度检查
                    total_grad_norm = 0.0
                    for name, param in student.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            total_grad_norm += grad_norm ** 2
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"NaN or Inf gradient in {name}")
                            if grad_norm > 1e3:  # 梯度值过大
                                print(f"Large gradient in {name}: {grad_norm:.4f}")

                    total_grad_norm = total_grad_norm ** 0.5
                    print(f"Total Gradient Norm: {total_grad_norm:.4f}")

    # 保存蒸馏后的模型
    student.save_pretrained("./distilled_qwen")
    tokenizer.save_pretrained("./distilled_qwen")


if __name__ == "__main__":
    train()