import numpy as np
from sentence_transformers import SentenceTransformer,models

model_path = r"/root/embedding_model/sungw111/text2vec-base-chinese-sentence"
bert = models.Transformer(model_path)
pooling = models.Pooling(bert.get_word_embedding_dimension(),
                        pooling_mode='mean')

# 添加缺失的归一化层
normalize = models.Normalize()

# 组合完整模型
full_model = SentenceTransformer(modules=[bert, pooling, normalize])
print(full_model)

save_path=r"/root/embedding_model/dyb/text2vec-base-chinese-sentence"
full_model.save(save_path)

# 加载修复后的模型
model = SentenceTransformer(r"/root/embedding_model/dyb/text2vec-base-chinese-sentence")

# 验证向量归一化
text = "测试文本"
vec = model.encode(text)
print("修正后模长:", np.linalg.norm(vec))  # 应输出≈1.0