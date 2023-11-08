import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AdamW
from data_set import GPTDataset
import data_preprocess
from tqdm import trange


def sample_text(model, tokenizer, text, n_words=100):
    model.eval()
    text = tokenizer.encode(text)
    inputs, past_key_values = torch.tensor([text]).to(device), None
    generated_text = []

    with torch.no_grad():
        for _ in range(n_words):
            output = model(inputs, past_key_values=past_key_values)
            logits = output.logits
            past_key_values = output.past_key_values
            log_probs = F.softmax(logits[:, -1], dim=-1)
            inputs = torch.multinomial(log_probs, 1)
            generated_text.append(inputs.item())

            if tokenizer.decode(inputs.item()) == eos:  # 定义 eos 作为终止标记
                break

    return tokenizer.decode(generated_text)



# 定義超參數
BATCH_SIZE = 16
plm = "EleutherAI/pythia-70m"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(plm)
special_tokens_dict = {"bos_token": "<|endoftext|>", "sep_token": "####", "eos_token": "<|END|>"}  
tokenizer.add_special_tokens(special_tokens_dict)


PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

seq_pairs = data_preprocess.process_medical_report(...)

tr_dataset = GPTDataset(seq_pairs, tokenizer, special_tokens_dict, PAD_IDX)

# 创建数据加载器
bucket_train_dataloader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, collate_fn=tr_dataset.collate_batch)

# 创建模型
model = AutoModelForCausalLM.from_pretrained(plm)
model.resize_token_embeddings(len(tokenizer))


param_optimizer = list(model.named_parameters())

# 不需要权重衰减（weight decay）的参数
no_decay = ['bias', 'LayerNorm.weight']

# 分组参数，区分需要和不需要权重衰减的参数
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01}
]

# 创建 AdamW 优化器
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

epochs = 20

# 设定使用的设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



for _ in trange(epochs, desc="Epoch"):
    model.train()  # 设置模型为训练模式

    total_loss = 0.0

    for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
        segs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # 梯度清零
        model.zero_grad()

        # 模型向前传播
        outputs = model(seqs, labels=labels, attention_mask=masks)
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()

        total_loss += loss.item()

        loss.backward()  # 向后传播计算损失梯度
        optimizer.step()

    # 计算每个 epoch 的平均损失
    avg_loss = total_loss / len(bucket_train_dataloader)

    print(f"Epoch {_ + 1}, Average Loss: {avg_loss}")


generated_text = sample_text(model, tokenizer, "Starting text for generation", n_words=100)
print(generated_text)
