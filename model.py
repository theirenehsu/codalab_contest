import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AdamW
from data_set import GPTDataset
import data_preprocess
from tqdm import trange
import os

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

            if tokenizer.decode(inputs.item()) == "<|END|>":  # 定义 eos 作为终止标记
                break

    return tokenizer.decode(generated_text)



# 定義超參數
BATCH_SIZE = 16
plm = "EleutherAI/pythia-70m"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(plm)
special_tokens_dict = {"bos_token": "<|endoftext|>", "sep_token": "####", "eos_token": "<|END|>"}  
tokenizer.add_special_tokens(special_tokens_dict)
annotation_data_path = "sample_data/answer.txt"
annos_dict = data_preprocess.generate_annotated_medical_report(annotation_data_path)

# 目前還等待理解這一段
PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

seq_pairs = []
train_data_path = "sample_data/First_Phase_Text_Dataset"

# 讀取該資料夾下的所有資料，往前迭代並且傳至 data_preprocess 去生成訓練資料
file_names = os.listdir(train_data_path)
for file_name in file_names:
    file_name = file_name.replace(".txt", "")
    seq_pairs.extend(data_preprocess.process_medical_report(file_name, train_data_path, annos_dict, special_tokens_dict))


tr_dataset = GPTDataset(seq_pairs, tokenizer, special_tokens_dict, 0)

# 創建 DataLoader
bucket_train_dataloader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, collate_fn=tr_dataset.collate_batch)

# 創建模型
model = AutoModelForCausalLM.from_pretrained(plm)
model.resize_token_embeddings(len(tokenizer))


# 目前還等待理解這一段
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

# 創建 AdamW 優化器
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

epochs = 10

# 使用哪個資源做訓練（GPU or CPU or MPS）
device = torch.device('mps') # 如果你是 macbook m1 以上，可以嘗試使用這個
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



for _ in trange(epochs, desc="Epoch"):
    model.train()  # 设置模型为训练模式

    total_loss = 0.0
    # 看起來是迭代 DataLoader 的資料
    for step, (seqs, labels, masks) in enumerate(bucket_train_dataloader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        model.to(device) # 如果是 M1 並且有開 mps 這一行在加不然可以助解掉
        
        # 目前梯度歸零要再思考一下
        # 梯度清零
        model.zero_grad()

        # 模型向前傳遞
        outputs = model(seqs, labels=labels, attention_mask=masks)
        logits = outputs.logits
        loss = outputs.loss
        loss = loss.mean()

        total_loss += loss.item()

        loss.backward()  # 向後傳遞
        optimizer.step()

    # 計算每個 epoch 平均loss
    avg_loss = total_loss / len(bucket_train_dataloader)

    print(f"Epoch {_ + 1}, Average Loss: {avg_loss}")

# 生成資料
generated_text = sample_text(model, tokenizer, "<|endoftext|>PERROTT, GILBERTE LOWELL\n####", n_words=100)
print(generated_text)
