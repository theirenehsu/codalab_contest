from torch.utils.data import Dataset, DataLoader
import torch

class GPTDataset(Dataset):
    def __init__(self, seq_pairs, tokenizer, special_tokens_dict, pad_idx):
        self.seq_pairs = seq_pairs
        self.tokenizer = tokenizer
        self.special_tokens_dict = special_tokens_dict
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.seq_pairs)

    def __getitem__(self, index):
        return self.seq_pairs[index]

    def collate_batch(self, datasets):
        tokens_list, labels_list, attention_mask_list = [], [], []

        for dataset in datasets:
            encoded_seq = self.tokenizer(dataset)  # 假设 tokenizer 是一个合法的方法
            indexed_tks = encoded_seq["input_ids"]
            attention_mask = encoded_seq["attention_mask"]

            tokens_list.append(torch.tensor(indexed_tks))
            labels_list.append(torch.tensor(indexed_tks))
            attention_mask_list.append(torch.tensor(attention_mask))

        return self.pad_sequence(tokens_list, labels_list, attention_mask_list)  # 请注意，这里需要补充 pad_sequence 方法的定义
    
    def pad_sequence(self, non_pad_token, non_pad_label, non_pad_attn):
        max_size = max([len(ele) for ele in non_pad_token])  # 找出该批次数据中的最长序列的长度
        pad_batch1 = torch.stack([torch.cat([t, torch.LongTensor([self.pad_idx] * (max_size - len(t)))]) for t in non_pad_token])
        pad_batch2 = torch.stack([torch.cat([t, torch.LongTensor([self.pad_idx] * (max_size - len(t)))]) for t in non_pad_label])
        pad_batch3 = torch.stack([torch.cat([t, torch.LongTensor([self.pad_idx] * (max_size - len(t)))]) for t in non_pad_attn])
        return pad_batch1, pad_batch2, pad_batch3