# 各 module 功用
## data_preprocess
- 資料的前處理，包括標記資料的前處理，還有我們的訓練資料
- 處理過後將資料return，其中 return 的資料應該是要可以直接放進去下面的 data_set 的物件之中的(不確定有沒有需要 train_test_split)
## data_set
- 建立 GPT_dataset 這個類別，繼承 pytorch 的 Dataset 這個物件
## model
- 關於模型的各種東西
- 包括超參數的設定，Optimizer，使用的 model，訓練與生成等等