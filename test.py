import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# --------------------------
# 配置
# --------------------------
MODEL_PATH = "model-state.bin"  # 替换成你的 model-state.bin 路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tags_vals = ["UNKNOWN", "O", "Name", "Degree", "Skills", "College Name", "Email Address",
             "Designation", "Companies worked at", "Graduation Year", "Years of Experience", "Location"]

tag2idx = {tag: i for i, tag in enumerate(tags_vals)}
idx2tag = {i: tag for i, tag in enumerate(tags_vals)}

MAX_LEN = 128  # 可以根据训练时调整

# --------------------------
# 初始化 tokenizer 和模型
# --------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict']
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


# --------------------------
# 测试函数
# --------------------------
def ner_predict(text):
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        return_tensors="pt",
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    offsets = encoding["offset_mapping"][0]

    # 模型预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs[0]  # shape: (1, seq_len, num_labels)

    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    # 还原 tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # 输出结果
    result = []
    for idx, pred_id in enumerate(predictions):
        if tokens[idx] in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        tag = idx2tag[pred_id]
        token = tokens[idx]
        # 合并 WordPiece 形式
        if token.startswith("##") and result:
            result[-1][0] += token[2:]
        else:
            result.append([token, tag])
    return result


# --------------------------
# 测试例子
# --------------------------
text = "John Doe graduated from MIT in 2020 and works at Google as a Software Engineer."
prediction = ner_predict(text)

for token, tag in prediction:
    print(f"{token}\t{tag}")
