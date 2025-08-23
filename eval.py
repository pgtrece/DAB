import json

ans_file = "Please write the address of the answer file, and add the 'label' key to the file."
label_file = ans_file


answers = [json.loads(q) for q in open(ans_file, 'r')]
label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

for answer in answers:
    text = answer['text']

    # Only keep the first sentence
    if text.find('.') != -1:
        text = text.split('.')[0]

    text = text.replace(',', '')
    words = text.split(' ')
    if 'No' in words or 'not' in words or 'no' in words:
        answer['text'] = 'no'
    else:
        answer['text'] = 'yes'

for i in range(len(label_list)):
    if label_list[i] == 'no' or label_list[i] == 'No':
        label_list[i] = 0
    else:
        label_list[i] = 1

pred_list = []
for answer in answers:
    if answer['text'] == 'no':
        pred_list.append(0)
    else:
        pred_list.append(1)

pos = 1
neg = 0
yes_ratio = pred_list.count(1) / len(pred_list)

TP, TN, FP, FN = 0, 0, 0, 0
for pred, label in zip(pred_list, label_list):
    if pred == pos and label == pos:
        TP += 1
    elif pred == pos and label == neg:
        FP += 1
    elif pred == neg and label == neg:
        TN += 1
    elif pred == neg and label == pos:
        FN += 1

print('TP\tFP\tTN\tFN\t')
print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

# precision = float(TP) / float(TP + FP)
# recall = float(TP) / float(TP + FN)
# f1 = 2*precision*recall / (precision + recall)
# acc = (TP + TN) / (TP + TN + FP + FN)
precision = float(TP) / float(TP + FP) if (TP + FP) != 0 else 0.0
recall = float(TP) / float(TP + FN) if (TP + FN) != 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0
print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1 score: {}'.format(f1))
print('Yes ratio: {}'.format(yes_ratio))
print(f"预测为 yes 的数量: {pred_list.count(1)}")
print(f"预测为 no 的数量: {pred_list.count(0)}")
print(f"标签中 yes 的数量: {label_list.count(1)}")
print(f"标签中 no 的数量: {label_list.count(0)}")
