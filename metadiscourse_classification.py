import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
)
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, BertForTokenClassification, Trainer, TrainingArguments, EvalPrediction
)

spanbert_version = "SpanBERT/spanbert-base-cased"
magic_number = 362
ugly_count = 0

log_file = open('debug_log.txt', 'w')

reversed_map = {
    0: 'AAR', 1: 'BST', 2: 'CLF', 3: 'EDP', 4: 'ENM', 5: 'EPA', 6: 'ERR',
    7: 'EXP', 8: 'HDG', 9: 'IMS', 10: 'MCD', 11: 'MIS', 12: 'MNC', 13: 'MNM',
    14: 'MNT', 15: 'ORS', 16: 'PED', 17: 'PRV', 18: 'RFL', 19: 'RPR',
    20: 'RVW', 21: 'SAL', 22: 'UCT'
}

strings_list = ["AAR", "BST", "CLF", "EDP", "ENM", "EPA", "ERR", "EXP", "HDG", "IMS", "MCD", "MIS", "MNC", "MNM", "MNT", "ORS", "PED", "PRV", "RFL", "RPR", "RVW", "SAL", "UCT"] #23
strings_list_without = ["AAR", "BST", "CLF", "EDP", "ENM", "EPA", "EXP", "HDG", "IMS", "MCD", "MNC", "MNM", "MNT", "ORS", "PED", "PRV", "RFL", "RPR", "RVW", "SAL"] #20

biggest_labels = ["ORS", "BST", "HDG", "EPA", "ENM", "EXP"] #6
smallest_labels = ['PED', 'MCD', 'UCT', 'MNT', 'SAL', 'MIS', 'CLF', 'RVW', 'PRV', 'MNC', 'MNM', 'IMS', 'RFL', 'AAR', 'EDP', 'RPR'] #16
small_100_labels = ['SAL', 'MIS', 'CLF', 'RVW', 'PRV', 'MNC', 'MNM', 'IMS', 'RFL', 'AAR', 'EDP', 'RPR'] #12


def log(text):
    print(text, file=log_file)


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]

    texts = []
    labels = []

    for entry in data:
        entry_labels = entry['spans']  # Assuming spans are dictionaries with 'start', 'end', and 'label'

        # Filter out spans with the labels 'RPR' and 'EDP'
        filtered_labels = [span for span in entry_labels if span['label'] in strings_list_without]

        if filtered_labels:  # Only keep the entry if there are remaining labels after filtering
            texts.append(entry['text'])
            labels.append(filtered_labels)

    return texts, labels

def align_labels_with_tokens(encodings, text_labels, label_map, texts):
    aligned_labels = []

    # Iterate over each document's encodings and corresponding labels
    for doc_encoding, doc_labels, doc_text in zip(encodings.encodings, text_labels, texts):
        # Initialize an array filled with -100 for each token in the document
        doc_aligned_labels = np.full(magic_number, -100, dtype=int)

        # Process each span in the document's labels
        for span in doc_labels:
            # Convert character-level span indices to token-level indices
            # Here token is char position, not word position, correct?
            start_token_idx = doc_encoding.char_to_token(span['start'])
            end_token_idx = doc_encoding.char_to_token(span['end'] - 1)

            # If valid token indices are found, assign the label to all tokens in the span
            # if start_token_idx is not None and end_token_idx is not None:
            if end_token_idx is not None:
                doc_aligned_labels[start_token_idx:end_token_idx + 1] = label_map[span['label']]

        aligned_labels.append(doc_aligned_labels.tolist())

    return aligned_labels

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def create_dataset(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=magic_number, return_tensors="pt")
    return CustomDataset(encodings, labels)

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=2)

    # Flatten predictions and true labels
    preds_flat = preds.flatten()
    labels_flat = p.label_ids.flatten()

    # Filter out '-100' values
    valid_mask = labels_flat != -100
    preds_flat = preds_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]

    # Calculate metrics
    accuracy = accuracy_score(labels_flat, preds_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='weighted')

    res = np.array(labels_flat)
    unique_res = np.unique(res)
    ur = unique_res.tolist()
    ur = [reversed_map[i] for i in ur]

    global ugly_count
    ugly_count += 1

    labels_flat_np = np.array(labels_flat)
    unique_values = np.unique(labels_flat_np)
    print("UNIQUE", len(unique_values))

    label_names = strings_list_without

    print(classification_report(labels_flat, preds_flat, target_names=label_names))


    conf_mat = confusion_matrix(labels_flat, preds_flat)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='g', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Save the figure
    plt.savefig(f'confusion_matrix_epoch_{ugly_count}.png')
    plt.close()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


file_path = 'data_v1.jsonl'
texts, span_labels = load_data(file_path)

md_dict = {key: {} for key in strings_list}
md_count = 0

for conversation_labels in span_labels:
    for md in conversation_labels:
        span = texts[md_count][md['start']:md['end']]
        # print(span)
        # input("Press Enter to continue...")

        if span in md_dict[md['label']].keys():
            md_dict[md['label']][span] += 1
        else:
            md_dict[md['label']][span] = 1

    md_count += 1

for md in md_dict.keys():
    print(md, ":", len(md_dict[md].keys()))

flat_labels = set([span['label'] for sublist in span_labels for span in sublist])
label_encoder = LabelEncoder()
label_encoder.fit(list(flat_labels))
label_map = {label: index for index, label in enumerate(label_encoder.classes_)}
tokenizer = AutoTokenizer.from_pretrained(spanbert_version)
encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=magic_number, return_tensors="pt")

aligned_labels = align_labels_with_tokens(encodings, span_labels, label_map, texts)


# random state van 24 bij alle classes, 2 bij kleinste 16
train_texts, remaining_texts, train_labels, remaining_labels = train_test_split(
    texts, aligned_labels, test_size=0.3, random_state=24)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    remaining_texts, remaining_labels, test_size=0.5, random_state=42)

# Check if all categories present
uniques = []
for text in val_labels:
    for token in text:
        if token not in uniques and token != -100:
            uniques.append(token)
a = len(uniques)
print("Uniques val", uniques, len(uniques))

uniques = []
for text in test_labels:
    for token in text:
        if token not in uniques and token != -100:
            uniques.append(token)
b = len(uniques)
print("Uniques test", uniques, len(uniques))

# GET random_state where both sets have all labels
x = 1
while a != 20 or b != 20:
    train_texts, remaining_texts, train_labels, remaining_labels = train_test_split(
        texts, aligned_labels, test_size=0.3, random_state=x)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        remaining_texts, remaining_labels, test_size=0.5, random_state=42)

    uniques = []
    for text in val_labels:
        for token in text:
            if token not in uniques and token != -100:
                uniques.append(token)
    a = len(uniques)
    print(a, "a")

    uniques = []
    for text in test_labels:
        for token in text:
            if token not in uniques and token != -100:
                uniques.append(token)
    b = len(uniques)
    print(b, "b")
    print(x)

    x += 1

train_dataset = create_dataset(train_texts, train_labels)
val_dataset = create_dataset(val_texts, val_labels)
test_dataset = create_dataset(test_texts, test_labels)

model = AutoModelForTokenClassification.from_pretrained(spanbert_version, num_labels=len(label_map))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained('./fine_tuned_model_new')
tokenizer.save_pretrained('./fine_tuned_model_new')

print("Evaluation results:", trainer.evaluate(eval_dataset=test_dataset))
