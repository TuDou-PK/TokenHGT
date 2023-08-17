from datasets import list_datasets
from datasets import load_dataset

datasets_list = list_datasets()
dataset = load_dataset('rotten_tomatoes')

# dataset:
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 8530
#     })
#     validation: Dataset({
#         features: ['text', 'label'],
#         num_rows: 1066
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 1066
#     })
# })

# dataset['train'][8529]
# {'text': 'things really get weird , though not particularly scary : the movie is all portent and no content .',
#  'label': 0}

train_data = dataset['train']
validation_data = dataset['validation']
test_data = dataset['test']

with open('data/mr_corpus.txt', 'w', encoding='utf-8') as file:
    for row in train_data:
        file.write(row['text'] + '\n')
    for row in validation_data:
        file.write(row['text'] + '\n')
    for row in test_data:
        file.write(row['text'] + '\n')

len_train = len(train_data)
len_val   = len(validation_data)
with open('data/mr_labels.txt', 'w', encoding='utf-8') as file:
    for i, row in enumerate(train_data):
        file.write(f"{i}\ttrain\t{'pos' if row['label'] == 1 else 'neg'}\n")
    for i, row in enumerate(validation_data):
        file.write(f"{len_train+i}\ttrain\t{'pos' if row['label'] == 1 else 'neg'}\n")
    for i, row in enumerate(test_data):
        file.write(f"{len_train+len_val+i}\ttest\t{'pos' if row['label'] == 1 else 'neg'}\n")