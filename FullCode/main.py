from preprocess import *
from utils import split_validation, Data
import argparse
from model_mr import *
from tqdm import trange, tqdm
from sklearn import metrics

# Parameters
args = argparse.Namespace()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.node_dim = 300
args.dp = 20
args.lap_node_id_eig_dropout = 0.15
args.batch_size = 8
args.eigen_norm = False
args.type_id = False
args.layernorm = True
args.dropout = 0.15
args.nhead = 10
args.num_layers = 2
args.num_classes = 2
args.SEED = 666
args.epochs = 3
args.lr = 0.01



# Load data
doc_content_list, doc_train_list, doc_test_list, vocab_dic, labels_dic, max_num_sentence, keywords_dic, class_weights \
    = read_file('mr', True)

args.max_words = len(vocab_dic)+2

gloveFile = 'data/glove.6B.300d.txt'
pre_trained_weight = loadGloveModel(gloveFile, vocab_dic, len(vocab_dic)+1)

train_data, valid_data = split_validation(doc_train_list, 0.1, args.SEED)
test_data = split_validation(doc_test_list, 0.0, args.SEED)

num_categories = len(labels_dic)

train_data = Data(train_data, max_num_sentence, keywords_dic, num_categories, True)
valid_data = Data(valid_data, max_num_sentence, keywords_dic, num_categories, True)
test_data  = Data(test_data,  max_num_sentence, keywords_dic, num_categories, True)




# Model
model = tokenHGT(args, pre_trained_weight).to(args.device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
loss_func = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

def train_model(model, train_data, epoch):
    """
    train torch model
    """
    model.train()

    total_loss = 0.0
    slices = train_data.generate_batch(args.batch_size, True)
    for step in tqdm(range(len(slices)), total=len(slices), leave=False):
        i = slices[step]
        if len(i) < args.batch_size:
            i = pad_batch(i, args.batch_size)
        batch_data = train_data.get_slice(i)
        optimizer.zero_grad()
        pred, targets = model(batch_data)
        #         print(torch.argmax(pred, 1), targets)
        loss = loss_func(pred, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss
    scheduler.step()

    print('epoch: ', epoch, '\tLoss:\t%.4f' % (total_loss / step))

def valid_model(model, valid_data):
    model.eval()

    preds = []
    targets = []

    slices = valid_data.generate_batch(args.batch_size, False)
    with torch.no_grad():
        for step in tqdm(range(len(slices)), total=len(slices), leave=False):
            i = slices[step]
            if len(i) < args.batch_size:
                i = pad_batch(i, args.batch_size)
            batch_data = valid_data.get_slice(i)
            pred, target = model(batch_data)
            if len(i) < args.batch_size:
                pred = torch.argmax(pred, 1)[:len(i)]
                target = target[:len(i)]
            else:
                pred = torch.argmax(pred, 1)

            preds.extend(pred.detach().cpu())
            targets.extend(target.detach().cpu())

    print('Accuracy: ', metrics.accuracy_score(targets, preds))




# Training

for epoch in trange(args.epochs):
    train_model(model, train_data, epoch)
    valid_model(model, valid_data)

valid_model(model, test_data)
