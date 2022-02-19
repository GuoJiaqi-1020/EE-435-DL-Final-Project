import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import WikitextDataset, Get_corpus
from models import FFNN, RNN
from train_test import train, test

class arguments:
    # data
    n = 5           # n gram model
    length = 30     # rnn time length
    # model
    model = 'ffnn'  # or 'rnn'
    # training setting
    epoch = 10
    batch_size = 32
    lr = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    args = arguments
    # prepare data
    # corpus = Get_corpus('wiki.train.txt')
    # training_data = WikitextDataset(args, corpus, 'wiki.train.txt')
    # val_data = WikitextDataset(args, corpus, 'wiki.valid.txt')
    # test_data = WikitextDataset(args, corpus, 'wiki.test.txt')
    training_dataset = torch.load(str(args.model)+'_train_dataset.pt')
    val_dataset = torch.load(str(args.model)+'_val_dataset.pt')
    test_dataset = torch.load(str(args.model)+'_test_dataset.pt')

    train_dataloader = DataLoader(training_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset ,batch_size=args.batch_size)

    # model
    if args.model == 'ffnn':
        model = FFNN()
    elif args.model == 'rnn':
        model = RNN()
    model.to(args.device)
    print(f"Using {args.device} device")

    # training and validation
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(args, train_dataloader, model, loss_fn, optimizer)
        val_loss, correct = test(args, val_dataloader, model, loss_fn)
        print('training loss: {}, val loss: {}, correct: {}'.format(train_loss, val_loss, correct))
    print("Done!")

    # test
    test_loss, correct = test(args, test_dataloader, model, loss_fn)
    print('\n\n ############## \n test loss: {} correct: {}'.format(test_loss, correct))










