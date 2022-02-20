import torch
from torch import nn
from dataset import get_train_val_test_dataloader
from models import CNNModel
from train_test import train, test

class arguments:
    # data
    data_dir = 'Data/Pixel50/'
    # train : val : test = 0.7 : 0.2 : 0.1
    train_val_proportion = 0.9
    train_proportion = 0.7
    seed = 42

    # training setting
    epochs = 10
    batch_size = 32
    lr = 0.0001
    device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    args = arguments
    train_dataloader, val_dataloader, test_dataloader = get_train_val_test_dataloader(args)

    # model
    model = CNNModel().to(args.device)
    print(f"Using {args.device} device")

    # training and validation
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(args, train_dataloader, model, loss_fn, optimizer)
        val_loss, correct = test(args, val_dataloader, model, loss_fn)
        print('training loss: {}, val loss: {}, correct: {}'.format(train_loss, val_loss, correct))
    print("Done!")

    # test
    test_loss, correct = test(args, test_dataloader, model, loss_fn)
    print('\n\n ############## \n test loss: {} correct: {}'.format(test_loss, correct))


if __name__ == '__main__':
    main()
    










