import torch

def train(args, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(args.device), y.to(args.device)

        # Compute prediction error
        pred = model(X)
        # print(pred[0])
        # print(y.squeeze())
        loss = loss_fn(pred, y.squeeze())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate loss
        running_loss += loss.item()*len(X)
    
    return running_loss / size

def test(args, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(args.device), y.to(args.device)

            pred = model(X)

            # calculate loss
            test_loss += loss_fn(pred, y.squeeze()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss, correct