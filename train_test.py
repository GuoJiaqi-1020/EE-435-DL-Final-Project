import torch

def rnn_crossentroy(args, prob, y, loss_fn):
    loss = 0
    for b in range(args.batch_size):
        loss += loss_fn(prob[b], y.squeeze()[b])
    loss = loss / args.batch_size
    return loss

def train(args, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(args.device), y.to(args.device)

        # Compute prediction error
        pred = model(X)
        if args.model == 'rnn':    # crossentropy loss for rnn
            loss = rnn_crossentroy(args, pred, y, loss_fn)
        else:
            loss = loss_fn(pred, y)

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

            # test ffnn model
            if args.model == 'ffnn':
                pred = model(X)
            
            # test rnn model. The prediction should be ouput one by one.
            elif args.model == 'rnn':
                pred_, h = model(X[:,0,:], h)
                pred = pred_
                for i in range(len(y)-1):
                    pred_, h = model(pred_, h)
                    pred = torch.cat((pred, pred_), 1)

            # calculate loss
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss, correct