import torch
import torch.optim as optim

def CheckAccuracy(loader, model, device):  
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        
        return num_correct, num_samples
        

def train(model, data, epochs=1, learning_rate=1e-2, sample_loss_every=100):
    
    input_dtype = data['input_dtype'] 
    target_dtype = data['target_dtype']
    device = data['device']
    train_dataloader = data['train_dataloader']
    val_dataloader = data['val_dataloader']
    
    performance_history = {'iter': [], 'loss': [], 'accuracy': []}
    
    model = model.to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for e in range(epochs):
        for t, (x,y) in enumerate(train_dataloader):
            x = x.to(device=device, dtype=input_dtype)
            y = y.to(device=device, dtype=target_dtype)

            # Forward pass
            scores = model(x)

            # Backward pass
            loss = model.loss(scores,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % sample_loss_every == 0:
                num_correct, num_samples = CheckAccuracy(val_dataloader, model, device)
                performance_history['iter'].append(t)
                performance_history['loss'].append(loss.item())
                performance_history['accuracy'].append(float(num_correct) / num_samples)
                print('Epoch: %d, Iteration: %d, Accuracy: %d/%d ' % (e, t, num_correct, num_samples))
                
    num_correct, num_samples = CheckAccuracy(val_dataloader, model, device)
    print('Final accuracy: %.2f%%' % (100 * float(num_correct) / num_samples) )
    
    return performance_history
    
    
    