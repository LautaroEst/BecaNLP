import torch
from torch.utils.data import DataLoader, sampler
import torch.optim as optim



def generate_data_batches(train_dataset, test_dataset, # Train y test datasets
                          batch_size = 64, # Tamaño del batch
                          val_size = .02): # Proporción de muestras utilizadas para validación 
    
    """
    Función para iterar sobre los batches de muestras. 
    Devuelve los dataloaders de train / validation / test.
    """

    # Separo las muestras aleatoriamente en Train y Validation:
    NUM_TRAIN = int((1 - val_size) * len(train_dataset)) 
    samples_idx = torch.randperm(len(train_dataset))
    train_samples_idx = samples_idx[:NUM_TRAIN]
    val_samples_idx = samples_idx[NUM_TRAIN:]
    my_sampler = lambda indices: sampler.SubsetRandomSampler(indices) # sampler
    
    # Dataloader para las muestras de entrenamiento:
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  sampler=my_sampler(train_samples_idx))

    # Dataloader para las muestras de validación:
    val_dataloader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                sampler=my_sampler(val_samples_idx))

    # Dataloader para las muestras de testeo:
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader



def CheckAccuracy(loader, model, device, input_dtype, target_dtype):  
    num_correct = 0
    num_samples = 0
    model.eval()  
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=input_dtype)  
            y = y.to(device=device, dtype=target_dtype)
            
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        return num_correct, num_samples
        

def SGDTrainModel(model, data, epochs=1, learning_rate=1e-2, sample_loss_every=100, check_on_train=False):
    
    try:
        input_dtype = data['input_dtype'] 
        target_dtype = data['target_dtype']
    except KeyError:
        print('Input or target data type not correctly defined')
        return
    
    try:
        device = torch.device('cuda:0') if torch.cuda.is_available() and data['use_gpu'] else torch.device('cpu')
    except KeyError:
        print('Device not specified')
        return
    
    try:
        train_dataloader = data['train_dataloader']
        val_dataloader = data['val_dataloader']
    except KeyError:
        print('Train or Validation dataloaders not defined')
        return
    
    performance_history = {'iter': [], 'loss': [], 'accuracy': []}
    
    model = model.to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    batch_size = len(train_dataloader)
    
    try:
    
        for e in range(epochs):
            for t, (x,y) in enumerate(train_dataloader):
                model.train()
                x = x.to(device=device, dtype=input_dtype)
                y = y.to(device=device, dtype=target_dtype)

                # Forward pass
                scores = model(x) 

                # Backward pass
                loss = model.loss(scores,y)                 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (e * batch_size + t) % sample_loss_every == 0:
                    print('Epoch: {}, Batch number: {}'.format(e, t))
                    num_correct_val, num_samples_val = CheckAccuracy(val_dataloader, model, device, input_dtype, target_dtype)
                    performance_history['iter'].append(t)
                    performance_history['loss'].append(loss.item())
                    performance_history['accuracy'].append(float(num_correct_val) / num_samples_val)
                    print('Accuracy on validation dataset: {}/{} ({:.2f}%)'.format(num_correct_val, num_samples_val, 100 * float(num_correct_val) / num_samples_val))
                    
                    if check_on_train:
                        num_correct_train, num_samples_train = CheckAccuracy(train_dataloader, model, device, input_dtype, target_dtype)
                        print('Accuracy on train dataset: {}/{} ({:.2f}%)'.format(num_correct_train, num_samples_train, 100 * float(num_correct_train) / num_samples_train))
                        print()
                    else:
                        print()
                        
        return performance_history
                    
    except KeyboardInterrupt:
        
        print('Exiting training...')
        #num_correct_val, num_samples_val = CheckAccuracy(val_dataloader, model, device, input_dtype, target_dtype)
        print('Final accuracy registered on validation dataset: {}/{} ({:.2f}%)'.format(num_correct_val, num_samples_val, 100 * float(num_correct_val) / num_samples_val) )
        if check_on_train:
            #num_correct_train, num_samples_train = CheckAccuracy(train_dataloader, model, device, input_dtype, target_dtype)
            print('Final accuracy registered on train dataset: {}/{} ({:.2f}%)'.format(num_correct_train, num_samples_train, 100 * float(num_correct_train) / num_samples_train))
            
        return performance_history

