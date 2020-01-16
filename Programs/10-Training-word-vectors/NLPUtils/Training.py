import torch
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
from .WordVectors import *



""" Función para calcular la cantidad de muestras predecidas correctamente:
"""
def CheckAccuracy(loader,         # Dataloader
                  model,          # Intancia del modelo 
                  device,         # Lugar en donde correr el entrenamiento (cpu o gpu)
                  input_dtype,    # Data type de las muestras de entrada
                  target_dtype):  # Data type de las muestras de salida 
    
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

    
""" Función para entrenar el modelo mediante el método del SGD:
"""
def SGDTrainModel(model,                 # Intancia del modelo
                  train_data,            # Dataloader del conjunto de entrenamiento
                  validation_data,       # Dataloader del conjunto de validación
                  epochs=1,              # Cantidad de epochs
                  learning_rate=1e-2,    # Tasa de aprendizaje
                  sample_loss_every=100, # Cada cuántas iteraciones calcular la loss
                  check_on_train=False,  # Calcular el accuracy en el conjunto de entrenamiento también
                  use_gpu=True):         # Usar GPUs 
    
    # Data:
    train_dataloader = train_data
    val_dataloader = validation_data
    
    # Data-types:
    input_dtype = next(iter(train_dataloader))[0].dtype
    target_dtype = next(iter(train_dataloader))[1].dtype
    
    # Device:
    device = torch.device('cuda:0') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
    
    
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
                    num_correct_val, num_samples_val = CheckAccuracy(val_dataloader, model, device, input_dtype, target_dtype)
                    performance_history['iter'].append(e * batch_size + t)
                    performance_history['loss'].append(loss.item())
                    performance_history['accuracy'].append(float(num_correct_val) / num_samples_val)
                    if verbose:
                        print('Epoch: {}, Batch number: {}'.format(e, t))
                        print('Accuracy on validation dataset: {}/{} ({:.2f}%)'.format(num_correct_val, num_samples_val, 100 * float(num_correct_val) / num_samples_val))
                    
                    if check_on_train and verbose:
                        num_correct_train, num_samples_train = CheckAccuracy(train_dataloader, model, device, input_dtype, target_dtype)
                        print('Accuracy on train dataset: {}/{} ({:.2f}%)'.format(num_correct_train, num_samples_train, 100 * float(num_correct_train) / num_samples_train))
                        print()
                    elif verbose:
                        print()               
        
        return performance_history
                    
    except KeyboardInterrupt:
        
        print('Exiting training...')
        print('Final accuracy registered on validation dataset: {}/{} ({:.2f}%)'.format(num_correct_val, num_samples_val, 100 * float(num_correct_val) / num_samples_val) )
        if check_on_train:
            num_correct_train, num_samples_train = CheckAccuracy(train_dataloader, model, device, input_dtype, target_dtype)
            print('Final accuracy registered on train dataset: {}/{} ({:.2f}%)'.format(num_correct_train, num_samples_train, 100 * float(num_correct_train) / num_samples_train))
            
        return performance_history

    


""" Función para entrenar Word Embeddings
"""
def SGDTrainWordVectors(data,                   # Corpus de entrenamiento (debe ser una lista de listas de strings).
                        lm='CBOW',              # Modelo de lenguaje a utilizar.
                        window_size=2,          # Tamaño de la ventana.
                        batch_size=64,          # Tamaño del batch.
                        embedding_dim=100,      # Dimensión de los word embeddings.
                        use_gpu=True,           # Flags para usar las GPUs.
                        epochs=1,               # Cantidad de epochs.
                        learning_rate=1e-2,     # Tasa de aprendizaje.
                        sample_loss_every=100): # Cada cuántas iteraciones calcular la loss.
    
    # Chequeo que se haya pasado bien el corpus:
    data_is_ok = True
    if isinstance(data,list):
        for doc in data:
            if isinstance(doc,list) or not data_is_ok:
                for token in doc:
                    if not isinstance(token,str):
                        data_is_ok = False
                        break
            else:
                data_is_ok = False
                break
    else:
        data_is_ok = False
                        
    if data_is_ok:
        corpus = data
    else:
        raise TypeError('data debe ser una lista de listas de tokens o un texto plano (string)')
        return
    
    # Obtengo los batches de muestras:
    dataset = Word2VecSamples(corpus, window_size=window_size)
    samples_idx = torch.randperm(len(dataset))
    my_sampler = lambda indices: sampler.SubsetRandomSampler(indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=my_sampler(samples_idx))
    
    vocab_size = len(dataset.vocabulary)    
    
    # Defino el modelo:
    if lm == 'CBOW':
        model = CBOWModel(vocab_size, embedding_dim)
    elif lm == 'SkipGram':
        model = SkipGramModel(vocab_size, embedding_dim)
    else:
        raise TypeError('El modelo de entrenamiento no es válido.')
    
    
    print('Starting training...')
    loss_history = {'iter': [], 'loss': []}
    device = torch.device('cuda:0') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
    model = model.to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    batch_len = len(dataloader)
    
    try:
        for e in range(epochs):
            for t, (x,y) in enumerate(dataloader):
                model.train()
                x = x.to(device=device, dtype=torch.long)
                y = y.to(device=device, dtype=torch.long)

                if lm == 'CBOW':
                    scores = model(y)
                    loss = model.loss(scores,x)
                elif lm == 'SkipGram':
                    scores = model(x)
                    loss = model.loss(scores,y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (e * batch_len + t) % sample_loss_every == 0:
                    print('Epoch: {}, Batch number: {}, Loss: {}'.format(e+1, t,loss.item()))
                    loss_history['iter'].append(e * batch_len + t)
                    loss_history['loss'].append(loss.item())
                    
                    
        return model.emb, dataset.vocabulary, loss_history
                    
    except KeyboardInterrupt:
        
        print('Exiting training...')
        return model.emb, dataset.vocabulary, loss_history

