import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import AdamW, get_linear_schedule_with_warmup, AdamWeightDecay
from tqdm import tqdm, trange

def process_dataset(df, max_len=128, random_state=None, batch_size=32, test_size=0.1, **kwargs):
    
    # Preprocesamos las muestras
    sentences = ["[CLS] {} [SEP]".format(query) for query in df['Pregunta']]

    # Tokenizamos las oraciones
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', **kwargs)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    
    MAX_LEN = max_len # Máxima longitud de las secuencias

    # Paddeamos y convertimos en idx:
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    
    # Creamos las máscaras que usa BERT para identificar a los paddings (0 para pad 1 para no-pad)
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
        
    labels = df['Intencion'].values
    
    if random_state is None:
        random_state = np.random.randint(1000000)
        
    if test_size == 0.:
        train_inputs, validation_inputs, train_labels, validation_labels = input_ids, input_ids, labels, labels
        train_masks, validation_masks = attention_masks, input_ids
    else:        
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                    random_state=random_state, test_size=test_size)
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                     random_state=random_state, test_size=test_size)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Create an iterator of our data with torch DataLoader 
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    return train_dataloader, validation_dataloader


def load_classification_model(use_gpu, config):
    
    model = BertForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased", config=config)
    device = torch.device('cuda:1') if torch.cuda.is_available() and use_gpu else torch.device('cpu')
    model = model.to(device)
    return model, device


def validate_model(model,dataloader,device,metrics='accuracy'):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_pred = []
    y_test = []
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    
        # Move logits and labels to CPU
        label_pred = np.argmax(logits[0].detach().cpu().numpy(), axis=1).flatten()
        label_ids = b_labels.to('cpu').numpy().flatten()
        
        y_pred.append(label_pred)
        y_test.append(label_ids)
    
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)
    
    return get_score(y_test,y_pred,metrics)
    
    
    
def get_score(y_test,y_pred,metrics):
    
    scores = {}
    
    if isinstance(metrics,str):
        if 'accuracy' == metrics:
            scores['accuracy'] = accuracy_score(y_test,y_pred)
        elif 'balanced_accuracy' == metrics:
            scores['balanced_accuracy'] = balanced_accuracy_score(y_test,y_pred)
        elif 'f1_macro' == metrics:
            scores['f1_macro'] = f1_score(y_test,y_pred,average='macro')
    else:

        if 'accuracy' in metrics:
            scores['accuracy'] = accuracy_score(y_test,y_pred)
        if 'balanced_accuracy' in metrics:
            scores['balanced_accuracy'] = balanced_accuracy_score(y_test,y_pred)
        if 'f1_macro' in metrics:
            scores['f1_macro'] = f1_score(y_test,y_pred,average='macro')
            
    return scores



def train(model,train_dataloader, validation_dataloader,device):
    
    # BERT fine-tuning parameters
    param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'gamma', 'beta']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.0}
#     ]

    lr = 2e-5
    max_grad_norm = 1.0
    num_training_steps = 1000
    num_warmup_steps = 100
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1
    #optimizer = AdamW(optimizer_grouped_parameters,lr=lr,correct_bias=False)
    optimizer = AdamW(model.parameters(),lr=lr,correct_bias=False)
#     scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                                 num_warmup_steps=num_warmup_steps, 
#                                                 num_training_steps=num_training_steps)  # PyTorch scheduler

    epochs = 15 # Number of training epochs 

    metrics = ['accuracy', 'balanced_accuracy']


    train_loss_set = [] # Store our loss and accuracy for plotting

    # BERT training loop
    for _ in trange(epochs, desc="Epoch"):  

        ## TRAINING:
        # Set our model to training mode
        model.train()  
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, _ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())    
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            #scheduler.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        score = validate_model(model,validation_dataloader,device,metrics)
        print(score)
    
    
    return train_loss_set


def train2(model,train_dataloader, validation_dataloader, optimizer, scheduler, epochs, device):

    metrics = ['accuracy', 'balanced_accuracy']

    max_grad_norm = 1.0
    train_loss_set = [] # Store our loss and accuracy for plotting

    # BERT training loop
    for _ in trange(epochs, desc="Epoch"):  

        ## TRAINING:
        # Set our model to training mode
        model.train()  
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, _ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())    
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            #print('lr:',scheduler.get_lr())
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        score = validate_model(model,validation_dataloader,device,metrics)
        print(score)
    
    
    return train_loss_set
    
    
def get_test_results(input_filename,output_filename,model,device):
    
    # Preprocesamos las muestras
    df_test = pd.read_csv(input_filename)
    sentences = ["[CLS] {} [SEP]".format(query) for query in df_test['Pregunta']]

    # Tokenizamos las oraciones
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    
    MAX_LEN = 128 # Máxima longitud de las secuencias

    # Paddeamos y convertimos en idx:
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    
    # Creamos las máscaras que usa BERT para identificar a los paddings (0 para pad 1 para no-pad)
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
        
    # Convertimos a tensores:
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    test_data = TensorDataset(input_ids, attention_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)
    
    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0
    y_pred = []
    y_test = []
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    
        # Move logits and labels to CPU
        label_pred = np.argmax(logits[0].detach().cpu().numpy(), axis=1).flatten()
        y_pred.append(label_pred)
    
    y_pred = np.concatenate(y_pred)
    
    df_test['Pregunta'] = y_pred
    df_test.to_csv(output_filename,index=False,header=False)
        
    
    
    
