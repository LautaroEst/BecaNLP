import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm, trange




transformers_dict = {'beto-cased': ('dccuchile/bert-base-spanish-wwm-cased', 
                      BertConfig, BertTokenizer, BertForSequenceClassification),
                     'beto-uncased': ('dccuchile/bert-base-spanish-wwm-uncased',
                      BertConfig, BertTokenizer, BertForSequenceClassification),
                     'xlm-roberta-spanish': ('xlm-roberta-large-finetuned-conll02-spanish',
                      XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification)}


def tokenize_and_split(df, max_len=128, random_state=None, batch_size=32, 
       test_size=0.1, transformer='beto-uncased', pad_token_id=0, **kwargs):
    
    source, _, tokenizer, _ = transformers_dict[transformer]
    
    # Tokenización y padding de los inputs:
    cls_token = kwargs.pop('cls_token','[CLS]')
    sep_token = kwargs.pop('sep_token','[SEP]')
    sentences = ["{} {} {}".format(cls_token,query,sep_token) for query in df['Pregunta']]
    tokenizer = tokenizer.from_pretrained(source, **kwargs)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype="long", truncating="post", padding="post",value=pad_token_id)
    
    # Máscaras de atención:
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
        
    # Etiquetas:
    labels = df['Intencion'].values
    
    # Split en train y validation:
    if random_state is None:
        random_state = np.random.randint(1000000)
        
    if test_size == 0.:
        train_inputs, validation_inputs, train_labels, validation_labels = input_ids, input_ids, labels, labels
        train_masks, validation_masks = attention_masks, attention_masks
    else:        
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                    random_state=random_state, test_size=test_size)
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                     random_state=random_state, test_size=test_size)

    # Datos de train:
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    # Datos de validation:
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    return train_dataloader, validation_dataloader
    

def _select_device(device, model):
    if device is None:
        device = torch.device('cpu')
        print('Warning: Dispositivo no seleccionado. Se utilizará la cpu.')
    elif device == 'parallelize':
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:0')
            model = nn.DataParallel(model)
        else:
            device = torch.device('cpu')
            print('Warning: No es posible paralelizar. Se utilizará la cpu.')
    elif device == 'cuda:0' or device == 'cuda:1':
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            device = torch.device('cpu')
            print('Warning: No se dispone de dispositivos tipo cuda. Se utilizará la cpu.')
    elif device == 'cpu':
        device = torch.device(device)
    else:
        raise RuntimeError('No se seleccionó un dispositivo válido')

    return device, model
    
    
def load_transformer(transformer='beto-cased',device='cpu',**kwargs):
    
    source, config_cls, _, model_cls = transformers_dict[transformer]
    model = model_cls.from_pretrained(source, config=config_cls(**kwargs))
    device, model = _select_device(device, model)
    model = model.to(device)
    return model, device


def validate_model(model,dataloader,device,metrics='accuracy'):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_pred = []
    y_test = []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    
        label_pred = np.argmax(logits[0].detach().cpu().numpy(), axis=1).flatten()
        label_ids = b_labels.to('cpu').numpy().flatten()
        
        y_pred.append(label_pred)
        y_test.append(label_ids)
    
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)
    
    return get_score(y_test,y_pred,metrics)
    
    
def train(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs, device):

    metrics = ['accuracy', 'balanced_accuracy']

    max_grad_norm = 1.0
    train_loss_set = [] # Store our loss and accuracy for plotting

    for _ in trange(epochs, desc="Epoch"):  

        model.train()  

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            loss, _ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            
            if step % 2 == 0:
                print(step)

        score = validate_model(model,validation_dataloader,device,metrics)
        print(score)
    
    return train_loss_set   
    
    
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


def get_test_results(input_filename,
                     output_filename,
                     model,
                     device,
                     transformer='beto-cased',
                     max_len=128,
                     batch_size=32):
    
    # Preprocesamos las muestras
    df_test = pd.read_csv(input_filename)
    sentences = ["[CLS] {} [SEP]".format(query) for query in df_test['Pregunta']]

    # Tokenizamos las oraciones
    source, _, tokenizer_cls, _ = transformers_dict[transformer]
    tokenizer = tokenizer_cls.from_pretrained(source, do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    
    MAX_LEN = max_len # Máxima longitud de las secuencias

    # Paddeamos y convertimos en idx:
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype="long", truncating="post", padding="post")
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
    
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
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    
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




                     