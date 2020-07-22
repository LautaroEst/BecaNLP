from utils_v2 import *


def main():
    # Cargamos el dataset:
    df = pd.read_csv('../train.csv',sep = '|')
    df['Intencion'] = df.Intencion.str.findall(r'\d+').apply(lambda x: int(x[0]))

    # Procesamos y dividimos las muestras en train y dev:
    train_dataloader, validation_dataloader = tokenize_and_split(df, 
                                                                 max_len=128, 
                                                                 random_state=None, 
                                                                 batch_size=64, 
                                                                 test_size=0., 
                                                                 transformer='xlm-roberta-spanish',
                                                                 pad_token_id=0,
                                                                 do_lower_case=True)
    
    # Definimos el transformer que vamos a usar:
    model, device = load_transformer(transformer='xlm-roberta-spanish',
                                     device='cpu',
                                     vocab_size=250002, 
                                     hidden_size=1024, 
                                     num_hidden_layers=24, 
                                     num_attention_heads=16, 
                                     intermediate_size=4096, 
                                     hidden_act='gelu', 
                                     hidden_dropout_prob=0.1, 
                                     attention_probs_dropout_prob=0.1, 
                                     max_position_embeddings=514, 
                                     type_vocab_size=1, 
                                     initializer_range=0.02, 
                                     layer_norm_eps=1e-12, 
                                     pad_token_id=1,
                                     bos_token_id=0, 
                                     eos_token_id=2,
                                     gradient_checkpointing=False, 
                                     num_labels=np.max(df['Intencion'].values)+1)
    
    # Entrenamiento
    epochs = 10
    lr = 2e-5
    warmup_proportion = 0.1

    num_training_steps =  len(train_dataloader) * epochs
    num_warmup_steps = int(warmup_proportion * num_training_steps)
    optimizer = AdamW(model.parameters(), weight_decay=0., lr=lr,correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=num_training_steps)
    try:
        
        train_loss_set = train(model,train_dataloader, validation_dataloader, optimizer, scheduler, epochs, device)
        
    except KeyboardInterrupt:
        pass
        

    # plot training performance
    plt.figure(figsize=(15,8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    plt.show()
    
    input_filename = '../test_santander.csv'
    output_filename = './results_xlm_roberta.csv'
    get_test_results(input_filename,
                     output_filename,
                     model,
                     device,
                     transformer='xlm-roberta-spanish',
                     max_len=128,
                     batch_size=32)
        
    
if __name__ == '__main__':
    main()
    
    