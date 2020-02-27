from Utils import *

def main():

    train_corpus = Corpus.from_text_files(['./wiki-corpus/{}_cleaned.txt'.format(i) for i in range(1,1)] \
                                          + ['./promptsl40_train_cleaned.txt'], r'[ \s]+', 3)

    model = 'SkipGram'
    window_size_list = [2,3,4,5]
    embedding_dim_list = [50,100,200,300]
    batch_size = 512
    device = 'cuda:0'

    trainers3 = []
    for window_size in window_size_list:
        for embedding_dim in embedding_dim_list:
            trainer =  Word2VecTrainer(model, train_corpus, window_size, embedding_dim, batch_size, device)
            trainers3.append(trainer)


    algorithm = 'Adam'
    epochs = 2
    sample_loss_every = 100
    learning_rate = 5e-4

    for trainer in trainers3:
        trainer.Train(algorithm, epochs, sample_loss_every, lr=learning_rate)
        
if "__main__" == __name__:
    main()
