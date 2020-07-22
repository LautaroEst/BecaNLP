import numpy as np
import pandas as pd


def main():
    df = pd.read_csv('../train.csv',sep = '|')
    df['Intencion'] = df.Intencion.str.findall(r'\d+').apply(lambda x: int(x[0]))
    
    i = 0
    lendf = len(df)
    with open('./my_dataset.csv','w') as f:
        for idx1, (preg1, int1) in df.iterrows():
            for idx2, (preg2, int2) in df.iterrows():
                f.write('[CLS] {} [SEP] {} [SEP],{}\n'.format(preg1,preg2,int(int1 == int2)))
            i += 1
                
            if i % 10 == 0:
                print('{}/{}'.format(idx1,lendf))

if __name__ == '__main__':
    main()