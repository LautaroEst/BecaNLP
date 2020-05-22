import torch

import time

def test_speed(device_name,rep=1):
    device = torch.device(device_name)
    tic1 = time.time()
    x = torch.randn(1024,10000,device=device)
    w = torch.randn(10000,30000,device=device)
    tic2 = time.time()
    for i in range(rep):
        y = torch.matmul(x,w)
    tic3 = time.time()
    print('Dispositivo: {}. Cantidad de repeticiones: {}. Duración: {:.2f}ms + {:.2f}ms = {:.2f}ms'\
          .format(device_name,rep,(tic2-tic1)*1e3,(tic3-tic2)*1e3,(tic3-tic1)*1e3))

if __name__ == '__main__':
    test_speed('cpu',1)
    test_speed('cpu',100)
    #test_speed('cuda',1)
    #test_speed('cuda',100)
