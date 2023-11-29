import numpy as np
import torch

x_len, y_len = 300, 300
seq_len = x_len * y_len

alibi_biases = - torch.ones(seq_len, seq_len,dtype=torch.int8)*100 # q,k

for j in range(seq_len):
    for i in range(j, seq_len):
        pos_x, pos_y = i, i-j
        qx, qy = pos_x // x_len, pos_x % x_len
        kx, ky = pos_y // y_len, pos_y % y_len
        val = np.sqrt(np.power(qx - kx, 2) + np.power(qy - ky, 2))
        if val > 10:
            continue
        alibi_biases[pos_x, pos_y] = -val
alibi_biases = torch.tril(alibi_biases)
x2 = alibi_biases + alibi_biases.T
torch.save(x2,'alibi_tensor_core.pt')

