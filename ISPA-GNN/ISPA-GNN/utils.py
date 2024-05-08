# 损失函数，打印精度

import torch as t
import numpy as np

def Metrics(realVec, estiVec):
    absError = t.abs(estiVec - realVec)
    mae = t.mean(absError)
    nmae = mae / (t.sum(realVec) / absError.shape[0])
    rmse = t.norm(absError) / np.sqrt(absError.shape[0])
    relativeError = absError / realVec
    mre = np.percentile(relativeError, 50)
    npre = np.percentile(relativeError, 90)
    return np.array([mae, nmae, rmse, mre, npre])


PRINT_LOSS = 'Epoch %02d : Loss = %.4f'

# mae, nmae, rmse, mre, npre
PRINT_RT_METRICS = 'RT: Epoch %02d : MAE=%.4f, NMAE=%.4f, RMSE=%.4f, MRE=%.4f, NPRE=%.4f'
PRINT_TP_METRICS = 'TP: Epoch %02d : MAE=%.4f, NMAE=%.4f, RMSE=%.4f, MRE=%.4f, NPRE=%.4f'

def ReportMetrics(format, data):
    print(format % data)

def ReportLoss(format, data):
    print(format % data)
