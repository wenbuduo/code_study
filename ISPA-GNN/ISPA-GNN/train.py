from datasets import *
from torch.utils.data import DataLoader
from model import DualGNN, PredictLoss
from tqdm import *
from utils import *

Number_of_workers = 1
Number_of_prefetch_factor = 2

Save_result_filename = 'Result.txt'


# RT TP  subgraph & netsubgraph
def EvaluateV4(args, datatype):
    trainGraph, testGraph = train_test_split(get_graph(), density=args.density)

    trainset = QoSDataSetV3(trainGraph, args.boxcox)
    testset = QoSDataSetV3(testGraph, args.boxcox)

    trainLoader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.gpu,
        drop_last=True,
        collate_fn=collateV3,
        # num_workers=Number_of_workers,
        # prefetch_factor=Number_of_prefetch_factor
    )

    testLoader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.gpu,
        drop_last=True,
        collate_fn=collateV3,
        # num_workers=Number_of_workers,
        # prefetch_factor=Number_of_prefetch_factor
    )

    if datatype == 1:
        model = DualGNN(dim=args.dim, order=args.order, gpu=args.gpu, ctx=True, type='RT').to(args.device)
    else:
        model = DualGNN(dim=args.dim, order=args.order, gpu=args.gpu, ctx=True, type='TP').to(args.device)

    lr = args.lr

    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)

    dataTransform = trainset.transform

    logger.info('Model V3 Training Started!')

    for epoch in range(args.epoch):
        if epoch % 5 == 0:
            optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
            lr /= 2
        trainingLoss = []  # 打印损失值
        trainIdx = 0
        model.train()
        for batch in tqdm(trainLoader):
            batchGraph, neighGraph, label = batch
            if args.gpu:
                batchGraph = batchGraph.to(args.device)
                neighGraph = neighGraph.to(args.device)
                label = label.to(args.device)
            print(batchGraph)
            pred = model.forward(batchGraph, neighGraph)
            pLoss = PredictLoss(pred, label)
            optimizer.zero_grad()
            pLoss.backward()
            optimizer.step()
            trainingLoss += [pLoss.item()]

        # ReportLoss(PRINT_LOSS, (epoch, np.mean(trainingLoss)))

        # Test Period
        accuracyRT = np.zeros((5,))
        model.eval()
        with t.no_grad():
            for batch in tqdm(testLoader):
                batchGraph, neighGraph, RT = batch
                if args.gpu:
                    batchGraph = batchGraph.to(args.device)
                    neighGraph = neighGraph.to(args.device)
                if args.gpu:
                    pred = model.forward(batchGraph, neighGraph).cpu()
                else:
                    pred = model.forward(batchGraph, neighGraph)
                pred = dataTransform.inv(pred)
                RT = dataTransform.inv(RT)
                accuracyRT += Metrics(pred, RT)

        accuracyRT /= len(testLoader)

        if datatype == 1:
            ReportMetrics(PRINT_RT_METRICS, (epoch + 1, *accuracyRT))
        else:
            ReportMetrics(PRINT_TP_METRICS, (epoch + 1, *accuracyRT))

        # Record the result
        if epoch == args.epoch - 1:
            if args.datatype == 1:
                Note = open(Save_result_filename, mode='a')
                Note.write(PRINT_RT_METRICS % (epoch + 1, *accuracyRT) + '\n')
                Note.write('\n')
                Note.close()
            else:
                Note = open(Save_result_filename, mode='a')
                Note.write(PRINT_TP_METRICS % (epoch + 1, *accuracyRT) + '\n')
                Note.close()

#
# # RT TP  subgraph
# def EvaluateV2(args, datatype):
#     trainGraph, testGraph = train_test_split(get_graph(), density=args.density)
#
#     trainset = QoSDataSetV3(trainGraph, args.boxcox)
#     testset = QoSDataSetV3(testGraph, args.boxcox)
#
#     trainLoader = DataLoader(
#         trainset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         pin_memory=args.gpu,
#         collate_fn=collateV3,
#         num_workers=Number_of_workers,
#         prefetch_factor=Number_of_prefetch_factor
#     )
#
#     testLoader = DataLoader(
#         testset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         pin_memory=args.gpu,
#         collate_fn=collateV3,
#         num_workers=Number_of_workers,
#         prefetch_factor=Number_of_prefetch_factor
#     )
#
#     ### 没有激活粗粒度网络图
#     if datatype == 1:
#         modelV2 = DualGNN(dim=args.dim, order=2, gpu=args.gpu, ctx=False, type='RT')
#     else:
#         modelV2 = DualGNN(dim=args.dim, order=2, gpu=args.gpu, ctx=False, type='TP')
#
#     lr = args.lr
#
#     optimizer = t.optim.AdamW(modelV2.parameters(), lr=lr, weight_decay=1e-1)
#
#     dataTransform = trainset.transform
#
#     logger.info('Model V2 Training Started!')
#
#     for epoch in range(args.epoch):
#         if epoch % 5 == 0:
#             optimizer = t.optim.AdamW(modelV2.parameters(), lr=lr, weight_decay=1e-1)
#             lr /= 2
#
#         trainingLoss = []
#
#         modelV2.train()
#
#         for batch in trainLoader:
#             batchGraph, neighGraph, label = batch
#
#             if args.gpu:
#                 batchGraph = batchGraph.to('cuda')
#                 neighGraph = neighGraph.to('cuda')
#                 label = label.cuda()
#
#             rtHat = modelV2.forward(batchGraph)
#             pLoss = PredictLoss(rtHat, label)
#             optimizer.zero_grad()
#             pLoss.backward()
#             optimizer.step()
#             trainingLoss += [pLoss.item()]
#
#         ReportLoss(PRINT_LOSS, (epoch, np.mean(trainingLoss)))
#         # Test Period
#         accuracyRT = np.zeros((5,))
#
#         modelV2.eval()
#         with t.no_grad():
#             for batch in testLoader:
#                 batchGraph, neighGraph, RT = batch
#                 if args.gpu:
#                     batchGraph = batchGraph.to('cuda')
#                 rtHat = modelV2.forward(batchGraph).cpu()
#                 rtHat = dataTransform.inv(rtHat)
#                 RT = dataTransform.inv(RT)
#                 accuracyRT += Metrics(rtHat, RT)
#
#         accuracyRT /= len(testLoader)
#
#         if datatype == 1:
#             ReportMetrics(PRINT_RT_METRICS, (epoch + 1, *accuracyRT))
#         else:
#             ReportMetrics(PRINT_TP_METRICS, (epoch + 1, *accuracyRT))
#
#         if epoch == args.epoch - 1:
#             if args.datatype == 1:
#                 Note = open(Save_result_filename, mode='a')
#                 Note.write(PRINT_RT_METRICS % (epoch + 1, *accuracyRT) + '\n')
#                 Note.write('\n')
#                 Note.close()
#             else:
#                 Note = open(Save_result_filename, mode='a')
#                 Note.write(PRINT_TP_METRICS % (epoch + 1, *accuracyRT) + '\n')
#                 Note.close()
#
#         # Record the result
