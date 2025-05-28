import torch
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
from augmentations import embed_data_mask
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def get_scheduler(args, optimizer):
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142], gamma=0.1)
    return scheduler

def imputations_acc_justy(model,dloader,device):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc, auc


def multiclass_acc_justy(model,dloader,device):
    model.eval()
    vision_dset = True
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    return acc, 0


def classification_scores(model, dloader, device, task,vision_dset):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,0,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()
            y_test = torch.cat([y_test,y_gts.squeeze().float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(y_outs, dim=1).float()],dim=0)
            if task == 'binary':
                prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)

    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    auc = 0
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc.cpu().numpy(), auc


def classification_metrics(model, dloader, device, task, vision_dset):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)

    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), \
            data[3].to(device), data[4].to(device)
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)

            y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
            y_pred = torch.cat([y_pred, torch.argmax(y_outs, dim=1).float()], dim=0)

            if task == 'binary':
                prob = torch.cat([prob, m(y_outs)[:, -1].float()], dim=0)

    # 计算指标
    metrics = {}

    # 准确率
    correct_results_sum = (y_pred == y_test).sum().float()
    metrics['accuracy'] = (correct_results_sum / y_test.shape[0] * 100).cpu().numpy()

    if task == 'binary':
        # 转换为numpy格式以便使用sklearn
        y_test_np = y_test.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        prob_np = prob.cpu().numpy()

        # AUROC
        metrics['auroc'] = roc_auc_score(y_true=y_test_np, y_score=prob_np)

        # AUPRC
        metrics['auprc'] = average_precision_score(y_true=y_test_np, y_score=prob_np)

        # F1分数
        metrics['f1_score'] = f1_score(y_true=y_test_np, y_pred=y_pred_np)

        # 召回率（Sensitivity）
        metrics['recall'] = recall_score(y_true=y_test_np, y_pred=y_pred_np)

        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true=y_test_np, y_pred=y_pred_np)

        # 敏感度（与召回率相同）
        metrics['sensitivity'] = metrics['recall']

        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        metrics['specificity'] = tn / (tn + fp)
    return metrics

from sklearn.metrics import mean_squared_error
import torch


def regression_metrics(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)

    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            # 确保数据正确加载
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), \
                data[3].to(device), data[4].to(device)

            # 确保 y_gts 是一维张量
            if len(y_gts.shape) > 1:
                y_gts = y_gts.squeeze()

            # 嵌入数据
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)

            # 前向传播
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)

            # 确保 y_outs 是一维张量
            if len(y_outs.shape) > 1:
                y_outs = y_outs.squeeze()

            # 收集结果
            y_test = torch.cat([y_test, y_gts.float()], dim=0)
            y_pred = torch.cat([y_pred, y_outs.float()], dim=0)

    # 将张量转换为 NumPy 数组
    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # 计算回归指标
    metrics = {
        "MSE": mean_squared_error(y_test_np, y_pred_np),
        "MAE": mean_absolute_error(y_test_np, y_pred_np),
        "R2": r2_score(y_test_np, y_pred_np),
        "MAPE": mean_absolute_percentage_error(y_test_np, y_pred_np)
    }

    return metrics

def mean_sq_error(model, dloader, device, vision_dset):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            # 确保数据正确加载
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), \
            data[3].to(device), data[4].to(device)

            # 确保 y_gts 是一维张量
            if len(y_gts.shape) > 1:
                y_gts = y_gts.squeeze()

            # 嵌入数据
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)

            # 前向传播
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)

            # 确保 y_outs 是一维张量
            if len(y_outs.shape) > 1:
                y_outs = y_outs.squeeze()

            # 收集结果
            y_test = torch.cat([y_test, y_gts.float()], dim=0)
            y_pred = torch.cat([y_pred, y_outs.float()], dim=0)

    # 计算均方根误差
    y_test_np = y_test.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    mse = mean_squared_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mse)

    return rmse

