from sklearn.metrics import confusion_matrix#混淆矩阵
import numpy as np
con_mat=[[19,1,0,0],[0,7,12,1],[2,1,17,0],[2,1,0,17]]

def sen_all(pred_y,res,n,se,ppv,):
    F1_score=[]
    # con_mat = confusion_matrix(pred_y, res)
    print(con_mat)
    for i in range(n):
        tp=con_mat[i][i]
        fn=np.sum(con_mat[i,:])-tp
        tn=0
        for j in range(n):
            tn +=con_mat[j][j]
        tn -=tp
        fp=np.sum(con_mat[:,i])-tp

        f1=(2*se*ppv)/(se+ppv)

        F1_score.append(f1)

    return F1_score


def sen(Y_test, Y_pred, n):  # n为分类
    sen = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return sen


def pre(Y_test, Y_pred, n):
    pre = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:, i]) - tp
        pre1 = tp / (tp + fp)
        pre.append(pre1)

    return pre


def spe(Y_test, Y_pred, n):
    spe = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe


def ACC(Y_test, Y_pred, n):
    acc = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)

    return acc

def npv(Y_test, Y_pred, n):
    npv= []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        npv1 = tn / (tn + fn)
        npv.append(npv1)

    return npv






