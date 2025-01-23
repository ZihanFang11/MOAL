import copy
import warnings
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from args_tNet import parameter_parser
from util.loadMatData import load_data, features_to_Lap, generate_partition
import scipy.sparse as sp
from util.label_utils import reassign_labels, special_train_test_split
from sklearn import metrics
from MSLNet import MSLNet_classfier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from util.config import load_config

np.set_printoptions(threshold=np.inf)

def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    F1_macro = metrics.f1_score(labels_true, labels_pred, average='macro')
    F1_micro = metrics.f1_score(labels_true, labels_pred, average='micro')

    return ACC, F1_macro, F1_micro

def gather_nd(params, indices):

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    out = torch.take(params, idx)

    return out.view(out_shape)

def evaluate(logits, mask_indices, show_matrix=False, filter_unseen=True,threshold=None):

    if isinstance(logits, list):
        logits = logits.mean(0)
        if args.use_softmax:
            probs_list = torch.softmax(logits, axis=1)
        else:
            probs_list = torch.sigmoid(logits)
        probs = probs_list.mean(0)
    else:
        probs = logits_to_probs(logits)

    masked_logits = logits[mask_indices]
    masked_y_pred = torch.argmax(masked_logits, 1)
    masked_y_true = y_true[mask_indices]
    if filter_unseen:
        probs = probs[mask_indices]
        probs = gather_nd(probs, torch.stack([torch.arange(masked_logits.size(0)), masked_y_pred], axis=1))
        probs = probs.cpu().detach().numpy()
        masked_y_pred = masked_y_pred.numpy()
        print("mean: ", probs.mean())
        if threshold is None:
            threshold = (probs[masked_y_true != args.unseen_label_index].mean()+probs[masked_y_true == args.unseen_label_index].mean())/2.0
            print("auto meanS: ", threshold)
        # threshold
        masked_y_pred[probs < threshold] = args.unseen_label_index

    else:
        masked_y_pred = masked_y_pred.numpy()

    masked_y_true = masked_y_true
    accuracy = accuracy_score(masked_y_true, masked_y_pred)
    macro_f_score = f1_score(masked_y_true, masked_y_pred, average="macro")

    if show_matrix:
        print(classification_report(masked_y_true, masked_y_pred))
        print(confusion_matrix(masked_y_true, masked_y_pred))

    if filter_unseen:
        return accuracy, macro_f_score, threshold
    else:
        return accuracy, macro_f_score

def logits_to_probs(logits):
    if args.use_softmax:
        probs = torch.softmax(logits, dim=1)
        # probs = F.log_softmax(logits, dim=1)
    else:
        probs = torch.nn.sigmoid(logits)
    return probs


def compute_loss(outputs, labels, mask_indices):

    logits = outputs
    labels = labels.long()

##################################################

    all_indices = np.arange(0, logits.size(0))
    unmasked_indices = np.delete(all_indices, mask_indices)
    unmasked_logits = logits[unmasked_indices]

    unmasked_probs = logits_to_probs(unmasked_logits)
    unmasked_probs = torch.clamp(unmasked_probs, 1e-7, 1.0)
    unmasked_preds = torch.argmax(unmasked_probs, 1).to(device)
    unmasked_prob = gather_nd(unmasked_probs, torch.stack([torch.arange(unmasked_logits.size(0)).to(device), unmasked_preds], axis=1))

    topk_indices_a = torch.logical_and(torch.greater(unmasked_prob, 1.0 / num_classes), torch.less(unmasked_prob, 0.5))
    topk_indices_b = torch.arange(topk_indices_a.size(0)).to(device)
    topk_indices = torch.masked_select(topk_indices_b, topk_indices_a)

    unmasked_probs = unmasked_probs[topk_indices]
    loss_unseen = (unmasked_probs * torch.log(unmasked_probs)).mean()

    #####################################################################################

    logits = F.log_softmax(logits, dim=1)
    masked_logits = logits[mask_indices]
    masked_y_true = labels[mask_indices].to(device)
    # h = F.one_hot(masked_y_true, num_classes)
    loss_seen = torch.nn.NLLLoss()(masked_logits, masked_y_true)
    #####################################################################################

    loss = args.lambda1 * loss_seen + args.lambda2 * loss_unseen

    return loss

#########################################################################################################################################################################

def train(args, device, features, labels):
    print(
        " unseen_num:{},fusion:{}, layer_num:{}, training_rate:{}, lambda1:{}, lambda2:{}, gamma:{}, epoch:{} \n".format(
         args.unseen_num, args.fusion_type, args.layer_num, args.training_rate,
            args.lambda1, args.lambda2, args.gamma, args.epoch))
    model = MSLNet_classfier(n_feats, n_view, num_classes, n, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ACC = 0
    begin_time = time.time()
    for epoch in range(args.epoch):
        t = time.time()
        Z_logit = model(features, lap)
        optimizer.zero_grad()
        loss = compute_loss(Z_logit, y_true, train_indices)
        loss.backward()

        optimizer.step()

        train_accuracy, macro_f_score = evaluate(Z_logit.cpu(), train_indices, filter_unseen=False)
        valid_accuracy, valid_macro_f_score, threshold = evaluate(Z_logit.cpu(), valid_indices, filter_unseen=True)
        if valid_accuracy >= best_ACC:
            best_ACC = valid_accuracy
            best_thre = threshold
            best_model_wts = copy.deepcopy(model.state_dict())


        print("Epoch:", '%04d' % (epoch + 1), "best_acc=", "{:.5f}".format(best_ACC),
              "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(train_accuracy), "val_acc=", "{:.5f}".format(valid_accuracy),
              "threshold=", "{:.5f}".format(threshold), "time=", "{:.5f}".format(time.time() - t))

    with torch.no_grad():
        model.load_state_dict(best_model_wts)
        Z_logit = model(features, lap)
        test_accuracy, test_macro_f_score, _ = evaluate(Z_logit.cpu(), test_indices, show_matrix=False,
                                                        filter_unseen=True,
                                                        threshold=best_thre)
    print( "test_acc=", "{:.5f}".format(test_accuracy), "test_macro_f_score=", "{:.5f}".format(test_macro_f_score),)
    run_time = time.time()-begin_time


    with open(args.save_file, "a") as f:
        f.write(
            "unseen_num:{}, fusion:{},  layer_num:{}, training_rate:{}, lambda1:{}, lambda2:{}, gamma:{}, epoch:{} \n".format(
                args.unseen_num,args.fusion_type, args.layer_num, args.training_rate, args.lambda1, args.lambda2, args.gamma, args.epoch))
        f.write("{}:{}\n".format(dataset, dict(
            zip(['acc', 'F1_macro', 'time'],
                [round( test_accuracy * 100, 2), round(test_macro_f_score * 100, 2),  round(run_time, 4)]))))




def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)


    dataset_dict = {1: 'esp_game', 2: 'Hdigit',3: 'NoisyMNIST_15000',4: 'OutdoorScene',5:  "Wikipedia_full" ,
                    6:"Caltech101all",7:"smallReuters",8:"UCI"}


    select_dataset = [1,2,3,4,5,6,8,7]
    select_dataset = [6]


    args.save_file = 'result.txt'
    for ii in select_dataset:
        dataset=dataset_dict[ii]
        config = load_config(args.config_name)
        args.layer_num = config[dataset]
        features, labels= load_data(dataset, args.data_path)
        labels = labels + 1


        n_view = len(features)
        n_feats = [x.shape[1] for x in features]
        n = features[0].shape[0]
        n_classes = len(np.unique(labels))

        feature_list = []
        for i in range(n_view):
            feature_list.append(features[i])
            features[i] = torch.from_numpy(features[i] / 1.0).float().to(device)


        original_num_classes = np.max(labels) + 1

        seen_labels = list(range(1, original_num_classes - args.unseen_num))
        y_true = reassign_labels(labels, seen_labels, args.unseen_label_index)

        train_indices, test_valid_indices = special_train_test_split(y_true, args.unseen_label_index,
                                                                     test_size=1 - args.training_rate)
        valid_indices, test_indices = generate_partition(y_true[test_valid_indices], test_valid_indices,
                                                         args.valid_rate / (1 - args.training_rate))

        num_classes = np.max(y_true) + 1
        y_true = torch.from_numpy(y_true)
        print('data:{}\tseen_labels:{}\tuse_softmax:{}\trandom_seed:{}\tunseen_num:{}\tnum_classes:{}'.format(
            dataset,
            seen_labels,
            args.use_softmax,
            args.seed,
            args.unseen_num,
            num_classes))

        print(dataset, n, n_view, n_feats,n_classes)
        labels = torch.from_numpy(labels).long().to(device)



        lap = features_to_Lap(dataset,feature_list,device, args.knn)
        if args.fix_seed:
            torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        train(args, device, features, y_true)

        with open(args.save_file, "a") as f:
            f.write("\n")
            continue

