#!/usr/bin/env python3

import argparse
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import json
from random import shuffle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_transformers.optimization import (
    AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)
from pytorch_pretrained_bert import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME

from mmbt.data.helpers import get_data_loaders
from mmbt.models import get_model
from mmbt.utils.logger import create_logger
from mmbt.utils.utils import *
from mmbt.models.vilbert import BertConfig

from os.path import expanduser
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased"])
    parser.add_argument("--data_path", type=str, default="/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="/path/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="mmtrvapt", choices=["mmtrvat", "mmtrvapt"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--visual", type=str, default="video", choices=["poster", "video", "both", "none"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/path/save_dir/")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--task", type=str, default="moviescope", choices=["iemocap", "mmimdb", "moviescope", "cmu-mosei", "cmu-mosi", "counseling"])
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument('--output_gates', action='store_true', help='Store GMU gates of test dataset to a file (default: false)')
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "att", "cls_att", "vert_att"], help='Type of pooling technique for BERT models')
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--train_type", type=str, default="split", choices=["split", "cross"], help='Use train-val-test splits or perform cross-validation')
    parser.add_argument("--just_test", action='store_true', help='If true, the model will not be trained.')
    parser.add_argument("--from_seed", type=int, default=1)
    parser.add_argument("--inverse_seed", action='store_true', help='If true, training seeds will be inversed.')
    parser.add_argument("--hybrid", action='store_true', help='If true, the model BPMulT will be complete with hybrid fusion.')

    '''MMTransformer parameters'''
    parser.add_argument('--vonly', action='store_false', help='use the crossmodal fusion into v (default: False)')
    parser.add_argument('--lonly', action='store_false', help='use the crossmodal fusion into l (default: False)')
    parser.add_argument('--aonly', action='store_false', help='use the crossmodal fusion into a (default: False)')
    parser.add_argument("--orig_d_v", type=int, default=2048)
    parser.add_argument("--orig_d_l", type=int, default=768)
    parser.add_argument("--orig_d_a", type=int, default=96)
    parser.add_argument("--orig_d_p", type=int, default=4096)
    parser.add_argument("--v_len", type=int, default=3)
    parser.add_argument("--l_len", type=int, default=512)
    parser.add_argument("--a_len", type=int, default=3)
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout')
    parser.add_argument('--attn_dropout_v', type=float, default=0.0, help='attention dropout (for visual)')
    parser.add_argument('--attn_dropout_a', type=float, default=0.0, help='attention dropout (for audio)')
    parser.add_argument('--relu_dropout', type=float, default=0.1, help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25, help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1, help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0, help='output layer dropout')
    parser.add_argument('--nlevels', type=int, default=5, help='number of layers in the network (default: 5)')
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--num_heads', type=int, default=5, help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--attn_mask', action='store_false', help='use attention mask for Transformer (default: true)')
        

def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes and args.task != "cmu-mosi":
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        if args.weight_classes and args.task != "cmu-mosi":
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.CrossEntropyLoss(weight=label_weights.cuda())
        else:
            if args.task == "cmu-mosi":
                criterion = nn.L1Loss()
                #criterion = nn.BCEWithLogitsLoss()
                #criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            
    return criterion


def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return optimizer


def get_scheduler(optimizer, args):
    if args.task == 'cmu-mosi':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=args.lr_patience, verbose=True, factor=args.lr_factor
        )
    else:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
        )

def weighted_acc(preds, truths, verbose):
    total = len(preds)
    tp = 0
    tn = 0
    p = 0
    n = 0
    for i in range(total):
        if truths[i] == 0:
            n += 1
            if preds[i] == 0:
                tn += 1
        elif truths[i] == 1:
            p += 1
            if preds[i] == 1:
                tp += 1

    w_acc = (tp * n / p + tn) / (2 * n)

    #if verbose:
    fp = n - tn
    fn = p - tp
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
  
    return w_acc, f1

def model_eval(i_epoch, data, model, args, criterion, store_preds=False, output_gates=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        all_gates = []  # For gmu gate interpretability
        raw_preds = []
        for batch in data:
            if output_gates:
                loss, out, tgt, gates = model_forward(i_epoch, model, args, criterion, batch, output_gates)
            else:
                loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
                raw_preds.append(torch.sigmoid(out).cpu().detach().numpy())
            else:
                if args.task == "cmu-mosi":
                    pred = torch.sigmoid(out).cpu().detach().numpy()
                    raw_preds.append(out.cpu().detach().numpy())
                else:
                    pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
                    raw_preds.append(torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy())

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)
            if output_gates:
                gates = gates.cpu().detach().numpy()
                all_gates.append(gates)

    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        raw_preds = np.vstack(raw_preds)
        if args.task == 'moviescope':
            metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
            metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
            metrics["auc_pr_macro"] = average_precision_score(tgts, raw_preds, average="macro")
            metrics["auc_pr_micro"] = average_precision_score(tgts, raw_preds, average="micro")
            metrics["auc_pr_samples"] = average_precision_score(tgts, raw_preds, average="samples")
        elif args.task == 'mmimdb':
            metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
            metrics["micro_f1"] = average_precision_score(tgts, raw_preds, average="micro")
            metrics["auc_pr_macro"] = f1_score(tgts, preds, average="weighted")
            metrics["auc_pr_micro"] = f1_score(tgts, preds, average="micro")
            metrics["auc_pr_samples"] = f1_score(tgts, preds, average="samples")
        elif args.task == 'counseling':
            metrics["f1_low"] = f1_score(tgts, preds, average=None)[1]
            metrics["f1_high"] = f1_score(tgts, preds, average=None)[0]
            metrics["acc"] = accuracy_score(tgts, preds)
            metrics["auc_pr_micro"] = average_precision_score(tgts, raw_preds, average="micro")
            accs = []
            f1s = []
            for emo_ind in range(2):
                preds_i = preds[:, emo_ind]
                truths_i = tgts[:, emo_ind]
                wacc, f1 = weighted_acc(preds_i, truths_i, verbose=False)
                accs.append(wacc) #weighted_acc(preds_i, truths_i, verbose=False))
                f1s.append(f1) #f1_score(truths_i, preds_i)) #, average='weighted'))
            accs.append(np.average(accs))
            f1s.append(np.average(f1s))
            metrics["f1_low"] = f1s[1]
            metrics["f1_high"] = f1s[0]
      #      metrics["auc_pr_micro"] = metrics["acc"] #accs[2]
        elif args.task == 'cmu-mosei':
            accs = []
            f1s = []
            for emo_ind in range(6):
                preds_i = preds[:, emo_ind]
                truths_i = tgts[:, emo_ind]
                wacc, f1 = weighted_acc(preds_i, truths_i, verbose=False)
                accs.append(wacc) #weighted_acc(preds_i, truths_i, verbose=False))
                f1s.append(f1) #f1_score(truths_i, preds_i)) #, average='weighted'))
            accs.append(np.average(accs))
            f1s.append(np.average(f1s))
            metrics["f1_emo1"] = f1s[0]
            metrics["f1_emo2"] = f1s[1]
            metrics["f1_emo3"] = f1s[2]
            metrics["f1_emo4"] = f1s[3]
            metrics["f1_emo5"] = f1s[4]
            metrics["f1_emo6"] = f1s[5]
            metrics["wacc_emo1"] = accs[0]
            metrics["wacc_emo2"] = accs[1]
            metrics["wacc_emo3"] = accs[2]
            metrics["wacc_emo4"] = accs[3]
            metrics["wacc_emo5"] = accs[4]
            metrics["wacc_emo6"] = accs[5]
            metrics["f1_emos"] = f1s[6]
            metrics["wacc_emos"] = average_precision_score(tgts, raw_preds, average="micro") #accs[6]
            metrics["auc_pr_micro"] = accs[6]#metrics["wacc_emos"] #average_precision_score(tgts, raw_preds, average="micro")
        #metrics["auc_pr_samples"] = np.average(roc_auc_score(tgts, raw_preds, labels=list(range(6)), average=None).tolist())
    else:
        raw_preds = np.array([p for sub in raw_preds for p in sub])
        tgts = np.array([p for sub in tgts for p in sub])
        preds = np.array([p for sub in preds for p in sub])
        predict = preds*6 - 3
        metrics["mae"] = np.mean(np.absolute(predict - tgts))   # Average L1 distance between preds and truths
        metrics["corr"] = np.corrcoef(predict, tgts)[0][1]
        metrics["accuracy_7"] = np.sum(np.round(predict) == np.round(tgts)) / float(len(tgts))
        non_zeros = np.array([i for i, e in enumerate(tgts) if e != 0])
        binary_truth = (tgts[non_zeros] > 0)
        binary_preds = (predict[non_zeros] > 0)
        metrics["weighted_f1"] = f1_score(binary_preds, binary_truth, average='weighted')
        print("Accuracy_2:", accuracy_score(binary_truth, binary_preds))
        metrics["weight_f1"] = metrics["mae"]

    if store_preds:
        if output_gates:
            all_gates = np.vstack(all_gates)
            print("gates: ", all_gates.shape)
            store_preds_to_disk(tgts, preds, args, preds_raw=raw_preds, gates=all_gates)
        else:
            store_preds_to_disk(tgts, preds, args, preds_raw=raw_preds)

    return metrics


def model_forward(i_epoch, model, args, criterion, batch, gmu_gate=False):

    if args.task in ["moviescope", "mmimdb"]:
        if args.model == "mmtrvapt":
            txt, segment, mask, img, tgt, audio, poster = batch
            #print(tgt.size(), img.size())
        else:
            txt, segment, mask, tgt = batch
            
    elif args.task in ["cmu-mosei", "cmu-mosi", "counseling", "iemocap"]:
        if args.model == "mmtrvat":
            txt, segment, mask, img, tgt, audio = batch
        else:
            txt, segment, mask, tgt = batch
            
    else:
        txt, segment, mask, img, tgt, _ = batch

    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt
    
    device = next(model.parameters()).device
    
    
    if args.model == "mmtrvapt":
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        img, audio, poster = img.cuda(), audio.cuda(), poster.cuda()
        if gmu_gate:
            out, gates = model(txt, mask, segment, img, audio, poster, gmu_gate)
        else:
            out = model(txt, mask, segment, img, audio, poster)
    elif args.model == "mmtrvat":
         # For cmu-mosei dataset
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        img, audio = img.cuda(), audio.cuda()
        if gmu_gate:
            out, gates = model(txt, mask, segment, img, audio, gmu_gate)
        else:
            out = model(txt, mask, segment, img, audio)
    
    if args.task == "cmu-mosi":
        if args.task_type == "multilabel":
            tgt = torch.tensordot(tgt, torch.FloatTensor([i/100 for i in range(101)]), dims = ([1],[0]))
            tgt = tgt.unsqueeze(1)
        else:
            #tgt = tgt.long()
            #tgt = tgt.unsqueeze(1)
            out = out.squeeze(1)
            #out = out.long()
    tgt = tgt.to(device)
    loss = criterion(out, tgt)
    
    if gmu_gate:
        return loss, out, tgt, gates
    else:
        return loss, out, tgt


def train(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)
    
    if args.trained_model_dir: # load in fine-tuned (with cloze-style LM objective) model
        args.previous_state_dict_dir = os.path.join(args.trained_model_dir, WEIGHTS_NAME)

    model = get_model(args)
        
    cuda_len = torch.cuda.device_count()
    if cuda_len > 1:
        model = nn.DataParallel(model)
        #torch.distributed.init_process_group(backend='nccl', world_size=2)
        #model = nn.parallel.DistributedDataParallel(model, device_ids=[0,1], output_device=1)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve = 0, 0, 0
    best_metric = np.inf if args.task == 'cmu-mosi' else -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Training..")
    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()
        
        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, _, _ = model_forward(i_epoch, model, args, criterion, batch)
            torch.cuda.empty_cache()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        model.eval()
        metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["auc_pr_micro"] if args.task_type == "multilabel" else metrics["weight_f1"]
        )
        scheduler.step(tuning_metric)
        if args.task == "cmu-mosi":
            is_improvement = tuning_metric <= best_metric
        else:
            is_improvement = tuning_metric >= best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1
        if is_improvement:
            save_checkpoint(
                {
                    "epoch": i_epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "n_no_improve": n_no_improve,
                    "best_metric": best_metric,
                },
                is_improvement,
                args.savedir,
            )

        if args.task == "cmu-mosi":
            if n_no_improve >= args.patience:
                logger.info("No improvement. Breaking out of loop.")
                break
        else:
            if n_no_improve >= args.patience:
                logger.info("No improvement. Breaking out of loop.")
                break

    #torch.cuda.empty_cache()
    #load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    #model.eval()

    #test_metrics = model_eval(
    #    np.inf, test_loader, model, args, criterion, store_preds=True, output_gates=args.output_gates
    #)
    #log_metrics(f"Test - ", test_metrics, args, logger)
    

def test(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)

    _, _, test_loader = get_data_loaders(args)
    
    if args.trained_model_dir: # load in fine-tuned (with cloze-style LM objective) model
        args.previous_state_dict_dir = os.path.join(args.trained_model_dir, WEIGHTS_NAME)

    model = get_model(args)
    
    cuda_len = torch.cuda.device_count()
    if cuda_len > 1:
        model = nn.DataParallel(model)

    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    model.cuda()

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()

    test_metrics = model_eval(
        np.inf, test_loader, model, args, criterion, store_preds=True, output_gates=args.output_gates
    )
    logger = create_logger("%s/logfileTest.log" % args.savedir, args)
    log_metrics(f"Test - ", test_metrics, args, logger)
    

def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    if args.train_type == "split":
        savedir = args.savedir
        for i in range(args.from_seed, 6):
            if args.inverse_seed:
                i = 6-i
            args.seed = i
            args.savedir = savedir
            args.name = f'moviescope_Seed{i}_mmbt_model_run'

            #You can perform a training or just the test
            if args.just_test:
                test(args)
            else:
                train(args)
                args.savedir = savedir
                test(args)
    else:
        cross_validation_train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
