#!/usr/bin/env python3
import contextlib
import numpy as np
import random
import shutil
import os

import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    filename = os.path.join(checkpoint_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint["state_dict"])


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copied from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def store_preds_to_disk(tgts, preds, args, preds_raw=None, gates=None):
    if args.task_type == "multilabel":
        if args.task == "cmu-mosi":
            with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
                fw.write("\n".join([str(x) for x in preds]))
            with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
                fw.write("\n".join([str(x) for x in tgts]))
            with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
                fw.write(" ".join([str(l) for l in args.labels]))
        else:
            with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
                fw.write(
                    "\n".join([" ".join(["1" if x else "0" for x in p]) for p in preds])
                )
            with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
                fw.write(
                    "\n".join([" ".join(["1" if x else "0" for x in t]) for t in tgts])
                )
            with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
                fw.write(" ".join([l for l in args.labels]))

    else:
        with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
            fw.write("\n".join([str(x) for x in preds]))
        with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
            fw.write("\n".join([str(x) for x in tgts]))
        with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
            fw.write(" ".join([str(l) for l in args.labels]))
            
    if preds_raw is not None:
        np.save(os.path.join(args.savedir, "preds_raw.npy"), preds_raw)
    
    if gates is not None:
        np.save(os.path.join(args.savedir, "gates.npy"), gates)


def log_metrics(set_name, metrics, args, logger):
    if args.task_type == "multilabel":
        if args.task == "cmu-mosi":
            logger.info(
            "{}: Loss: {:.5f} | MAE: {:.5f} | Corr: {:.5f} | Accuracy_7: {:.5f} | Weighted F1: {:.5f}".format(
                set_name, metrics["loss"], metrics["mae"], metrics["corr"], metrics["accuracy_7"], metrics["weighted_f1"]
                )
            )
        elif args.task == "cmu-mosei":
            logger.info(
            "{}: Loss: {:.5f}\n|    Anger   |   Disgust  |    Fear    |    Happy   |     Sad    |  Surprise  |   Average  | APS_micro\n  WA: {:.3f} | WA: {:.3f} | WA: {:.3f} | WA: {:.3f} | WA: {:.3f} | WA: {:.3f} | WA: {:.3f} | APS: {:.3f}\n  F1: {:.3f} | F1: {:.3f} | F1: {:.3f} | F1: {:.3f} | F1: {:.3f} | F1: {:.3f} | F1: {:.3f}".format(
                set_name, metrics["loss"], metrics["wacc_emo1"]*100, metrics["wacc_emo2"]*100, metrics["wacc_emo3"]*100, metrics["wacc_emo4"]*100, metrics["wacc_emo5"]*100, metrics["wacc_emo6"]*100, metrics["wacc_emos"]*100, metrics["auc_pr_micro"]*100, metrics["f1_emo1"]*100, metrics["f1_emo2"]*100, metrics["f1_emo3"]*100, metrics["f1_emo4"]*100, metrics["f1_emo5"]*100, metrics["f1_emo6"]*100, metrics["f1_emos"]*100
                )
            )
        elif args.task == "mmimdb":
            logger.info(
                "{}: Loss: {:.5f}\n| Micro F1 {:.3f} | Macro F1: {:.3f} | Weighted F1: {:.3f} | Samples F1: {:.3f} | AP Micro: {:.3f}".format(
                    set_name, metrics["loss"], metrics["auc_pr_micro"]*100, metrics["macro_f1"]*100, metrics["auc_pr_macro"]*100, metrics["auc_pr_samples"]*100, metrics["micro_f1"]*100
                )
            )
        elif args.task == "counseling":
            logger.info(
                "{}: Loss: {:.5f}\n| F1 Low {:.3f} | F1 High: {:.3f} | Accuracy: {:.3f} | AP Micro: {:.3f}".format(
                    set_name, metrics["loss"], metrics["f1_low"]*100, metrics["f1_high"]*100, metrics["acc"]*100, metrics["auc_pr_micro"]*100
                )
            )
        else:
            logger.info(
                "{}: Loss: {:.5f}\n| Macro F1 {:.3f} | Micro F1: {:.3f} | AP Macro: {:.3f} | AP Micro: {:.3f} | AP Samples: {:.3f}".format(
                    set_name, metrics["loss"], metrics["macro_f1"]*100, metrics["micro_f1"]*100, metrics["auc_pr_macro"]*100, metrics["auc_pr_micro"]*100, metrics["auc_pr_samples"]*100
                )
            )
    else:
        logger.info(
            "{}: Loss: {:.5f} | MAE: {:.5f} | Corr: {:.5f} | Accuracy_7: {:.5f} | Weighted F1: {:.5f}".format(
                set_name, metrics["loss"], metrics["mae"], metrics["corr"], metrics["accuracy_7"], metrics["weighted_f1"]
            )
        )


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
