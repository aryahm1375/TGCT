import torch
import numpy as np
import os
import sys
import math
import logging
import json
from tqdm import tqdm, trange
from process_kumc_gct import get_datasets #add your dataset retrieval
from utils import *
from graph_convolutional_transformer import GraphConvolutionalTransformer
import pickle
from tensorboardX import SummaryWriter


def generate_new_index(proc1, proc2, proc3):
    all_keys = list(proc1.keeys()) + list(proc2.keys()) + list(proc3.keys())
    all_keys = np.unique(np.array(all_keys))
    new_dict = {}
    for i in range(0, len(all_keys)):
        new_dict[all_keys[i]] = i


def prediction_loop(args, model, dataloader1, priors_dataloader1, dataloader2, priors_dataloader2, dataloader3,
                    priors_dataloader3, description='Evaluating'):
    # batch_size = dataloader1.batch_size
    eval_losses = []
    preds = None
    label_ids = None
    model.eval()

    for data1, priors_data1, data2, priors_data2, data3, priors_data3 in tqdm(zip(dataloader1,
                                                                                  priors_dataloader1,
                                                                                  dataloader2,
                                                                                  priors_dataloader2,
                                                                                  dataloader3,
                                                                                  priors_dataloader3)):
        data1, priors_data1 = prepare_data(data1, priors_data1, args.device, 1)
        data2, priors_data2 = prepare_data(data2, priors_data2, args.device, 2)
        data3, priors_data3 = prepare_data(data3, priors_data3, args.device, 3)

        priors_data = priors_data1
        priors_data['indices1'] = priors_data1['indices']
        priors_data['values1'] = priors_data1['values']
        priors_data['indices2'] = priors_data2['indices']
        priors_data['values2'] = priors_data2['values']
        priors_data['indices3'] = priors_data3['indices']
        priors_data['values3'] = priors_data3['values']

        data = data1
        data['dx_ints1'] = data['dx_ints']
        data['proc_ints1'] = data['proc_ints']
        data['dx_masks1'] = data['dx_masks']
        data['proc_masks1'] = data['proc_masks']

        data['dx_ints2'] = data2['dx_ints']
        data['proc_ints2'] = data2['proc_ints']
        data['dx_masks2'] = data2['dx_masks']
        data['proc_masks2'] = data2['proc_masks']

        data['dx_ints3'] = data3['dx_ints']
        data['proc_ints3'] = data3['proc_ints']
        data['dx_masks3'] = data3['dx_masks']
        data['proc_masks3'] = data3['proc_masks']
        with torch.no_grad():
            outputs = model(data, priors_data)
            loss = outputs[0].mean().item()
            logits = outputs[1]

        labels = data[args.label_key]
        print(data1[args.label_key],data2[args.label_key],data3[args.label_key])
        batch_size = data[list(data.keys())[0]].shape[0]
        eval_losses.extend([loss] * batch_size)
        preds = logits if preds is None else nested_concat(preds, logits, dim=0)
        label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)

    # if preds is not None:
    #    preds = nested_numpify(preds)
    if label_ids is not None:
        label_ids = nested_numpify(label_ids)
    metrics = compute_metrics(preds, label_ids, description)

    metrics[f'{description}_loss'] = np.mean(eval_losses)

    for key in list(metrics.keys()):
        if not key.startswith('eval_'):
            metrics[f'{description}' + '_{}'.format(key)] = metrics.pop(key)

    return metrics


def main():
    args = ArgParser().parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger = logging.getLogger(__name__)
    with open(args.data_dir+'1/fold_0/dx.p', 'rb') as f:
        result1 = pickle.load(f)
        args.vocab_sizes['dx_ints1'] = len(result1)
        args.vocab_sizes['dx_ints2'] = len(result1)
        args.vocab_sizes['dx_ints3'] = len(result1)
    with open(args.data_dir+'1/fold_0/proc_map.p', 'rb') as f:
        result1 = pickle.load(f)
        args.vocab_sizes['proc_ints1'] = len(result1)
        args.vocab_sizes['proc_ints2'] = len(result1)
        args.vocab_sizes['proc_ints3'] = len(result1)

    logging.info("Arguments %s", args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging_dir = os.path.join(args.output_dir, 'logging')
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    tb_writer = SummaryWriter(log_dir=logging_dir)
    datasets1, prior_guides1 = get_datasets(args.data_dir + '1', fold=args.fold)
    train_dataset1, eval_dataset1, test_dataset1 = datasets1
    train_priors1, eval_priors1, test_priors1 = prior_guides1
    train_priors_dataset1 = eICUDataset(train_priors1)
    eval_priors_dataset1 = eICUDataset(eval_priors1)
    test_priors_dataset1 = eICUDataset(test_priors1)

    datasets2, prior_guides2 = get_datasets(args.data_dir + '2', fold=args.fold)
    train_dataset2, eval_dataset2, test_dataset2 = datasets2
    train_priors2, eval_priors2, test_priors2 = prior_guides2
    train_priors_dataset2 = eICUDataset(train_priors2)
    eval_priors_dataset2 = eICUDataset(eval_priors2)
    test_priors_dataset2 = eICUDataset(test_priors2)

    datasets3, prior_guides3 = get_datasets(args.data_dir + '3', fold=args.fold)
    train_dataset3, eval_dataset3, test_dataset3 = datasets3
    train_priors3, eval_priors3, test_priors3 = prior_guides3
    train_priors_dataset3 = eICUDataset(train_priors3)
    eval_priors_dataset3 = eICUDataset(eval_priors3)
    test_priors_dataset3 = eICUDataset(test_priors3)

    train_dataloader1 = DataLoader(train_dataset1, batch_size=args.batch_size)
    eval_dataloader1 = DataLoader(eval_dataset1, batch_size=args.batch_size)
    test_dataloader1 = DataLoader(test_dataset1, batch_size=args.batch_size)
    train_priors_dataloader1 = DataLoader(train_priors_dataset1, batch_size=args.batch_size,
                                          collate_fn=priors_collate_fn)
    eval_priors_dataloader1 = DataLoader(eval_priors_dataset1, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    test_priors_dataloader1 = DataLoader(test_priors_dataset1, batch_size=args.batch_size, collate_fn=priors_collate_fn)

    train_dataloader2 = DataLoader(train_dataset2, batch_size=args.batch_size)
    eval_dataloader2 = DataLoader(eval_dataset2, batch_size=args.batch_size)
    test_dataloader2 = DataLoader(test_dataset2, batch_size=args.batch_size)

    train_priors_dataloader2 = DataLoader(train_priors_dataset2, batch_size=args.batch_size,
                                          collate_fn=priors_collate_fn)
    eval_priors_dataloader2 = DataLoader(eval_priors_dataset2, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    test_priors_dataloader2 = DataLoader(test_priors_dataset2, batch_size=args.batch_size, collate_fn=priors_collate_fn)

    train_dataloader3 = DataLoader(train_dataset3, batch_size=args.batch_size)
    eval_dataloader3 = DataLoader(eval_dataset3, batch_size=args.batch_size)
    test_dataloader3 = DataLoader(test_dataset3, batch_size=args.batch_size)

    train_priors_dataloader3 = DataLoader(train_priors_dataset3, batch_size=args.batch_size,
                                          collate_fn=priors_collate_fn)
    eval_priors_dataloader3 = DataLoader(eval_priors_dataset3, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    test_priors_dataloader3 = DataLoader(test_priors_dataset3, batch_size=args.batch_size, collate_fn=priors_collate_fn)


    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device(f'cuda:{args.cuda}')
    if args.device.type == 'cuda':
        torch.cuda.set_device(args.device)

    if args.do_train:
        model = GraphConvolutionalTransformer(args)
        model = model.to(args.device)


        num_update_steps_per_epoch = len(train_dataloader1)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0)
        else:
            max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
            num_train_epochs = args.num_train_epochs
        num_train_epochs = int(np.ceil(num_train_epochs))

        args.eval_steps = num_update_steps_per_epoch // 2

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        warmup_steps = max_steps // (1 / args.warmup)

        logger.info('***** Running Training *****')
        logger.info(' Num examples = {}'.format(len(train_dataloader1.dataset)))
        logger.info(' Num epochs = {}'.format(num_train_epochs))
        logger.info(' Train batch size = {}'.format(args.batch_size))
        logger.info(' Total optimization steps = {}'.format(max_steps))

        epochs_trained = 0
        global_step = 0
        tr_loss = torch.tensor(0.0).to(args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        number = 40
        train_flag = False
        train_pbar = trange(epochs_trained, num_train_epochs, desc='Epoch')
        for epoch in range(epochs_trained, num_train_epochs):
            save_value = 0
            loss_values = []
            epoch_pbar = tqdm(train_dataloader1, desc='Iteration')

            check = False
            for data1, priors_data1, data2, priors_data2, data3, priors_data3 in zip(train_dataloader1,
                                                                                     train_priors_dataloader1,
                                                                                     train_dataloader2,
                                                                                     train_priors_dataloader2,
                                                                                     train_dataloader3,
                                                                                     train_priors_dataloader3):
                model.train()
                if epoch == 24 and not check:
                    save_value += 1
                data1, priors_data1 = prepare_data(data1, priors_data1, args.device, 1)
                data2, priors_data2 = prepare_data(data2, priors_data2, args.device, 2)
                data3, priors_data3 = prepare_data(data3, priors_data3, args.device, 3)
                priors_data = priors_data1
                priors_data['indices1'] = priors_data1['indices']
                priors_data['values1'] = priors_data1['values']
                priors_data['indices2'] = priors_data2['indices']
                priors_data['values2'] = priors_data2['values']
                priors_data['indices3'] = priors_data3['indices']
                priors_data['values3'] = priors_data3['values']

                data = data1
                data['dx_ints1'] = data['dx_ints']
                data['proc_ints1'] = data['proc_ints']
                data['dx_masks1'] = data['dx_masks']
                data['proc_masks1'] = data['proc_masks']
                data['patient_id1'] = data['patient_id']

                data['dx_ints2'] = data2['dx_ints']
                data['proc_ints2'] = data2['proc_ints']
                data['dx_masks2'] = data2['dx_masks']
                data['proc_masks2'] = data2['proc_masks']
                data['patient_id2'] = data2['patient_id']

                data['dx_ints3'] = data3['dx_ints']
                data['proc_ints3'] = data3['proc_ints']
                data['dx_masks3'] = data3['dx_masks']
                data['proc_masks3'] = data3['proc_masks']
                data['patient_id3'] = data3['patient_id']

                outputs = model(data, priors_data, save_value)
                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()
                loss_values.append(loss.item())
                loss.backward()

                tr_loss += loss.detach()
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    logs = {}
                    tr_loss_scalar = tr_loss.item()
                    logs['loss'] = (tr_loss_scalar - logging_loss_scalar) / args.logging_steps
                    logs['learning_rate'] = args.learning_rate  # scheduler.get_last_lr()[0]
                    logging_loss_scalar = tr_loss_scalar
                    if tb_writer:
                        for k, v in logs.items():
                            if isinstance(v, (int, float)):
                                tb_writer.add_scalar(k, v, global_step)
                        tb_writer.flush()
                    output = {**logs, **{"step": global_step}}

                    if (args.eval_steps > 0 and global_step % args.eval_steps == 0):
                        metrics = prediction_loop(args, model, eval_dataloader1, eval_priors_dataloader1, eval_dataloader2,
                                                  eval_priors_dataloader2, eval_dataloader3, eval_priors_dataloader3)
                        logger.info('**** Checkpoint Eval Results ****')
                        for key, value in metrics.items():
                            logger.info('{} = {}'.format(key, value))
                            tb_writer.add_scalar(key, value, global_step)
                        train_flag = not train_flag
                        if train_flag:
                            metrics = prediction_loop(args, model, train_dataloader1, train_priors_dataloader1,train_dataloader2, train_priors_dataloader2,train_dataloader3, train_priors_dataloader3, description = 'Training')
                            logger.info('**** Checkpoint TRAIN Results ****')
                            for key, value in metrics.items():
                                logger.info('{} = {}'.format(key, value))
                                tb_writer.add_scalar(key, value, global_step)

                epoch_pbar.update(1)
                if global_step >= max_steps:
                    break
            epoch_pbar.close()
            train_pbar.update(1)
            if global_step >= max_steps:
                break

        train_pbar.close()
        if tb_writer:
            tb_writer.close()

        logging.info('\n\nTraining completed')

def get_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        shape = param.shape
        param_size = 1
        for dim in shape:
            param_size *= dim
        print(name, shape, param_size)
        total_params += param_size
    print(total_params)


if __name__ == "__main__":
    main()

