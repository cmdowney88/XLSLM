"""
Script to train or use a Segmental Language Model using PyTorch, with the model
specification being read from a configuration .json file

Can be run in ``train`` or ``eval``/``predict`` mode. Train mode can either
initialize a new model or continue training a pre-existing model. Eval mode
requires the model to already exist. See ``main()`` for argument details

Authors:
    C.M. Downey (cmdowney@uw.edu)
"""
import argparse
import gc
import logging
import os
import random
import sys
import time
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple

import torch
import yaml
import chonker.chonktorch.functions as ctf
import chonker.wrangle as wr
import numpy as np
import torch.nn as nn
from chonker.chonktorch.data import VariableLengthDataset
from sklearn.metrics import matthews_corrcoef as mcc_metric
from sklearn.metrics import precision_recall_fscore_support as fscore
from torch.utils.data import DataLoader, RandomSampler

from .mslm_config import MSLMConfig
from .dev_config import devConfig
from .slm import SegmentalLanguageModel

# ------------------------------------------------------------------------------
# Auxiliary Function Definitions
# ------------------------------------------------------------------------------


def import_or_create_subword_vocab(
    vocab: wr.Vocab,
    subword_vocab_file: str = None,
    train_file: str = None,
    model_path: str = None,
    max_seg_length: int = None,
    lexicon_min_count: int = None,
    preserve_case: bool = None
) -> Dict[tuple, int]:
    """
    Get a subword/segment vocabulary, either by importing or creating it.
    Returns a dictionary from character-index tuples to integer subword indices

    Args:
        vocab: The Vocab object for the model
        subword_vocab_file: Path to file from which to import subword vocab. If
            `None`, will create subword vocab from provided train file
        train_file: Path to train text file from which to create subword vocab.
            Must be provided if `subword_vocab_file` is `None`. Default: `None`
        model_path: Base path to which to write the subword vocab file. Must be
            provided if `subword_vocab_file` is `None`. Default: `None`
        max_seg_length: The maximum segment length for the SLM model. Must be
            provided if `subword_vocab_file` is `None`. Default: `None`
        lexicon_min_count: The minimum count at which to keep gathered
            subwords/segments. Must be provided if `subword_vocab_file` is
            `None`. Default: `None`
        preserve_case: Whether to preserve the case of the train file from which
            the subwords are gathered. Must be provided if `subword_vocab_file`
            is `None`. Default: `None`
    Returns:
        A mapping from tuples of character/symbol indices to the index of the
            subword that they constitute
    """
    if subword_vocab_file:
        with open(subword_vocab_file, 'r') as f:
            id_to_subword = yaml.load(f, Loader=yaml.SafeLoader)
            id_to_subword = {
                key: tuple(value)
                for key, value in id_to_subword.items()
            }
        subword_to_id = {}
        for index in id_to_subword.keys():
            subword_to_id[id_to_subword[index]] = index
    elif train_file:
        train_text_wo_edges = wr.character_tokenize(
            train_file, preserve_case=preserve_case
        )
        _, subword_vocab = wr.get_ngrams(
            train_text_wo_edges,
            max_seg_length,
            min_length=2,
            min_count=lexicon_min_count
        )
        subword_to_id = subword_vocab[0]
        id_to_subword = subword_vocab[1]
        next_index = len(subword_to_id)
        subword_to_id[('<eos>', )] = next_index
        id_to_subword[next_index] = ('<eos>', )
        id_to_subword = {
            key: list(value)
            for key, value in id_to_subword.items()
        }
        subword_vocab_file = model_path + '_subword_vocab.yaml'
        with open(subword_vocab_file, 'w') as f:
            yaml.dump(id_to_subword, f)
    else:
        raise ValueError(
            "Must either specify a subword vocab file or a train"
            " text file to get a subword vocab"
        )

    char_ids_to_subword_id = {}
    for subword in subword_to_id.keys():
        char_ids = tuple(vocab.to_ids(list(subword)))
        char_ids_to_subword_id[char_ids] = subword_to_id[subword]

    return char_ids_to_subword_id


def get_boundary_vector(sequence: List[List[str]]) -> List[int]:
    """
    Extract a binary vector representing segmentation boundaries from a list of
    segments

    The vector's length is equal to the number of symbols in the
    sequence, where for every symbol, vector(x) = 1 if there is a boundary
    after the symbol at position x, and 0 if there is no boundary after the
    symbol at x. If the segmentation is considered "gold", the last value will
    be 1, otherwise 0

    Args:
        sequence: A list of segments, which are themselves lists of string
            symbols, from which to construct a (segment) boundary vector. Needs
            to be a list of lists because some items representing one "symbol"
            may be more than one character (e.g. `<tag>`)
    Returns:
        A list vector representing the segment boundaries (1 for boundary, 0 for
            no boundary)
    """

    lengths = [len(segment) for segment in sequence]
    num_symbols = sum(lengths)
    vector = [0 for x in range(num_symbols)]
    current_index = 0
    for length in lengths[:-1]:
        current_index += length
        vector[current_index - 1] = 1
    return vector[:-1]


def do_eval(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_seg_len: int,
    eoseg_idx: int,
    chars_to_subword_id: dict,
    vocab: wr.Vocab,
    gold_boundaries: List[int],
    unsort_permutation: List[int],
    length_exponent: float = None,
    length_penalty_lambda: float = None,
    clear_cache_by_batch: bool = False
) -> Tuple[dict, list]:
    """
    Run model evaluation over a set of input data

    Output total loss as well as optimal sequence segmentations as determined
    by the Viterbi algorithm. Runs a forward pass on each input sequence in eval
    mode (model does not regress or track gradients), outputting scores for each
    possible segment of each sequence

    Args:
        model: The model to run eval with
        data_loader: The DataLoader containing the eval/dev/predict data
        device: The PyTorch device on which the model resides
        max_seg_len: The maximum segment length for the SLM model
        eoseg_idx: The index of the end-of-segment token
        chars_to_subword_id:
        vocab: The Vocab object for the model
        gold_boundaries: The list of gold boundary vectors for the data
        unsort_permutation: The permutation with which to unsort the data
        length_exponent: The exponent with which to raise segment lengths when
            doing expected length regularization. Default: `None`
        length_penalty_lambda: The lambda for combining the expected length
            penalty with the main bits per character loss. Default: `None`
        clear_cache_by_batch: Whether to run `torch.cuda.empty_cache()` after
            each batch to clear up memory
    Returns:
        stat_dict: A dictionary containing various eval metrics (dev loss, mcc,
            f1, precision, recall)
        dev_segmentations: A list of the dev segmentations in their original
            order. Each sequence is List[List[str]] (see `get_boundary_vector`)
    """

    forward_args = {
        'data': None,
        'lengths': None,
        'max_seg_len': max_seg_len,
        'eoseg_index': eoseg_idx,
        'chars_to_subword_id': chars_to_subword_id,
        'length_exponent': length_exponent,
        'length_penalty_lambda': length_penalty_lambda
    }

    segmentations = []
    num_dev_batches = len(data_loader)
    total_dev_loss = 0

    for batch in data_loader:
        batch_lengths = batch[1]
        cuda_batch = batch[0].to(device)
        forward_args['data'] = cuda_batch
        forward_args['lengths'] = batch_lengths
        if length_penalty_lambda:
            loss, bpc_loss, best_paths, _ = model.forward(**forward_args)
            total_dev_loss += bpc_loss.item() / num_dev_batches
        else:
            loss, best_paths, _ = model.forward(**forward_args)
            total_dev_loss += loss.item() / num_dev_batches
        del cuda_batch
        gc.collect()
        if clear_cache_by_batch:
            torch.cuda.empty_cache()

        # Convert vocab IDs to characters and segment the sequence for
        # comparison to the gold
        for j in range(len(best_paths)):
            raw_string = vocab.to_tokens(batch[0][1:, j].tolist())
            segmentations.append(
                [
                    raw_string[segment[0]:segment[1]]
                    for segment in best_paths[j][:-1]
                ]
            )

    dev_segmentations = [segmentations[i] for i in unsort_permutation]
    dev_boundaries = [get_boundary_vector(ex) for ex in dev_segmentations]
    all_dev_boundaries = np.array(wr.flatten(dev_boundaries))
    prec, rec, f1, _ = fscore(
        gold_boundaries, all_dev_boundaries, average='binary'
    )
    mcc = mcc_metric(gold_boundaries, all_dev_boundaries)

    # Convert dev loss to log base 2 to represent bits per character
    dev_loss = total_dev_loss / np.log(2)

    # Print out the initial f1 score and some sample segmentations
    stat_dict = {
        "dev loss": round(dev_loss, 3),
        "mcc": round(mcc, 3),
        "f1": round(100 * f1, 1),
        "precision": round(100 * prec, 1),
        "recall": round(100 * rec, 1)
    }

    return stat_dict, dev_segmentations


def statbar_string(stat_dict: dict) -> str:
    """
    Return a printable "statbar" string from a dictionary of named statistics
    """
    stat_items = []
    for key, value in stat_dict.items():
        stat_items.append(f"{key} {value}")
    return ' | '.join(stat_items)


def dev_file_load(
    file_list: List[str], config: dict, vocab: wr.Vocab, pad_idx: int
) -> Dict:
    """
    Create a DataLoader and gold segmentations for the dev file(s)

    Args:
        file_list: The list of dev files
        config: The configuration file for training
        vocab: The vocabulary created from the training text
        pad_idx: The index for the <pad> token
    """
    # Tokenize the dev file by characters, adding <bos> and <eos> tags.
    # Convert text to integer ids based on Vocab, and read into Dataset and
    # Dataloader
    dev_texts = []
    for d_file in file_list:
        d_text = wr.character_tokenize(
            d_file, preserve_case=config.preserve_case, edge_tokens=True
        )
        dev_texts.append(d_text)

    dev_datas = []
    for d_text in dev_texts:
        d_data = [vocab.to_ids(line) for line in d_text]
        dev_datas.append(d_data)

    dev_sets = []
    for d_data in dev_datas:
        d_set = VariableLengthDataset(
            d_data,
            batch_size=config.batch_size,
            pad_value=pad_idx,
            batch_by=config.batch_by,
            max_padding=config.max_padding,
            max_pad_strategy='split'
        )
        dev_sets.append(d_set)

    dev_dataloaders = []
    for d_set in dev_sets:
        d_dataloader = DataLoader(d_set, batch_size=None)
        dev_dataloaders.append(d_dataloader)

    gold_dev_texts = []
    gold_boundaries = []
    all_gold_boundaries = []

    # Also whitespace-tokenize the dev data, using the spaces to establish
    # gold-standard segmentations for the dev data, which are converted to a
    # binary "boundary" vector using get_boundary_vector
    for d_file in file_list:
        counter = 0
        gold_d_text = [
            wr.chars_from_words(sent) for sent in wr.basic_tokenize(
                d_file, preserve_case=config.preserve_case, split_tags=True
            )
        ]
        gold_d_text = gold_d_text[:dev_sets[counter].total_num_instances]
        gold_dev_texts.append(gold_d_text)

        g_boundaries = [get_boundary_vector(ex) for ex in gold_d_text]
        gold_boundaries.append(g_boundaries)

        all_g_boundaries = np.array(wr.flatten(g_boundaries))
        all_gold_boundaries.append(all_g_boundaries)
        counter += 1

    return {
        'data_loaders': dev_dataloaders,
        'gold_boundaries': all_gold_boundaries,
        'unsort_permutation': [dev_set.unsort_pmt for dev_set in dev_sets]
    }


def compute_eval_args(
    model, dev_dataloaders, device, max_seg_length, eoseg_idx,
    char_ids_to_subword_id, vocab, all_gold_boundaries, dev_set_unsort_pmt,
    clear_cache_by_batch
) -> List[dict]:
    """
    Create a dictionary containing eval arguments corresponding to a dev file
    """

    list_eval_args = []
    for i in range(len(dev_dataloaders)):
        eval_args = {
            'model': model,
            'data_loader': dev_dataloaders[i],
            'device': device,
            'max_seg_len': max_seg_length,
            'eoseg_idx': eoseg_idx,
            'chars_to_subword_id': char_ids_to_subword_id,
            'vocab': vocab,
            'gold_boundaries': all_gold_boundaries[i],
            'unsort_permutation': dev_set_unsort_pmt[i],
            'clear_cache_by_batch': clear_cache_by_batch
        }
        list_eval_args.append(eval_args)

    return list_eval_args


def eval_step_metrics(
    eval_args,
    logger,
    global_step=0,
    max_train_steps=0,
    time_per_batch=0.0,
    lr=0.0,
    current_train_loss=0.0,
    mode='both',
    print_sample=True
) -> List[float]:
    """
    Perform an eval run and return metrics based on the mode of evaluation
    """
    # Perform an eval run on the dev data
    with torch.no_grad():
        dev_stat_dict, dev_segmentations = do_eval(**eval_args)

    if print_sample and (global_step != 0):
        # Print the current step, batch, learning rate, average time
        # per batch, training loss, dev f1, and example segmentations
        # for the current model
        dev_stat_dict.update(
            {
                "step": f"{global_step}/{max_train_steps}",
                "s/batch": time_per_batch,
                "lr": round(lr, 5),
                "train loss": round(current_train_loss, 3),
            }
        )

    if print_sample:
        logger.info(statbar_string(dev_stat_dict))
        print("Sample dev segmentations:")
        for seg in dev_segmentations[:8]:
            print("    " + ' '.join([''.join(segment) for segment in seg]))

    if mode == 'bpc':
        metrics = [
            dev_stat_dict['dev loss'], dev_stat_dict['precision'],
            dev_stat_dict['recall']
        ]
    elif mode == 'seg':
        metrics = [
            dev_stat_dict['mcc'], dev_stat_dict['f1'],
            dev_stat_dict['precision'], dev_stat_dict['recall']
        ]
    else:
        metrics = [
            dev_stat_dict['dev loss'], dev_stat_dict['mcc'],
            dev_stat_dict['f1'], dev_stat_dict['precision'],
            dev_stat_dict['recall']
        ]

    return metrics


def eval_save_models(
    model_architecture,
    model_state_dict,
    optimizer_state_dict,
    scheduler_state_dict,
    vocab_file,
    subword_vocab_file,
    logger,
    dev_loss,
    mcc,
    f1,
    best_loss,
    best_mcc,
    best_f1,
    checkpoints_wo_improvement,
    is_final,
    args_model_path,
    config_early_stopping,
    global_step,
    mode='both'
):

    return_dict = {
        'checkpoints_wo_improvement': checkpoints_wo_improvement,
        'best_loss': best_loss,
        'best_mcc': best_mcc,
        'best_f1': best_f1,
        'early_stop': False
    }

    # If the dev loss from the current checkpoint is better than
    # the current best, save the current model to a file
    checkpoint = {
        'model_architecture': model_architecture,
        'state_dict': model_state_dict,
        'optimizer': optimizer_state_dict,
        'scheduler': scheduler_state_dict,
        'vocab_file': vocab_file,
        'subword_vocab_file': subword_vocab_file
    }
    if dev_loss < return_dict['best_loss']:
        return_dict['best_loss'] = dev_loss
        torch.save(checkpoint, args_model_path + '_best_loss.model')
        logger.info(f"New best loss at step {global_step}")
        return_dict['checkpoints_wo_improvement'] = 0
    else:
        return_dict['checkpoints_wo_improvement'] += 1
        improvement_stopped = (
            config_early_stopping and
            return_dict['checkpoints_wo_improvement'] >= config_early_stopping
        )
        if improvement_stopped:
            logger.info(
                f"Stopping early at step {global_step} due to no"
                f" dev loss improvement in {config_early_stopping}"
                " checkpoints"
            )
            torch.save(checkpoint, args_model_path + '_final.model')
            logger.info(f"Saved model at step {global_step} as final model")
            return_dict['early_stop'] = True
            return return_dict
    if mode != 'bpc' and (mcc > return_dict['best_mcc']):
        return_dict['best_mcc'] = mcc
        torch.save(checkpoint, args_model_path + '_best_mcc.model')
        logger.info(f"New best mcc at step {global_step}")
    if mode != 'bpc' and (f1 > return_dict['best_f1']):
        return_dict['best_f1'] = f1
        torch.save(checkpoint, args_model_path + '_best_f1.model')
        logger.info(f"New best f1 at step {global_step}")
    if is_final:
        torch.save(checkpoint, args_model_path + '_final.model')

    return return_dict


# ------------------------------------------------------------------------------
# Train and Predict Functions
# ------------------------------------------------------------------------------


def train(args, config, dev_config, device, logger) -> None:
    """
    Train a Segmental Language Model
    """

    # Tokenize the train file by characters, adding <bos> and <eos>
    # tags
    train_text = wr.character_tokenize(
        args.train_file, preserve_case=config.preserve_case, edge_tokens=True
    )

    #
    pretrained_embeddings = None
    char_ids_to_subword_id = None

    if args.preexisting_model:
        if args.load_model_path is None:
            raise ValueError(
                'Filepath for loading previous model not specified'
            )
        checkpoint = torch.load(args.load_model_path + '.model')
        vocab_file = checkpoint['vocab_file']
        vocab = wr.Vocab.from_saved(vocab_file)
        subword_vocab_file = checkpoint['subword_vocab_file']
        if subword_vocab_file:
            char_ids_to_subword_id = import_or_create_subword_vocab(
                vocab, subword_vocab_file
            )
    else:
        vocab = wr.Vocab(train_text, other_tokens=['<pad>', '<eoseg>'])
        vocab_file = args.model_path + '_vocab.yaml'
        vocab.save(vocab_file)
        subword_vocab_file = args.model_path + '_subword_vocab.yaml'
        if config.use_lexicon:
            char_ids_to_subword_id = import_or_create_subword_vocab(
                vocab,
                train_file=args.train_file,
                model_path=args.model_path,
                max_seg_length=config.max_seg_length,
                lexicon_min_count=config.lexicon_min_count,
                preserve_case=config.preserve_case
            )
        if config.pretrained_embedding:
            pretrained_embeddings = ctf.import_embeddings(
                config.pretrained_embedding,
                vocab,
                init_range=0.1,
                logger=logger
            )

    vocab_size = vocab.size()
    pad_idx = vocab.tok_to_id['<pad>']
    eoseg_idx = vocab.tok_to_id['<eoseg>']

    # If using the lexicon, log its size
    subword_vocab_size = None
    if config.use_lexicon:
        subword_vocab_size = len(char_ids_to_subword_id)
        logger.info(f"Subword lexicon size: {subword_vocab_size}")

    # Convert the train text to integer ids based on the Vocab mapping
    train_data = [vocab.to_ids(line) for line in train_text]

    # Delete the lines that are less than or equal to the max segment length
    # (Causes errors with masking schemes)
    train_data = [
        line for line in train_data if len(line) > config.max_seg_length
    ]

    # Read the training data into a PyTorch Dataset and Dataloader
    train_set = VariableLengthDataset(
        train_data,
        batch_size=config.batch_size,
        pad_value=pad_idx,
        batch_by=config.batch_by,
        max_padding=config.max_padding
    )
    random_sampler = RandomSampler(train_set)
    train_dataloader = DataLoader(
        train_set, batch_size=None, sampler=random_sampler
    )

    # dev file using command line argument
    if args.dev_file:
        dev_load_dict = dev_file_load([args.dev_file], config, vocab, pad_idx)
        dev_dataloader = dev_load_dict['data_loaders'][0]
        all_gold_boundaries = dev_load_dict['gold_boundaries'][0]
        dev_set_unsort_pmt = dev_load_dict['unsort_permutation'][0]
    #primary dev file in dev_config
    elif args.dev_config:
        dev_load_dict = dev_file_load(
            [dev_config.primary_dev_file], config, vocab, pad_idx
        )
        dev_dataloader = dev_load_dict['data_loaders'][0]
        all_gold_boundaries = dev_load_dict['gold_boundaries'][0]
        dev_set_unsort_pmt = dev_load_dict['unsort_permutation'][0]

        #bpc mode secondary dev files
        if dev_config.bpc_secondary_dev_files:
            dev_load_dict = dev_file_load(
                dev_config.bpc_secondary_dev_files, config, vocab, pad_idx
            )
            bpc_secondary_dev_dataloaders = dev_load_dict['data_loaders']
            bpc_secondary_all_gold_boundaries = dev_load_dict['gold_boundaries']
            bpc_secondary_dev_set_unsort_pmt = dev_load_dict[
                'unsort_permutation']

        #seg qual mode secondary dev files
        if dev_config.seg_secondary_dev_files:
            dev_load_dict = dev_file_load(
                dev_config.seg_secondary_dev_files, config, vocab, pad_idx
            )
            seg_secondary_dev_dataloaders = dev_load_dict['data_loaders']
            seg_secondary_all_gold_boundaries = dev_load_dict['gold_boundaries']
            seg_secondary_dev_set_unsort_pmt = dev_load_dict[
                'unsort_permutation']

        #both mode secondary dev files
        if dev_config.both_secondary_dev_files:
            dev_load_dict = dev_file_load(
                dev_config.both_secondary_dev_files, config, vocab, pad_idx
            )
            both_secondary_dev_dataloaders = dev_load_dict['data_loaders']
            both_secondary_all_gold_boundaries = dev_load_dict['gold_boundaries'
                                                              ]
            both_secondary_dev_set_unsort_pmt = dev_load_dict[
                'unsort_permutation']

    # If training an existing model, read it from the checkpoint and load in the
    # parameters, else create a new model instantiation based on the input
    # configuration
    if args.preexisting_model:
        model_architecture = checkpoint['model_architecture']
        model = SegmentalLanguageModel(model_architecture).to(device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model_architecture = {
            'vocab_size': vocab_size,
            'use_lexicon': config.use_lexicon,
            'subword_vocab_size': subword_vocab_size,
            'model_dim': config.model_dim,
            'model_dropout': config.model_dropout,
            'pretrained_embedding': pretrained_embeddings,
            'enc_type': config.encoder_type,
            'mask_type': config.transformer_mask_type,
            'num_enc_layers': config.num_encoder_layers,
            'encoder_dim': config.encoder_dim,
            'num_heads': config.num_heads,
            'ffwd_dim': config.feedforward_dim,
            'encoder_dropout': config.encoder_dropout,
            'num_dec_layers': config.num_decoder_layers,
            'autoencoder': config.autoencoder,
            'attention_window': config.attention_window,
            'smart_position': config.smart_position
        }
        model = SegmentalLanguageModel(model_architecture).to(device)

    # Have the positional embedding proportion be learned at half the rate
    # as the rest of the parameters
    slowed_param_names = [
        'encoder.positional_proportion.weight',
        'encoder.positional_proportion.bias'
    ]
    slowed_params = [
        tensor for name, tensor in model.named_parameters()
        if name in slowed_param_names
    ]
    params = [
        tensor for name, tensor in model.named_parameters()
        if name not in slowed_param_names
    ]
    param_list = [
        {
            'params': params
        }, {
            'params': slowed_params,
            'lr': config.learning_rate / 2
        }
    ]

    # Initialize the optimizer using either Stochastic Gradient Descent or Adam
    if config.optimizer_algorithm == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=config.learning_rate)
    elif config.optimizer_algorithm == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=config.learning_rate)
    else:
        raise ValueError(
            f'Optimizer mode {config.optimizer_algorithm} is not valid'
        )

    # Get the learning rate lambda
    lr_lambda = ctf.get_lr_lambda_by_steps(
        config.max_train_steps,
        num_warmup_steps=config.num_warmup_steps,
        warmup=config.warmup,
        decay=config.decay,
        gamma=config.gamma,
        gamma_steps=config.gamma_steps
    )

    # Initialize the scheduler based on the learning rate lambda
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda
    )

    # Write the metric field headers to the output csv file(s)
    fixed_metrics = ["step", "s/batch", "lr", "train_loss"]
    all_metrics = ["dev_loss", "mcc", "f1", "precision", "recall"]
    bpc_metrics = ["dev_loss", "precision", "recall"]
    seg_metrics = ["mcc", "f1", "precision", "recall"]
    metric_file_headers = []
    if args.dev_file:
        metric_file_headers = [
            os.path.basename(args.dev_file) + '_' + metric
            for metric in all_metrics
        ]
    elif args.dev_config:
        if dev_config.primary_dev_mode == 'bpc':
            metric_file_headers += [
                os.path.basename(dev_config.primary_dev_file) + '_' + metric
                for metric in bpc_metrics
            ]
        elif dev_config.primary_dev_mode == 'seg':
            metric_file_headers += [
                os.path.basename(dev_config.primary_dev_file) + '_' + metric
                for metric in seg_metrics
            ]
        elif dev_config.primary_dev_mode == 'both':
            metric_file_headers += [
                os.path.basename(dev_config.primary_dev_file) + '_' + metric
                for metric in all_metrics
            ]

        if dev_config.bpc_secondary_dev_files:
            for d_file in dev_config.bpc_secondary_dev_files:
                metric_file_headers += [
                    os.path.basename(d_file) + '_' + metric
                    for metric in bpc_metrics
                ]
        if dev_config.seg_secondary_dev_files:
            for d_file in dev_config.seg_secondary_dev_files:
                metric_file_headers += [
                    os.path.basename(d_file) + '_' + metric
                    for metric in seg_metrics
                ]
        if dev_config.both_secondary_dev_files:
            for d_file in dev_config.both_secondary_dev_files:
                metric_file_headers += [
                    os.path.basename(d_file) + '_' + metric
                    for metric in all_metrics
                ]

    metric_file_headers = fixed_metrics + metric_file_headers
    with open(args.model_path + '.csv', 'w+') as data_file:
        print(', '.join(metric_file_headers), file=data_file)

    #Make eval_args
    if args.dev_file:
        eval_args = compute_eval_args(
            model, [dev_dataloader], device, config.max_seg_length, eoseg_idx,
            char_ids_to_subword_id, vocab, [all_gold_boundaries],
            [dev_set_unsort_pmt], config.clear_cache_by_batch
        )[0]

    elif args.dev_config:
        #primary dev file
        eval_args = compute_eval_args(
            model, [dev_dataloader], device, config.max_seg_length, eoseg_idx,
            char_ids_to_subword_id, vocab, [all_gold_boundaries],
            [dev_set_unsort_pmt], config.clear_cache_by_batch
        )[0]

        #bpc mode secondary dev files
        if dev_config.bpc_secondary_dev_files:
            list_of_bpc_secondary_eval_args = compute_eval_args(
                model, bpc_secondary_dev_dataloaders, device,
                config.max_seg_length, eoseg_idx, char_ids_to_subword_id, vocab,
                bpc_secondary_all_gold_boundaries,
                bpc_secondary_dev_set_unsort_pmt, config.clear_cache_by_batch
            )

        #seg qual mode secondary dev files
        if dev_config.seg_secondary_dev_files:
            list_of_seg_secondary_eval_args = compute_eval_args(
                model, seg_secondary_dev_dataloaders, device,
                config.max_seg_length, eoseg_idx, char_ids_to_subword_id, vocab,
                seg_secondary_all_gold_boundaries,
                seg_secondary_dev_set_unsort_pmt, config.clear_cache_by_batch
            )

        #both mode secondary dev files
        if dev_config.both_secondary_dev_files:
            list_of_both_secondary_eval_args = compute_eval_args(
                model, both_secondary_dev_dataloaders, device,
                config.max_seg_length, eoseg_idx, char_ids_to_subword_id, vocab,
                both_secondary_all_gold_boundaries,
                both_secondary_dev_set_unsort_pmt, config.clear_cache_by_batch
            )

    train_args = {
        'data': None,
        'lengths': None,
        'max_seg_len': config.max_seg_length,
        'eoseg_index': eoseg_idx,
        'chars_to_subword_id': char_ids_to_subword_id,
        'length_exponent': config.length_exponent,
        'length_penalty_lambda': config.length_penalty_lambda
    }

    # Initialize base metrics and counters, begin training
    best_loss = float("inf")
    best_f1 = 0
    best_mcc = -1
    forward_batch = 0
    global_step = 0
    checkpoints_wo_improvement = 0
    early_stop = False

    logger.info(f"Starting Training")
    model.train()

    while global_step < config.max_train_steps and not early_stop:

        # At the very beginning of training, do an eval run and record baseline
        # metrics, writing to the output metric file
        if global_step == 0:
            metrics_list = [
                str(metric) for metric in [
                    global_step, "n/a",
                    round(scheduler.get_last_lr()[0], 7), "n/a"
                ]
            ]
            if args.dev_file:
                metrics = eval_step_metrics(eval_args, logger)
                metrics_list += [str(m) for m in metrics]

            elif args.dev_config:
                if dev_config.primary_dev_mode == 'bpc':
                    metrics = eval_step_metrics(eval_args, logger, mode='bpc')
                    metrics_list += [str(m) for m in metrics]

                elif dev_config.primary_dev_mode == 'seg':
                    metrics = eval_step_metrics(eval_args, logger, mode='seg')
                    metrics_list += [str(m) for m in metrics]

                elif dev_config.primary_dev_mode == 'both':
                    metrics = eval_step_metrics(eval_args, logger, mode='both')
                    metrics_list += [str(m) for m in metrics]

                if dev_config.bpc_secondary_dev_files:
                    for e_args in list_of_bpc_secondary_eval_args:
                        metrics = eval_step_metrics(
                            e_args, logger, mode='bpc', print_sample=False
                        )
                        metrics_list += [str(m) for m in metrics]

                if dev_config.seg_secondary_dev_files:
                    for e_args in list_of_seg_secondary_eval_args:
                        metrics = eval_step_metrics(
                            e_args, logger, mode='seg', print_sample=False
                        )
                        metrics_list += [str(m) for m in metrics]

                if dev_config.both_secondary_dev_files:
                    for e_args in list_of_both_secondary_eval_args:
                        metrics = eval_step_metrics(
                            e_args, logger, mode='both', print_sample=False
                        )
                        metrics_list += [str(m) for m in metrics]

            with open(args.model_path + '.csv', 'a+') as data_file:
                print(', '.join(metrics_list), file=data_file)

            checkpoint_start_time = time.time()
            checkpoint_stats = defaultdict(list)

        # Loop through the input data by batch
        for batch in train_dataloader:

            # Call forward function to get the segment probabilities and total
            # loss for the batch
            batch_lengths = batch[1]
            cuda_batch = batch[0].to(device)
            train_args['data'] = cuda_batch
            train_args['lengths'] = batch_lengths
            forward_start_time = time.time()
            if config.length_penalty_lambda:
                batch_loss, _, _, time_profile = model.forward(**train_args)
            else:
                batch_loss, _, time_profile = model.forward(**train_args)
            checkpoint_stats['forward_time'].append(
                time.time() - forward_start_time
            )
            checkpoint_stats['nn_time'].append(time_profile['nn'])
            checkpoint_stats['lattice_time'].append(time_profile['lattice'])
            backward_start_time = time.time()
            batch_loss.backward()
            batch_loss = batch_loss.item() / np.log(2)
            checkpoint_stats['loss'].append(batch_loss)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            checkpoint_stats['backward_time'].append(
                time.time() - backward_start_time
            )
            lr = scheduler.get_last_lr()[0]
            forward_batch += 1
            is_global_step = (
                forward_batch % config.gradient_accumulation_steps == 0
            )
            if is_global_step:
                optimizer_start_time = time.time()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                checkpoint_stats['optimizer_time'].append(
                    time.time() - optimizer_start_time
                )
                global_step += 1
            del cuda_batch
            gc.collect()
            if config.clear_cache_by_batch:
                torch.cuda.empty_cache()

            is_checkpoint = (
                is_global_step and global_step % config.checkpoint_interval == 0
            )
            is_final = (
                is_global_step and global_step == config.max_train_steps
            )

            # When at a checkpoint interval, provide printouts of model
            # statistics as well as the results of a dev run, as well as saving
            # the model if it is the best performing
            if is_checkpoint or is_final:

                # Track the time elapsed for the current checkpoint interval,
                # as well as the total loss
                elapsed = (time.time() - checkpoint_start_time)
                batches_in_checkpoint = len(checkpoint_stats['loss'])
                time_per_batch = round(elapsed / batches_in_checkpoint, 2)
                current_train_loss = mean(checkpoint_stats['loss'])

                time_profile_dict = {
                    "average forward time":
                        f"{round(mean(checkpoint_stats['forward_time']), 3)}s",
                    "forward nn":
                        f"{round(mean(checkpoint_stats['nn_time']), 3)}s",
                    "forward lattice":
                        f"{round(mean(checkpoint_stats['lattice_time']), 3)}s",
                    "backward":
                        f"{round(mean(checkpoint_stats['backward_time']), 3)}s",
                    "optimizer":
                        f"{round(mean(checkpoint_stats['optimizer_time']), 3)}s"
                }
                logger.info(statbar_string(time_profile_dict))

                # Initialize the metrics list with fixed parameters
                metrics_list = [
                    str(metric) for metric in [
                        global_step, time_per_batch,
                        round(lr, 7),
                        round(current_train_loss, 2)
                    ]
                ]

                if args.dev_file:
                    metrics = eval_step_metrics(
                        eval_args, logger, global_step, config.max_train_steps,
                        time_per_batch, lr, current_train_loss
                    )
                    metrics_list += [str(m) for m in metrics]
                    dev_loss, mcc, f1 = metrics[0], metrics[1], metrics[2]

                    return_dict = eval_save_models(
                        model_architecture, model.state_dict(),
                        optimizer.state_dict(), scheduler.state_dict(),
                        vocab_file, subword_vocab_file, logger, dev_loss, mcc,
                        f1, best_loss, best_mcc, best_f1,
                        checkpoints_wo_improvement, is_final, args.model_path,
                        config.early_stopping, global_step
                    )
                    checkpoints_wo_improvement = return_dict[
                        'checkpoints_wo_improvement']
                    best_loss = return_dict['best_loss']
                    best_mcc = return_dict['best_mcc']
                    best_f1 = return_dict['best_f1']
                    early_stop = return_dict['early_stop']
                    if is_final or early_stop:
                        break

                elif args.dev_config:
                    sec_metrics_list = []
                    if dev_config.bpc_secondary_dev_files:
                        for e_args in list_of_bpc_secondary_eval_args:
                            metrics = eval_step_metrics(
                                e_args,
                                logger,
                                global_step,
                                mode='bpc',
                                print_sample=False
                            )
                            sec_metrics_list += [str(m) for m in metrics]

                    if dev_config.seg_secondary_dev_files:
                        for e_args in list_of_seg_secondary_eval_args:
                            metrics = eval_step_metrics(
                                e_args,
                                logger,
                                global_step,
                                mode='seg',
                                print_sample=False
                            )
                            sec_metrics_list += [str(m) for m in metrics]

                    if dev_config.both_secondary_dev_files:
                        for e_args in list_of_both_secondary_eval_args:
                            metrics = eval_step_metrics(
                                e_args,
                                logger,
                                global_step,
                                mode='both',
                                print_sample=False
                            )
                            sec_metrics_list += [str(m) for m in metrics]

                    if dev_config.primary_dev_mode == 'bpc':
                        metrics = eval_step_metrics(
                            eval_args,
                            logger,
                            global_step,
                            config.max_train_steps,
                            time_per_batch,
                            lr,
                            current_train_loss,
                            mode='bpc'
                        )
                        metrics_list += [
                            str(m) for m in metrics
                        ] + sec_metrics_list
                        dev_loss = metrics[0]

                        return_dict = eval_save_models(
                            model_architecture,
                            model.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            vocab_file,
                            subword_vocab_file,
                            logger,
                            dev_loss,
                            None,
                            None,
                            best_loss,
                            best_mcc,
                            best_f1,
                            checkpoints_wo_improvement,
                            is_final,
                            args.model_path,
                            config.early_stopping,
                            global_step,
                            mode='bpc'
                        )
                        checkpoints_wo_improvement = return_dict[
                            'checkpoints_wo_improvement']
                        best_loss = return_dict['best_loss']
                        early_stop = return_dict['early_stop']
                        if is_final or early_stop:
                            break

                    elif dev_config.primary_dev_mode == 'seg':
                        metrics = eval_step_metrics(
                            eval_args,
                            logger,
                            global_step,
                            config.max_train_steps,
                            time_per_batch,
                            lr,
                            current_train_loss,
                            mode='seg'
                        )
                        metrics_list += [
                            str(m) for m in metrics
                        ] + sec_metrics_list
                        dev_loss, mcc, f1 = metrics[0], metrics[1], metrics[2]

                        return_dict = eval_save_models(
                            model_architecture,
                            model.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            vocab_file,
                            subword_vocab_file,
                            logger,
                            dev_loss,
                            mcc,
                            f1,
                            best_loss,
                            best_mcc,
                            best_f1,
                            checkpoints_wo_improvement,
                            is_final,
                            args.model_path,
                            config.early_stopping,
                            global_step,
                            mode='seg'
                        )
                        checkpoints_wo_improvement = return_dict[
                            'checkpoints_wo_improvement']
                        best_loss = return_dict['best_loss']
                        best_mcc = return_dict['best_mcc']
                        best_f1 = return_dict['best_f1']
                        early_stop = return_dict['early_stop']
                        if is_final or early_stop:
                            break

                    elif dev_config.primary_dev_mode == 'both':
                        metrics = eval_step_metrics(
                            eval_args,
                            logger,
                            global_step,
                            config.max_train_steps,
                            time_per_batch,
                            lr,
                            current_train_loss,
                            mode='both'
                        )
                        metrics_list += [
                            str(m) for m in metrics
                        ] + sec_metrics_list
                        dev_loss, mcc, f1 = metrics[0], metrics[1], metrics[2]

                        return_dict = eval_save_models(
                            model_architecture,
                            model.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            vocab_file,
                            subword_vocab_file,
                            logger,
                            dev_loss,
                            mcc,
                            f1,
                            best_loss,
                            best_mcc,
                            best_f1,
                            checkpoints_wo_improvement,
                            is_final,
                            args.model_path,
                            config.early_stopping,
                            global_step,
                            mode='both'
                        )
                        checkpoints_wo_improvement = return_dict[
                            'checkpoints_wo_improvement']
                        best_loss = return_dict['best_loss']
                        best_mcc = return_dict['best_mcc']
                        best_f1 = return_dict['best_f1']
                        early_stop = return_dict['early_stop']
                        if is_final or early_stop:
                            break

                with open(args.model_path + '.csv', 'a+') as data_file:
                    print(', '.join(metrics_list), file=data_file)

                # Reset the checkpoint loss and start time
                checkpoint_start_time = time.time()
                checkpoint_stats = defaultdict(list)

    with open(args.model_path + '.csv', 'a+') as data_file:
        print(', '.join(metrics_list), file=data_file)
    logger.info("Training finished")


def predict(args, config, device, logger):

    start_time = time.time()

    checkpoint = torch.load(args.model_path + '.model', map_location=device)
    vocab = wr.Vocab.from_saved(checkpoint['vocab_file'])
    eoseg_idx = vocab.tok_to_id['<eoseg>']
    pad_idx = vocab.tok_to_id['<pad>']

    subword_vocab_file = checkpoint['subword_vocab_file']
    char_ids_to_subword_id = None
    if config.use_lexicon and subword_vocab_file:
        char_ids_to_subword_id = import_or_create_subword_vocab(
            vocab, subword_vocab_file
        )

    model = SegmentalLanguageModel(checkpoint['model_architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    input_text = wr.character_tokenize(
        args.input_file, preserve_case=config.preserve_case, edge_tokens=True
    )
    input_data = [vocab.to_ids(line) for line in input_text]
    num_lines = len(input_data)
    input_set = VariableLengthDataset(
        input_data,
        batch_size=16384,
        batch_by='tokens',
        pad_value=pad_idx,
        drop_final=False
    )
    input_dataloader = DataLoader(input_set, batch_size=None)

    # Also whitespace-tokenize the dev data, using the spaces to establish
    # gold-standard segmentations for the dev data, which are converted to a
    # binary "boundary" vector using get_boundary_vector
    gold_input_text = [
        wr.chars_from_words(sent) for sent in wr.basic_tokenize(
            args.input_file,
            preserve_case=config.preserve_case,
            split_tags=True
        )
    ]
    gold_boundaries = [get_boundary_vector(ex) for ex in gold_input_text]
    all_gold_boundaries = np.array(wr.flatten(gold_boundaries))

    eval_args = {
        'model': model,
        'data_loader': input_dataloader,
        'device': device,
        'max_seg_len': config.max_seg_length,
        'eoseg_idx': eoseg_idx,
        'chars_to_subword_id': char_ids_to_subword_id,
        'vocab': vocab,
        'gold_boundaries': all_gold_boundaries,
        'unsort_permutation': input_set.unsort_pmt
    }

    with torch.no_grad():
        stat_dict, segmentations = do_eval(**eval_args)

    elapsed = round(time.time() - start_time, 2)

    stat_dict.update({"elapsed time": f"{elapsed}s"})

    print(statbar_string(stat_dict), file=sys.stderr)

    for seg in segmentations:
        print(' '.join([''.join(segment) for segment in seg]))


# ------------------------------------------------------------------------------
# Main Script
# ------------------------------------------------------------------------------


def main():

    # Set the device for tensor storage to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # Set logging configuration
    logger = logging.getLogger(__name__)
    out_handler = logging.StreamHandler(sys.stdout)
    message_format = '%(asctime)s - %(message)s'
    date_format = '%m-%d-%y %H:%M:%S'
    out_handler.setFormatter(logging.Formatter(message_format, date_format))
    out_handler.setLevel(logging.INFO)
    logger.addHandler(out_handler)
    logger.setLevel(logging.INFO)

    # Take model hyperparameters from input, otherwise using default values
    parser = argparse.ArgumentParser(
        description='Training script for unsupervised segmentation'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='File containing the training data'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='File to which best model parameters will be saved'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        help='Whether model should be opened for training or eval'
    )
    parser.add_argument(
        '--preexisting_model',
        action='store_true',
        help='Flag indicating whether script is starting from previous model'
    )
    parser.add_argument(
        '--load_model_path',
        type=str,
        default=None,
        help='File from which previous model parameters will be loaded'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default=None,
        help='JSON file from which to load the script configuration'
    )
    parser.add_argument(
        '--dev_file',
        type=str,
        default=None,
        help='File containing the dev data'
    )
    parser.add_argument(
        '--dev_config',
        type=str,
        default=None,
        help='JSON file from which to load the dev data files configuration'
    )
    args = parser.parse_args()
    args.preexisting_model = bool(args.preexisting_model)

    #Make sure user uses either dev_file or dev_config, not both or neither
    if args.dev_file and args.dev_config:
        raise ValueError(f'Only use --dev_file or --dev_config, not both')
    if not args.dev_file and not args.dev_config:
        raise ValueError(f'Must use --dev_file or --dev_config')

    # Read in the configuration file if one is supplied, else use the default
    # configuration
    if hasattr(args, "config_file"):
        config = MSLMConfig.from_json(args.config_file)
    else:
        config = MSLMConfig()

    #Read in dev configuration file if one is supplied
    if args.dev_config:
        dev_config = devConfig.from_json(args.dev_config)
        #If there is a dev config file, it must have a primary dev file
        #and a primary dev mode
        if not dev_config.primary_dev_file:
            raise ValueError(
                f'Primary dev file {dev_config.primary_dev_file} is not valid'
            )
        if dev_config.primary_dev_mode not in ['bpc', 'seg', 'both']:
            raise ValueError(
                f'Primary dev file mode {dev_config.primary_dev_mode} is not valid'
            )
    else:
        dev_config = None

    # Set the random seed for all necessary packages
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Call the proper function (train/predict) based on what mode the script is
    # run in
    if args.mode == 'train':
        logger.info('Model Configuration:')
        print(config.__dict__)
        args.train_file = args.input_file
        train(
            args=args,
            config=config,
            dev_config=dev_config,
            device=device,
            logger=logger
        )
    elif args.mode == 'eval' or args.mode == 'predict':
        predict(args=args, config=config, device=device, logger=logger)
    else:
        raise ValueError(f'Model mode of {args.mode} is not valid')


if __name__ == "__main__":
    main()
