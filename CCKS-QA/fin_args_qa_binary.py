# -*- coding:utf-8 -*-
"""
@Project ：eeqa_yc 
@File    ：fin_args_qa_binary.py
@IDE     ：PyCharm 
@Author  ：yc1999
@Date    ：2021/4/30 15:30 
"""
from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import time
import re
import string
import sys
from io import open
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import ArgExtractor, BertForQuestionAnswering_withIfTriggerEmbedding
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from transformers import BertTokenizer

import pickle
from tqdm import tqdm
from pathlib import Path

"""Run QA on argument extraction (with threshold)."""
# TODO: ~~筛选出不重要的论元，保证论元的质量~~
# TODO: 重新确定超参数max_answer_length的大小，这个需要对数据进行分析之后才能得到结果
# TODO: 是否考虑删除不包含论元，论元超出句子范围的样本，原作者的代码已经进行了处理
# TODO: Normal File还没有生成的，不想生成了
# TODO: TriggerInformation，在这个模型里面是没有的


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

max_seq_length = 180

class AceExample(object):
    """
    A single training/test example for the ace dataset.
    """

    def __init__(self, sentence, events, s_start):
        self.sentence = sentence
        self.events = events
        self.s_start = s_start

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "event sentence: %s" % (" ".join(self.sentence))
        event_triggers = []
        for event in self.events:
            if event:
                event_triggers.append(self.sentence[event[0][0] - self.s_start])
                event_triggers.append(event[0][1])
                event_triggers.append(str(event[0][0] - self.s_start))
                event_triggers.append("|")
        s += " ||| event triggers: %s" % (" ".join(event_triggers))
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens, token_to_orig_map, input_ids, input_mask, segment_ids, if_trigger_ids,
                 #
                 event_type, argument_type, fea_trigger_offset, sentence_offset,
                 #
                 start_position=None, end_position=None,
                 #
                 labels = None):

        self.example_id = example_id
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.if_trigger_ids = if_trigger_ids

        self.event_type = event_type
        self.argument_type = argument_type
        self.fea_trigger_offset = fea_trigger_offset
        self.sentence_offset = sentence_offset

        self.start_position = start_position
        self.end_position = end_position

        self.labels = labels

def read_ace_examples(input_file, cached_examples_file, is_training):
    """Read a ACE json file into a list of AceExample."""
    # TODO: 这里的example是包含所有数据的，我们的小样本需要筛选出包含argument的样本，不能包含不包含argument的样本。
    # 这里面一个AceExample包含多个event，但是这并不影响我在金融数据集上的实验
    raw_data = []

    # 使用pickle文件进行加载
    if 'pkl' in str(input_file):
        with open(str(input_file), 'rb') as f:
            raw_data= pickle.load(f)
    else:
        raw_data = input_file

    examples = []
    if cached_examples_file.exists():
        logger.info("Loading examples from cached file %s", cached_examples_file)
        examples = torch.load(cached_examples_file)
    else:
        for example in tqdm(raw_data,desc="read_ace_examples"):
            sentence, events, s_start = example["sentence"], example["event"], example["s_start"]
            example = AceExample(sentence=sentence, events=events, s_start=s_start)
            examples.append(example)
        logger.info("Saving examples into cached file %s", cached_examples_file)
        if(cached_examples_file.parent.exists() == False):
            cached_examples_file.parent.mkdir(parents=True)
        torch.save(examples, cached_examples_file)
    return examples

def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens

def convert_examples_to_features(examples, tokenizer, query_templates, nth_query, is_training, cached_features_file):
    """
    Loads a data file into a list of `InputBatch`s.
    :param examples: shape of [num_examples], a list of AceExample objects
    :param tokenizer: BertTokenzier
    :param query_templates: dict of (dict of list), 内置问题模板
    :param nth_query: 选择第几种query,一种有6种方式
    :param is_training: 是否是训练模式
    :return: 是存在负样本的
    """

    features = []
    if cached_features_file.exists():
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        return features
    else:

        for (example_id, example) in enumerate(tqdm(examples,desc="convert_examples_to_features")):
            # 需要遍历所有的事件，每一个事件里面的每一个argument作为一个训练样本
            for event in example.events:
                trigger_offset = event[0][0] - example.s_start
                event_type = event[0][2]
                trigger_token = example.sentence[event[0][0]:event[0][1]+1]    # 找出trigger是哪个词语
                arguments = event[1:]   # 得到所有的argument，[[73, 73, 'Vehicle'], [78, 78, 'Artifact'], [82, 82, 'Destination']]
                for argument_type in query_templates[event_type]:   # 遍历这个事件类型拥有的所有argument
                    if argument_type == "trigger":  # 需要跳过触发词类型的论元
                        continue
                    query = query_templates[event_type][argument_type][nth_query]
                    query = query.replace("[trigger]", trigger_token)

                    # prepare [CLS] query [SEP] sentence [SEP]
                    tokens = []
                    segment_ids = []
                    token_to_orig_map = {}  # 用于映射bert_input到sentence
                    # add [CLS]
                    tokens.append("[CLS]")
                    segment_ids.append(0)
                    # add query
                    query_tokens = tokenizer.tokenize(query)
                    for token in query_tokens:
                        tokens.append(token)
                        segment_ids.append(0)
                    # add [SEP]
                    tokens.append("[SEP]")
                    segment_ids.append(0)
                    # add sentence
                    tmpcnt = 0
                    for (i, token) in enumerate(example.sentence):
                        # 需要特殊处理转义字符
                        if(token in [' ', '\t', '\n']):
                            token = '[BLANK]'
                        token_to_orig_map[len(tokens)] = i  # 当前位置和原始句子位置的映射
                        if not len(tokenizer.tokenize(token)):
                            sub_tokens = ['[INV]']
                        else:
                            sub_tokens = tokenizer.tokenize(token)
                        assert len(sub_tokens) >0 , f"{example_id}\n{example.sentence}\n{i}\n{token}\n{sub_tokens}"
                        tokens.append(sub_tokens[0])
                        segment_ids.append(1)
                        tmpcnt += 1
                    assert tmpcnt == len(example.sentence),"tokens and sentence must be the same"
                    # add [SEP]
                    tokens.append("[SEP]")
                    segment_ids.append(1)
                    # transform to input_ids ...
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)
                    while len(input_ids) < max_seq_length:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)

                    # start & end position
                    start_position, end_position = None, None

                    sentence_start = example.s_start
                    sentence_offset = len(query_tokens) + 2
                    fea_trigger_offset = trigger_offset + sentence_offset

                    if_trigger_ids = [0] * len(segment_ids)     # 标注是否是trigger
                    trigger_len = event[0][1] - event[0][0] + 1
                    # 掌握全新Python技能
                    if_trigger_ids[fea_trigger_offset:fea_trigger_offset+trigger_len] = [1] * trigger_len

                    if is_training:
                        no_answer = True
                        labels = [[0] * 2 for i in range(max_seq_length)]
                        start_position = 0
                        end_position = 0

                        for argument in arguments:
                            gold_argument_type = argument[2]
                            if gold_argument_type == argument_type:
                                no_answer = False
                                answer_start, answer_end = argument[0], argument[1]

                                start_position = answer_start - sentence_start + sentence_offset
                                end_position = answer_end - sentence_start + sentence_offset

                                # 构建label
                                labels[start_position][0] = 1   # 设置start为True
                                labels[end_position][1] = 1 # 设置end为True
                        
                        if no_answer == False:
                            features.append(InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, if_trigger_ids=if_trigger_ids,
                                                          event_type=event_type, argument_type=argument_type, fea_trigger_offset=fea_trigger_offset, sentence_offset = sentence_offset,
                                                          start_position=start_position, end_position=end_position, labels=labels))    # TODO 每个argument role都要对应一个feature
                        if no_answer:
                            start_position, end_position = 0, 0

                            # 构建label
                            labels = [[0] * 2 for i in range(max_seq_length)]   # 这里没有考虑[CLS]

                            features.append(InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, if_trigger_ids=if_trigger_ids,
                                                          event_type=event_type, argument_type=argument_type, fea_trigger_offset=fea_trigger_offset, sentence_offset = sentence_offset,
                                                          start_position=start_position, end_position=end_position, labels=labels))
                    else:
                        labels=None
                        features.append(InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, if_trigger_ids=if_trigger_ids,
                                                      event_type=event_type, argument_type=argument_type, fea_trigger_offset=fea_trigger_offset, sentence_offset = sentence_offset,
                                                      start_position=start_position, end_position=end_position, labels=labels))
        logger.info("Saving features into cached file %s",cached_features_file)
        if cached_features_file.parent.exists() == False:
            cached_features_file.parent.mkdir(parents=True)
        torch.save(features, cached_features_file)
        return features


def read_query_templates(normal_file, des_file):
    """
    Load query templates
    normal_file是最朴素的query文件，des_file是添加了guideline annotation描述信息的文件。
    """
    query_templates = dict()    # query_templates是字典的字典，里面字典的value还是list() {'Business.Declare-Bankruptcy': {'Org': ['Org', 'Org in [trigger]', 'What is the Org?', 'What is the Org in [trigger]?', 'What declare bankruptcy?', 'What declare bankruptcy in [trigger]?']}}
    with open(normal_file, "r", encoding='gbk') as f:
        for line in f:
            event_arg, query = line.strip().split(",")
            event_type, arg_name = event_arg.split("_")

            if event_type not in query_templates:
                query_templates[event_type] = dict()
            if arg_name not in query_templates[event_type]:
                query_templates[event_type][arg_name] = list()

            # 0 template arg_name
            query_templates[event_type][arg_name].append(arg_name)
            # 1 template arg_name + in trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(arg_name + "在[trigger]中")
            # 2 template arg_query
            query_templates[event_type][arg_name].append(query)
            # 3 arg_query + trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(query[:-1] + "在[trigger]中")

    with open(des_file, "r", encoding='gbk') as f:
        for line in f:
            event_arg, query = line.strip().split(",")
            event_type, arg_name = event_arg.split("_")
            # 4 template des_query
            query_templates[event_type][arg_name].append(query)
            # 5 template des_query + trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(query[:-1] + "在[trigger]中")

    for event_type in query_templates:
        for arg_name in query_templates[event_type]:
            assert len(query_templates[event_type][arg_name]) == 6

    return query_templates

RawResult = collections.namedtuple("RawResult",
                                   ["example_id", "event_type_offset_argument_type", "start_logits", "end_logits"])

def pointer_arg_decode(start_logits, end_logits, start_threshold=0.5, end_threshold=0.5, one_arg=False):
    candidate_entities = []

    start_ids = np.argwhere(start_logits > start_threshold)[:,0]
    end_ids = np.argwhere(end_logits > end_threshold)[:,0]

    # TODO 对于每一个_start 都尝试给它找一个最短的，长度在命令行参数max_answer_length以内？可能不需要这个参数，先不用
    for _start in start_ids:
        for _end in end_ids:
            if _end >= _start:
                # (start, end, logits)
                candidate_entities.append([_start, _end, start_logits[_start] + end_logits[_end] ])
                break   # break用于求最近匹配的

    entities = []   # 最终的结果集
    # 找整个候选集，如果存在包含的实体对选 logits 最高的作为候选
    for x in candidate_entities:
        flag = True
        for y in candidate_entities:
            if x == y:
                continue

            # 比较两个区间的范围是否重合，如果重合，选择概率更高的那个span作为答案
            if (x[0]>=y[0] and x[1]<=y[1]) or (x[0]<=y[0] and x[1]>=y[1]):
                if y[2] > x[2]:
                    flag = False
                    break
        if flag:
            entities.append(x[:2])  # 不保留概率

    # 不要求强制解码，因为确实有些论元不会有
    return entities

def make_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length, larger_than_cls):
    """
    根据all_results的内容，来预测结果
    :param all_examples: all_examples的内容为(sentence=sentence, events=events, s_start=s_start)
    :param all_features: 内容为InputFeatures(example_id, tokens, token_to_orig_map, input_ids, input_mask, segment_ids, if_trigger_ids,
                                            event_type, argument_type, fea_trigger_offset, start_position, end_position)
    :param all_results: RawResult(example_id, event_type_offset_argument_type, start_logits, end_logits)
    :param n_best_size: 在函数_get_best_indexes()中使用，表示得到排列前n个的结果
    :param max_answer_length: 最大答案区间长度，从命令行参数而来
    :param larger_than_cls: 是否需要larger than cls
    :return:
    """
    example_id_to_features = collections.defaultdict(list)  #得到数据集中每个example id对应的所有feature（由于一个example里面有多个论元，所以这里使用的是list~）
    for feature in all_features:
        example_id_to_features[feature.example_id].append(feature)

    example_id_to_results = collections.defaultdict(list)   #得到数据集中每个example id对应的所有result（由于一个example有多个论元，因此有多个result。这里的result就是RawResult对象）
    for result in all_results:
        example_id_to_results[result.example_id].append(result)

    final_all_predictions = collections.OrderedDict()   # 会记得插入顺序的dict，这是最终函数返回的结果

    for (example_id, example) in enumerate(all_examples):   # 遍历所有的example，包括不包含event的example
        features = example_id_to_features[example_id]
        results = example_id_to_results[example_id]
        final_all_predictions[example_id] = []
        for (feature_index, feature) in enumerate(features):    #遍历所有的
            event_type_argument_type = "_".join([feature.event_type, feature.argument_type])    # '股份股权转让_sub-org'
            event_type_offset_argument_type = "_".join([feature.event_type, str(feature.token_to_orig_map[feature.fea_trigger_offset]), feature.argument_type]) # '股份股权转让_76_sub-org'

            for result in results:  # 遍历所有的result
                if result.event_type_offset_argument_type == event_type_offset_argument_type:   # 如果两者匹配上了，说明这个result是用来预测这个论元的
                    # 修正一下start_logits和end_logits的范围，只选择有用的范围
                    start_logits = result.start_logits[feature.sentence_offset:feature.sentence_offset+len(feature.token_to_orig_map)]
                    end_logits = result.end_logits[feature.sentence_offset:feature.sentence_offset+len(feature.token_to_orig_map)]

                    pred_args = pointer_arg_decode(start_logits, end_logits, start_threshold=0.5, end_threshold=0.5, one_arg=False)

                    for args in pred_args:
                        final_all_predictions[example_id].append([event_type_argument_type, args])
    return final_all_predictions

def evaluate(args, model, device, eval_dataloader, eval_examples, gold_examples, eval_features, na_prob_thresh=1.0, pred_only=False):
    """

    :param args: 就是传来的命令行参数
    :param model: 算法模型model
    :param device: 训练设备
    :param eval_dataloader: eval_dataset的专属dataloader，eval_dataset的内容为 (all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids, all_example_index)
    :param eval_examples: eval_example的内容为(sentence=sentence, events=events, s_start=s_start)
    :param gold_examples: gold_example的内容为(sentence=sentence, events=events, s_start=s_start)
    :param eval_features: 内容为InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map, input_ids=input_ids,
                                                input_mask=input_mask, segment_ids=segment_ids, if_trigger_ids=if_trigger_ids,
                                                event_type=event_type, argument_type=argument_type, fea_trigger_offset=fea_trigger_offset,
                                                start_position=start_position, end_position=end_position, labels=labels)
    :param na_prob_thresh: na_prob的阈值，由于这个程序控制了threshold，所以这个参数没有用
    :param pred_only: 是否只是预测，test阶段时为true
    :return:
    """
    all_results = []
    model.eval()
    for idx, (input_ids, input_mask, segment_ids, if_trigger_ids, example_indices) in enumerate(eval_dataloader):
        # input_ids, input_mask, segment_ids, if_trigger_ids: shape of [batch_size, max_seq_len]
        if pred_only and idx % 10 == 0:
            logger.info("Running test: %d / %d" % (idx, len(eval_dataloader)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        if_trigger_ids = if_trigger_ids.to(device)
        with torch.no_grad():
            if not args.add_if_trigger_embedding:
                logits = model(input_ids, segment_ids, input_mask)[0].cpu().numpy()
                batch_start_logits = logits[:, :, 0]
                batch_end_logits = logits[:, :, 1]
            else:
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, if_trigger_ids, input_mask).cpu().numpy()
        for i, example_index in enumerate(example_indices):
            # 这里的example_index是重新生成的index，不是原来样本里面的sentence_index
            start_logits = batch_start_logits[i]    # shape of [max_seq_len]
            end_logits = batch_end_logits[i]        # shape of [max_seq_len]
            eval_feature = eval_features[example_index.item()]  # 要获得Tensor的值，一般都要加item()
            example_id = eval_feature.example_id    # 这里的example_id就是原来的sentence在样本中的位置
            event_type_offset_argument_type = "_".join([eval_feature.event_type, str(eval_feature.token_to_orig_map[eval_feature.fea_trigger_offset]), eval_feature.argument_type]) # 如此就可以精确定位到一个argument了
            all_results.append(RawResult(example_id=example_id, event_type_offset_argument_type=event_type_offset_argument_type,
                                         start_logits=start_logits, end_logits=end_logits))

    # preds是一个OrderedDict，key为example_id，value就是模型预测得到的结果的list.
    # {..., 5: [['Movement.Transport_Destination', [15, 15]], ['Movement.Transport_Origin', [15, 15]]], ... }
    preds = make_predictions(eval_examples, eval_features, all_results, args.n_best_size, args.max_answer_length, args.larger_than_cls)    # 没有看懂n_best_size的意思

    # get all_gold, all_gold也是OrderedDict()类型的。 里面每个论元的格式为: [event_type_argument_type, [start_offset, end_offset]]
    all_gold = collections.OrderedDict()
    for (example_id, example) in enumerate(gold_examples):
        all_gold[example_id] = []
        for event in example.events:
            event_type = event[0][2]    # 修改为[2]
            for argument in event[1:]:
                argument_start, argument_end, argument_type = argument[0] - example.s_start, argument[1] - example.s_start, argument[2]
                event_type_argument_type = "_".join([event_type, argument_type])
                all_gold[example_id].append([event_type_argument_type, [argument_start , argument_end]])

    # linearize the preds and all_gold
    new_preds = []  # 用于将所有的论元整合到一个大的list中，不再以example_id进行层次划分 [... , ['Movement.Transport_Origin', [15, 15], example_id] , ...]
    new_all_gold = []   # 同上[... , ['Movement.Transport_Origin', [15, 15], example_id] , ...]
    for (example_id, _) in enumerate(gold_examples):
        pred_arg = preds[example_id]    # 是1个list
        gold_arg = all_gold[example_id] # 是1个list
        for argument in pred_arg:
            argument.append(example_id)
            new_preds.append(argument)
        for argument in gold_arg:
            argument.append(example_id)
            new_all_gold.append(argument)

    # get results (classification)
    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    # pred_arg_n
    pred_arg_n = len(new_preds) # for argument in new_preds: pred_arg_n += 1
    # gold_arg_n
    gold_arg_n = len(new_all_gold) #  for argument in new_all_gold: gold_arg_n += 1
    # pred_in_gold_n
    for argument in new_preds:
        if argument in new_all_gold:
            pred_in_gold_n += 1
    # gold_in_pred_n
    gold_in_pred_n = pred_in_gold_n
    # for argument in new_all_gold:
    #     if argument in new_preds:
    #         gold_in_pred_n += 1

    prec_c, recall_c, f1_c = 0, 0, 0
    if pred_arg_n != 0: prec_c = 100.0 * pred_in_gold_n / pred_arg_n
    else: prec_c = 0
    if gold_arg_n != 0: recall_c = 100.0 * gold_in_pred_n / gold_arg_n
    else: recall_c = 0
    if prec_c or recall_c: f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else: f1_c = 0
    # import ipdb; ipdb.set_trace()

    ################################################################################################################################################
    # get results (identification)
    new_preds_identification = []
    for item in new_preds:
        new_item = [item[0].split("_")[0]] # event_type
        new_item += item[1:] # offset and example_id
        new_preds_identification.append(new_item)

    new_all_gold_identification = []
    for item in new_all_gold:
        new_item = [item[0].split("_")[0]] # event_type
        new_item += item[1:] # offset and example_id
        new_all_gold_identification.append(new_item)

    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    # pred_arg_n
    pred_arg_n = len(new_preds_identification)    # for argument in new_preds_identification: pred_arg_n += 1
    # gold_arg_n
    gold_arg_n = len(new_all_gold_identification)    # for argument in new_all_gold_identification: gold_arg_n += 1
    # pred_in_gold_n
    for argument in new_preds_identification:
        if argument in new_all_gold_identification:
            pred_in_gold_n += 1
    # gold_in_pred_n
    gold_in_pred_n = pred_in_gold_n
    # for argument in new_all_gold_identification:
    #     if argument in new_preds_identification:
    #         gold_in_pred_n += 1

    prec_i, recall_i, f1_i = 0, 0, 0
    if pred_arg_n != 0: prec_i = 100.0 * pred_in_gold_n / pred_arg_n
    else: prec_c = 0
    if gold_arg_n != 0: recall_i = 100.0 * gold_in_pred_n / gold_arg_n
    else: recall_i = 0
    if prec_i or recall_i: f1_i = 2 * prec_i * recall_i / (prec_i + recall_i)
    else: f1_i = 0

    result = collections.OrderedDict([('prec_c',  prec_c), ('recall_c',  recall_c), ('f1_c', f1_c), ('prec_i',  prec_i), ('recall_i',  recall_i), ('f1_i', f1_i)])
    return result, preds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    # read query templates
    query_templates = read_query_templates(normal_file = args.normal_file, des_file = args.des_file)

    if args.do_train or (not args.eval_test):
        eval_examples = read_ace_examples(input_file=args.dev_file,
                                          cached_examples_file=Path.cwd() / "dataset/args_mul_arg" / f"cached_dev_{args.task_type}_examples_{args.arch}", #Path.cwd() / args.dataset / f"cached_dev_{args.task_type}_examples_{args.arch}",
                                          is_training=False)
        gold_examples = read_ace_examples(input_file=args.dev_file,
                                          cached_examples_file=Path.cwd() / "dataset/args_mul_arg" / f"cached_dev_{args.task_type}_examples_{args.arch}", #Path.cwd() / args.dataset / f"cached_dev_{args.task_type}_examples_{args.arch}",
                                          is_training=False)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            query_templates=query_templates,
            nth_query=args.nth_query,
            is_training=False,
            cached_features_file=
            Path.cwd() / "dataset/args_mul_arg" / "cached_dev_{}_features_{}_{}".format(args.task_type,args.max_seq_length, args.arch)# Path.cwd() / args.dataset / "cached_dev_{}_features_{}_{}".format(args.task_type,args.max_seq_length, args.arch)
        )
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_if_trigger_ids = torch.tensor([f.if_trigger_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)   # 这是新的example_index，与feature里面的example_id是不同的概念
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    if args.do_train:
        train_examples = read_ace_examples(input_file=args.train_file,
                                           cached_examples_file=Path.cwd() / args.dataset / f"cached_train_{args.task_type}_examples_{args.arch}",  # 在这里使用args.dataset是为了few-shot着想
                                           is_training=True)
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            query_templates=query_templates,
            nth_query=args.nth_query,
            is_training=True,
            cached_features_file=
            Path.cwd() / args.dataset / "cached_train_{}_features_{}_{}".format(
                args.task_type,args.max_seq_length, args.arch)
        )

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_if_trigger_ids = torch.tensor([f.if_trigger_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features],dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids,
                                   all_start_positions, all_end_positions, all_labels)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs   # 总共的优化次数

        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        best_result = None
        lrs = [args.learning_rate] if args.learning_rate else \
            [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            if not args.add_if_trigger_embedding:
                if args.task_type == "trans":
                    model = ArgExtractor.from_pretrained("output/fin_args_qa_thresh_output/binary/best-model")
                elif args.resume_path != "":
                    model = ArgExtractor.from_pretrained(args.resume_path)  # 加载checkpoint
                else:
                    model = ArgExtractor.from_pretrained(args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            else:
                model = BertForQuestionAnswering_withIfTriggerEmbedding.from_pretrained(args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            if args.fp16:
                model.half()

            # 需要重新写一下如何进行多卡训练
            gpu_ids = args.gpu_ids.split(',')
            device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
            model.to(device)
            # set to device to the first cuda
            if len(gpu_ids) > 1:
                logger.info(f'Use multi gpus in: {gpu_ids}')
                gpu_ids = [int(x) for x in gpu_ids]
                model = torch.nn.DataParallel(model, device_ids=gpu_ids)

            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]  # param_optimizer里面全部都是元组，元组的内容为(parameter_name, parameter_value)
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            state = dict()
            if args.resume_path != "":
                # 载入checkpoints文件
                state = torch.load(os.path.join(args.resume_path, 'checkpoint_info.bin'))
                optimizer = BertAdam()
                optimizer.load_state_dict(state['optimizer_state_dict'])
                tr_loss = state["tr_loss"]
                nb_tr_examples = state["nb_tr_examples"]
                nb_tr_steps = state["nb_tr_examples"]
                global_step = state["global_step"]
            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)
                tr_loss = 0
                nb_tr_examples = 0
                nb_tr_steps = 0
                global_step = 0

            start_time = time.time()

            # 需要调整epoch的起始位置
            if args.resume_path != "":
                start_epoch = state["epoch"]
            else:
                start_epoch = 0

            for epoch in range(start_epoch,int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))

                # 在这个时候保存一下checkpoint！
                logger.info("Saving checkpoint at epoch #{} (lr = {})...".format(epoch, lr))
                model_to_save = model.module if hasattr(model, 'module') else model
                subdir = os.path.join(args.output_dir, "epoch{epoch}".format(epoch=epoch))  # 注意目录的不同
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                output_model_file = os.path.join(subdir, WEIGHTS_NAME)
                output_config_file = os.path.join(subdir, CONFIG_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(subdir)

                # 保存epoch, optimizer_state_dict, loss。其实需要的也只有step=0的~
                state = dict()
                state["epoch"] = epoch
                state["optimizer_state_dict"] = optimizer.state_dict()
                state["tr_loss"] = tr_loss
                state["nb_tr_examples"] = nb_tr_examples
                state["nb_tr_steps"] = nb_tr_steps
                state["global_step"] = global_step
                state_dir = os.path.join(subdir, "checkpoint_info.bin")
                torch.save(state, state_dir)
                ################################

                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(train_batches):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, if_trigger_ids, start_positions, end_positions, labels = batch  # 输入到模型中的数据
                    if not args.add_if_trigger_embedding:
                        loss = model(input_ids, segment_ids, input_mask, labels)[0]
                    else:
                        loss = model(input_ids, segment_ids, if_trigger_ids, input_mask, labels)
                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if (step + 1) % eval_step == 0 or step == 0:
                        save_model = False
                        if args.do_eval:
                            result, preds = evaluate(args, model, device, eval_dataloader, eval_examples, gold_examples, eval_features)
                            # import ipdb; ipdb.set_trace()
                            model.train()   # 重新设置为train的状态，保证Dropout和Normalization的运行正常
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
                            if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                best_result = result
                                save_model = True
                                logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                    epoch, step + 1, len(train_batches), time.time() - start_time, tr_loss / nb_tr_steps))
                                logger.info("!!! Best dev %s (lr=%s, epoch=%d): p_c: %.2f, r_c: %.2f, f1_c: %.2f, p_i: %.2f, r_i: %.2f, f1_i: %.2f" %
                                            (args.eval_metric, str(lr), epoch, result["prec_c"], result["recall_c"], result["f1_c"], result["prec_i"], result["recall_i"], result["f1_i"]))
                                model_to_save = model.module if hasattr(model, 'module') else model
                                subdir = os.path.join(args.output_dir, "best-model")
                                if not os.path.exists(subdir):
                                    os.makedirs(subdir)
                                output_model_file = os.path.join(subdir, WEIGHTS_NAME)
                                output_config_file = os.path.join(subdir, CONFIG_NAME)
                                torch.save(model_to_save.state_dict(), output_model_file)
                                model_to_save.config.to_json_file(output_config_file)
                                tokenizer.save_vocabulary(subdir)
                                if best_result:
                                    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
                                        for key in sorted(best_result.keys()):
                                            writer.write("%s = %s\n" % (key, str(best_result[key])))
                        else:
                            save_model = True
                        if (int(args.num_train_epochs)-epoch<3 and (step+1)/len(train_batches)>0.7) or step == 0:
                            save_model = True
                        else:
                            save_model = False
                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            subdir = os.path.join(args.output_dir, "epoch{epoch}-step{step}".format(epoch=epoch, step=step))
                            if not os.path.exists(subdir):
                                os.makedirs(subdir)
                            output_model_file = os.path.join(subdir, WEIGHTS_NAME)
                            output_config_file = os.path.join(subdir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(subdir)
                            if best_result:
                                with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
                                    for key in sorted(best_result.keys()):
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))

    if args.do_eval:
        if args.eval_test:
            eval_examples = read_ace_examples(input_file=args.test_file,
                                              cached_examples_file=Path.cwd() / args.dataset / f"cached_test_{args.task_type}_examples_{args.arch}",
                                              is_training=False)
            gold_examples = read_ace_examples(input_file=args.gold_file,
                                              cached_examples_file=Path.cwd() / args.dataset / f"cached_gold_{args.task_type}_examples_{args.arch}",
                                              is_training=False)
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                query_templates=query_templates,
                nth_query=args.nth_query,
                is_training=False,
                cached_features_file=
                Path.cwd() / args.dataset / "cached_test_{}_features_{}_{}".format(
                    args.task_type,args.max_seq_length, args.arch))
            logger.info("***** Test *****")
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_if_trigger_ids = torch.tensor([f.if_trigger_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids, all_example_index)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        if not args.add_if_trigger_embedding:
            model = ArgExtractor.from_pretrained(args.model_dir)
        else:
            model = BertForQuestionAnswering_withIfTriggerEmbedding.from_pretrained(args.model_dir)
        if args.fp16:
            model.half()
        model.to(device)

        # 这里的preds是make_predictions()函数output的结果，并不是通过threshold之后的结果
        result, preds = evaluate(args, model, device, eval_dataloader, eval_examples, gold_examples, eval_features, pred_only=True)

        with open(os.path.join(args.model_dir, "test_results.txt"), "w") as writer:
            for key in result:
                writer.write("%s = %s\n" % (key, str(result[key])))
        with open(os.path.join(args.model_dir, "arg_predictions.json"), "w") as writer:
            for key in preds:
                writer.write(json.dumps(preds[key], default=int, ensure_ascii=False) + "\n")    # 中文要加入ensure_ascii=False
        with open(os.path.join(args.model_dir, "arg_predictions_new.json"), "w") as writer:
            for key in preds:
                # 输出需要像原始文件一样
                example = dict()
                example["sentence"] = eval_examples[key].sentence
                example["s_start"] = eval_examples[key].s_start
                example["event"] = []
                example["event"].append([])
                # 首先把触发词接上
                example["event"][0].append(eval_examples[key].events[0][0])
                for arg in preds[key]:
                    arg_role = arg[0].split("_")[1]
                    span = arg[1]
                    span.append(arg_role)
                    example["event"][0].append(span)
                writer.write(json.dumps(example, default=int, ensure_ascii=False) + "\n")    # 中文要加入ensure_ascii=False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_dir", default="args_qa_output_thresh/epoch0-step0", type=str, required=True, help="eval/test model")
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--dev_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--gold_file", default=None, type=str)
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=180, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action='store_true', help='Wehther to run eval on the test set.')
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_metric", default='f1_c', type=str)
    parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. "
                             "This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument("--nth_query", default=0, type=int, help="use n-th template query")
    parser.add_argument("--normal_file", default=None, type=str)
    parser.add_argument("--des_file", default=None, type=str)
    parser.add_argument("--larger_than_cls", action='store_true', help="when indexing s and e")
    parser.add_argument("--add_if_trigger_embedding", action='store_true', help="add the if_trigger_embedding")
    parser.add_argument('--gpu_ids', type=str, default='0',help='gpu ids to use, -1 for cpu, "1, 3" for multi gpu')

    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--task_type", type=str)
    parser.add_argument("--arch",type=str)

    # model checkpoints 功能
    parser.add_argument("--resume_path", type=str, default="")

    args = parser.parse_args()

    if max_seq_length != args.max_seq_length: max_seq_length = args.max_seq_length

    main(args)
