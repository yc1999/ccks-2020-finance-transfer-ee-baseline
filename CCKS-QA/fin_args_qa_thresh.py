# -*- coding:utf-8 -*-
"""
@Project ：eeqa_yc 
@File    ：fin_args_qa_thresh.py
@IDE     ：PyCharm 
@Author  ：yc1999
@Date    ：2021/3/31 10:51 
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
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertForQuestionAnswering_withIfTriggerEmbedding
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from transformers import BertTokenizer

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
                 event_type, argument_type, fea_trigger_offset,
                 #
                 start_position=None, end_position=None):

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

        self.start_position = start_position
        self.end_position = end_position


def read_ace_examples(input_file, is_training):
    """Read a ACE json file into a list of AceExample."""
    # TODO: 这里的example是包含所有数据的，我们的小样本需要筛选出包含argument的样本，不能包含不包含argument的样本。
    # 这里面一个AceExample包含多个event，但是这并不影响我在金融数据集上的实验
    examples = []
    with open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            sentence, events, s_start = example["sentence"], example["event"], example["s_start"]
            example = AceExample(sentence=sentence, events=events, s_start=s_start)
            examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, query_templates, nth_query, is_training):
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
    for (example_id, example) in enumerate(examples):
        # 需要遍历所有的事件，每一个事件里面的每一个argument作为一个训练样本
        for event in example.events:
            trigger_offset = event[0][0] - example.s_start
            event_type = event[0][1]
            trigger_token = example.sentence[trigger_offset]    # 找个trigger是哪个单词
            arguments = event[1:]   # 得到所有的argument，[[73, 73, 'Vehicle'], [78, 78, 'Artifact'], [82, 82, 'Destination']]
            for argument_type in query_templates[event_type]:   # 遍历这个事件类型拥有的所有argument

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
                for (i, token) in enumerate(example.sentence):
                    # TODO: 需要特殊处理转义字符
                    # 需要特殊处理转义字符
                    if(token == '\xa0'):
                        token = '[BLANK]'
                    token_to_orig_map[len(tokens)] = i  # 当前位置和原始句子位置的映射
                    sub_tokens = tokenizer.tokenize(token)
                    tokens.append(sub_tokens[0])
                    segment_ids.append(1)
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
                if_trigger_ids[fea_trigger_offset] = 1

                # TODO: 理解训练模式是啥意思：用于标注这个训练样例是否包含该argument_type
                if is_training:
                    no_answer = True
                    for argument in arguments:
                        gold_argument_type = argument[2]
                        if gold_argument_type == argument_type:
                            no_answer = False
                            answer_start, answer_end = argument[0], argument[1]

                            start_position = answer_start - sentence_start + sentence_offset
                            end_position = answer_end - sentence_start + sentence_offset
                            features.append(InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, if_trigger_ids=if_trigger_ids,
                                                          event_type=event_type, argument_type=argument_type, fea_trigger_offset=fea_trigger_offset,
                                                          start_position=start_position, end_position=end_position))    # 每个argument都要对应一个feature
                    if no_answer:
                        # TODO: 没有这个argument，那么就标注[CLS]为开始和结束的ground_truth，这样就能学到如何判别不包含这个argument了。
                        start_position, end_position = 0, 0
                        features.append(InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, if_trigger_ids=if_trigger_ids,
                                                      event_type=event_type, argument_type=argument_type, fea_trigger_offset=fea_trigger_offset,
                                                      start_position=start_position, end_position=end_position))
                else:
                    features.append(InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, if_trigger_ids=if_trigger_ids,
                                                  event_type=event_type, argument_type=argument_type, fea_trigger_offset=fea_trigger_offset,
                                                  start_position=start_position, end_position=end_position))
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
            query_templates[event_type][arg_name].append(arg_name + " in [trigger]")
            # 2 template arg_query
            query_templates[event_type][arg_name].append(query)
            # 3 arg_query + trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")

    with open(des_file, "r", encoding='gbk') as f:
        for line in f:
            event_arg, query = line.strip().split(",")
            event_type, arg_name = event_arg.split("_")
            # 4 template des_query
            query_templates[event_type][arg_name].append(query)
            # 5 template des_query + trigger (replace [trigger] when forming the instance)
            query_templates[event_type][arg_name].append(query[:-1] + " in [trigger]?")

    for event_type in query_templates:
        for arg_name in query_templates[event_type]:
            assert len(query_templates[event_type][arg_name]) == 6

    return query_templates

RawResult = collections.namedtuple("RawResult",
                                   ["example_id", "event_type_offset_argument_type", "start_logits", "end_logits"])

def make_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length, larger_than_cls):
    """
    根据all_results的内容，来预测结果
    :param all_examples: all_examples的内容为(sentence=sentence, events=events, s_start=s_start)
    :param all_features: 内容为InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map, input_ids=input_ids,
                                             input_mask=input_mask, segment_ids=segment_ids, if_trigger_ids=if_trigger_ids,
                                             event_type=event_type, argument_type=argument_type, fea_trigger_offset=fea_trigger_offset,
                                             start_position=start_position, end_position=end_position)
    :param all_results: RawResult(example_id=example_id, event_type_offset_argument_type=event_type_offset_argument_type,
                                  start_logits=start_logits, end_logits=end_logits)
    :param n_best_size: 在函数_get_best_indexes()中使用，表示得到排列前n个的结果
    :param max_answer_length: 最大答案区间长度，从命令行参数而来
    :param larger_than_cls: 是否需要larger than cls
    :return:
    """
    example_id_to_features = collections.defaultdict(list)  #得到数据集中每个example的feature，由于一个example里面有多个论元，所以这里使用的是list~
    for feature in all_features:
        example_id_to_features[feature.example_id].append(feature)
    example_id_to_results = collections.defaultdict(list)   #得到数据集中每个example的result，由于一个example有多个论元，因此有多个result。这里的result就是RawResult对象
    for result in all_results:
        example_id_to_results[result.example_id].append(result)
    _PrelimPrediction = collections.namedtuple("PrelimPrediction",
                                               ["start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict() # 会记得插入顺序的dict
    final_all_predictions = collections.OrderedDict()   # 会记得插入顺序的dict，这是最终函数返回的结果
    # all_nbest_json = collections.OrderedDict()
    # scores_diff_json = collections.OrderedDict()

    for (example_id, example) in enumerate(all_examples):   # 遍历所有的example，包括不包含event的example
        features = example_id_to_features[example_id]
        results = example_id_to_results[example_id]
        all_predictions[example_id] = collections.OrderedDict() # OrderedDict的OrderedDict，貌似是没有用的
        final_all_predictions[example_id] = []
        for (feature_index, feature) in enumerate(features):    #遍历所有的
            event_type_argument_type = "_".join([feature.event_type, feature.argument_type])
            event_type_offset_argument_type = "_".join([feature.event_type, str(feature.token_to_orig_map[feature.fea_trigger_offset]), feature.argument_type])

            start_indexes, end_indexes = None, None
            prelim_predictions = []
            for result in results:  # 遍历所有的result
                if result.event_type_offset_argument_type == event_type_offset_argument_type:   # 如果两者匹配上了，说明这个result是用来预测这个论元的
                    start_indexes = _get_best_indexes(result.start_logits, n_best_size, larger_than_cls, result.start_logits[0])    # 开始和结束的位置
                    end_indexes = _get_best_indexes(result.end_logits, n_best_size, larger_than_cls, result.end_logits[0])  #开始和结束的位置
                    # add span preds
                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            if start_index >= len(feature.tokens) or end_index >= len(feature.tokens):  # 因为存在max_seq_len，有些地方在padding，所以需要进行判断
                                continue
                            if start_index not in feature.token_to_orig_map or end_index not in feature.token_to_orig_map:  # 只筛选出那些在原句子中的index
                                continue
                            if end_index < start_index:
                                continue
                            length = end_index - start_index + 1
                            if length > max_answer_length:  # 答案长度不能超过最大长度
                                continue
                            prelim_predictions.append(
                                _PrelimPrediction(start_index=start_index, end_index=end_index,
                                                  start_logit=result.start_logits[start_index], end_logit=result.end_logits[end_index]))

                    ## add null pred
                    if not larger_than_cls:
                        feature_null_score = result.start_logits[0] + result.end_logits[0]
                        prelim_predictions.append(
                            _PrelimPrediction(start_index=0, end_index=0,
                                              start_logit=result.start_logits[0], end_logit=result.end_logits[0]))

                    ## sort
                    prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)  # 对prelim_predictions按照start和end的和进行排序
                    # print(len(prelim_predictions))
                    # if len(prelim_predictions) > 0:
                    #     print(prelim_predictions[0].start_logit + prelim_predictions[0].end_logit - result.start_logits[0] - result.end_logits[0])
                    #     print(prelim_predictions[0].start_index, prelim_predictions[0].end_index)
                    # all_predictions[example_id][event_type_offset_argument_type] = prelim_predictions

                    ## get final pred in format: [event_type_offset_argument_type, [start_offset, end_offset]]
                    max_num_pred_per_arg = 4    # 每个论元最多拥有的预测数量，原来还可以设置这个，那岂不是precision值和recall会降低啊！因为你的预测结果都可以预测多个了，还好后面有threshold压着
                    for idx, pred in enumerate(prelim_predictions):
                        if (idx + 1) > max_num_pred_per_arg: break
                        if pred.start_index == 0 and pred.end_index == 0: break
                        orig_sent_start, orig_sent_end = feature.token_to_orig_map[pred.start_index], feature.token_to_orig_map[pred.end_index]
                        na_prob = (result.start_logits[0] + result.end_logits[0]) - (pred.start_logit + pred.end_logit)
                        final_all_predictions[example_id].append([event_type_argument_type, [orig_sent_start, orig_sent_end], na_prob])
                        # final_all_predictions[example_id].append([event_type_argument_type, [orig_sent_start, orig_sent_end]])

    return final_all_predictions


def _get_best_indexes(logits, n_best_size=1, larger_than_cls=False, cls_logit=None):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)   #妙啊，既能够得到原来的index，也能得到排序结果

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        if larger_than_cls:
            if index_and_score[i][1] < cls_logit:
                break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def find_best_thresh(new_preds, new_all_gold):
    """
    找到最好的threshold值
    :param new_preds: 数据格式变换了一下的preds，[... , ['Movement.Transport_Origin', [15, 15], -0.6534180045127869, example_id] , ...]
    :param new_all_gold: [... , ['Movement.Transport_Origin', [15, 15], example_id] , ...]
    :return:
    """
    best_score = 0
    best_na_thresh = 0
    gold_arg_n, pred_arg_n = len(new_all_gold), 0

    candidate_preds = []
    for i, argument in enumerate(new_preds):
        print(f"Progoress: {i} / {len(new_preds)}")
        candidate_preds.append(argument[:-2] + argument[-1:])
        pred_arg_n += 1

        pred_in_gold_n, gold_in_pred_n = 0, 0
        # pred_in_gold_n
        for argu in candidate_preds:
            if argu in new_all_gold:
                pred_in_gold_n += 1
        # gold_in_pred_n
        for argu in new_all_gold:
            if argu in candidate_preds:
                gold_in_pred_n += 1

        prec_c, recall_c, f1_c = 0, 0, 0
        if pred_arg_n != 0: prec_c = 100.0 * pred_in_gold_n / pred_arg_n
        else: prec_c = 0
        if gold_arg_n != 0: recall_c = 100.0 * gold_in_pred_n / gold_arg_n
        else: recall_c = 0
        if prec_c or recall_c: f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
        else: f1_c = 0

        if f1_c > best_score:   #每一步都要重新计算一下f1_c，以此来求得最佳的score
            best_score = f1_c
            best_na_thresh = argument[-2]
            logger.info(f"best_na_thresh: {best_na_thresh}")

    # import ipdb; ipdb.set_trace()
    return best_na_thresh + 1e-10

def evaluate(args, model, device, eval_dataloader, eval_examples, gold_examples, eval_features, na_prob_thresh=1.0, pred_only=False):
    """

    :param args: 就是传来的命令行参数
    :param model: 算法模型model
    :param device: 设备
    :param eval_dataloader: eval_dataset的专属dataloader，eval_dataset的内容为 (all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids, all_example_index)
    :param eval_examples: eval_example的内容为(sentence=sentence, events=events, s_start=s_start)
    :param gold_examples: gold_example的内容为(sentence=sentence, events=events, s_start=s_start)
    :param eval_features: 内容为InputFeatures(example_id=example_id, tokens=tokens, token_to_orig_map=token_to_orig_map, input_ids=input_ids,
                                                input_mask=input_mask, segment_ids=segment_ids, if_trigger_ids=if_trigger_ids,
                                                event_type=event_type, argument_type=argument_type, fea_trigger_offset=fea_trigger_offset,
                                                start_position=start_position, end_position=end_position)
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
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
            else:
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, if_trigger_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            # 这里的example_index是重新生成的index，不是原来样本里面的sentence_index
            start_logits = batch_start_logits[i].detach().cpu().tolist()    # shape of [max_seq_len]
            end_logits = batch_end_logits[i].detach().cpu().tolist()        # shape of [max_seq_len]
            eval_feature = eval_features[example_index.item()]  # 要获得Tensor的值，一般都要加item()
            example_id = eval_feature.example_id    # 这里的example_id就是原来的sentence在样本中的位置
            event_type_offset_argument_type = "_".join([eval_feature.event_type, str(eval_feature.token_to_orig_map[eval_feature.fea_trigger_offset]), eval_feature.argument_type])
            all_results.append(RawResult(example_id=example_id, event_type_offset_argument_type=event_type_offset_argument_type,
                                         start_logits=start_logits, end_logits=end_logits))

    # preds, nbest_preds, na_probs = \
    preds = make_predictions(eval_examples, eval_features, all_results,
                             args.n_best_size, args.max_answer_length, args.larger_than_cls)    # 没有看懂n_best_size的意思
    # preds是一个OrderedDict，key为example_id，value就是模型预测得到的结果的list，{..., 5: [['Movement.Transport_Destination', [15, 15], -0.6047092229127884], ['Movement.Transport_Origin', [15, 15], -0.6534180045127869]], ... }
    preds_init = copy.deepcopy(preds)   # 这里采用的是深克隆，不是浅克隆。

    # get all_gold in format: [event_type_argument_type, [start_offset, end_offset]]
    # 从数据集中得到所有的ground-truth label
    all_gold = collections.OrderedDict()
    for (example_id, example) in enumerate(gold_examples):
        all_gold[example_id] = []
        for event in example.events:
            # if not event: continue 这句语句是多次一举啊~
            trigger_offset = event[0][0] - example.s_start
            event_type = event[0][1]
            for argument in event[1:]:
                argument_start, argument_end, argument_type = argument[0] - example.s_start, argument[1] - example.s_start, argument[2]
                # event_type_offset_argument_type = "_".join([event_type, str(trigger_offset), argument_type])  为什么不要求offset相同了呢？很奇怪啊，按道理来说，应该是要相同的。
                event_type_argument_type = "_".join([event_type, argument_type])
                all_gold[example_id].append([event_type_argument_type, [argument_start , argument_end]])

    # linearize the preds and all_gold
    new_preds = []  # 用于将所有的论元整合到一个大的list中，不再以example_id进行层次划分 [... , ['Movement.Transport_Origin', [15, 15], -0.6534180045127869, example_id] , ...]
    new_all_gold = []   # [... , ['Movement.Transport_Origin', [15, 15], example_id] , ...]
    for (example_id, _) in enumerate(gold_examples):
        pred_arg = preds[example_id]    # 是1个list
        gold_arg = all_gold[example_id] # 是1个list
        for argument in pred_arg:
            argument.append(example_id)
            new_preds.append(argument)
        for argument in gold_arg:
            argument.append(example_id)
            new_all_gold.append(argument)

    new_preds = sorted(new_preds, key=lambda x: x[-2])  # 以na_prob作为排序的依据，na_prob越小，说明预测成功的概率越大。
    # best_na_thresh = 0
    best_na_thresh = find_best_thresh(new_preds, new_all_gold)

    final_new_preds = []
    for argument in new_preds:
        if argument[-2] < best_na_thresh:
            final_new_preds.append(argument[:-2] + argument[-1:]) # no na_prob，不使用na_prob里面的数据

    ################################################################################################################################################
    # # logging for DEBUG results
    # if pred_only:
    #     debug_preds = collections.OrderedDict()
    #     for example_id in preds:
    #         debug_preds[example_id] = []
    #         for argument in preds[example_id]:
    #             arg_new = argument[:-2] + argument[-1:]
    #             if arg_new in final_new_preds:
    #                 debug_preds[example_id].append(argument)
    #         debug_preds[example_id] = sorted(debug_preds[example_id], key=lambda x: x[2])

    #     # import ipdb; ipdb.set_trace()
    #     for (example_id, example) in enumerate(gold_examples):
    #         # if example_id > 120: break
    #         if debug_preds[example_id] or all_gold[example_id]:
    #             token_idx = []
    #             for idx, token in enumerate(example.sentence): token_idx.append(" ".join([token, str(idx)]))
    #             logger.info("sent: {}".format(" | ".join(token_idx)))

    #             logger.info("trigger: {}".format(str(example).split("|||")[1]))

    #             gold_str_list = []
    #             for gold in all_gold[example_id]: gold_str_list.append(" ".join([gold[0], str(gold[1][0]), str(gold[1][1])]))
    #             logger.info("gold: {}".format(" | ".join(gold_str_list)))

    #             pred_str_list = []
    #             for pred in debug_preds[example_id]: pred_str_list.append(" ".join([pred[0], str(pred[1][0]), str(pred[1][1]), str(pred[2])]))
    #             logger.info("pred: {} \n".format(" | ".join(pred_str_list)))

    ################################################################################################################################################

    # get results (classification)
    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    # pred_arg_n
    for argument in final_new_preds: pred_arg_n += 1
    # gold_arg_n
    for argument in new_all_gold: gold_arg_n += 1
    # pred_in_gold_n
    for argument in final_new_preds:
        if argument in new_all_gold:
            pred_in_gold_n += 1
    # gold_in_pred_n
    for argument in new_all_gold:
        if argument in final_new_preds:
            gold_in_pred_n += 1

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
    final_new_preds_identification = []
    for item in final_new_preds:
        new_item = [item[0].split("_")[0]] # event_type
        new_item += item[1:] # offset and example_id
        final_new_preds_identification.append(new_item)

    new_all_gold_identification = []
    for item in new_all_gold:
        new_item = [item[0].split("_")[0]] # event_type
        new_item += item[1:] # offset and example_id
        new_all_gold_identification.append(new_item)

    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    # pred_arg_n
    for argument in final_new_preds_identification: pred_arg_n += 1
    # gold_arg_n
    for argument in new_all_gold_identification: gold_arg_n += 1
    # pred_in_gold_n
    for argument in final_new_preds_identification:
        if argument in new_all_gold_identification:
            pred_in_gold_n += 1
    # gold_in_pred_n
    for argument in new_all_gold_identification:
        if argument in final_new_preds_identification:
            gold_in_pred_n += 1

    prec_i, recall_i, f1_i = 0, 0, 0
    if pred_arg_n != 0: prec_i = 100.0 * pred_in_gold_n / pred_arg_n
    else: prec_c = 0
    if gold_arg_n != 0: recall_i = 100.0 * gold_in_pred_n / gold_arg_n
    else: recall_i = 0
    if prec_i or recall_i: f1_i = 2 * prec_i * recall_i / (prec_i + recall_i)
    else: f1_i = 0


    result = collections.OrderedDict([('prec_c',  prec_c), ('recall_c',  recall_c), ('f1_c', f1_c), ('prec_i',  prec_i), ('recall_i',  recall_i), ('f1_i', f1_i), ('best_na_thresh', best_na_thresh)])
    return result, preds_init


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
    # TODO: 将BertTokenizer改成transformers里面的tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    # read query templates
    query_templates = read_query_templates(normal_file = args.normal_file, des_file = args.des_file)

    if args.do_train or (not args.eval_test):
        eval_examples = read_ace_examples(input_file=args.dev_file, is_training=False)
        # TODO: 这里的gold_examples使用的也是dev_file，原作者的书写是错误的，他在GitHub的issues里面有说
        gold_examples = read_ace_examples(input_file=args.dev_file, is_training=False)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            query_templates=query_templates,
            nth_query=args.nth_query,
            is_training=False)
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
        # TODO: convert_examples_to_features函数中is_training的作用非常重要
        train_examples = read_ace_examples(input_file=args.train_file, is_training=True)
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            query_templates=query_templates,
            nth_query=args.nth_query,
            is_training=True)

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
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_if_trigger_ids,
                                   all_start_positions, all_end_positions)
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
                model = BertForQuestionAnswering.from_pretrained(args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            else:
                model = BertForQuestionAnswering_withIfTriggerEmbedding.from_pretrained(args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            if args.fp16:
                model.half()
            # model.to(device)
            # if n_gpu > 1:
            #     model = torch.nn.DataParallel(model)

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

            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=lr,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            global_step = 0
            start_time = time.time()
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(train_batches):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, if_trigger_ids, start_positions, end_positions = batch  # 输入到模型中的数据
                    if not args.add_if_trigger_embedding:
                        loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                    else:
                        loss = model(input_ids, segment_ids, if_trigger_ids, input_mask, start_positions, end_positions)
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
                            # result, _, _ = evaluate(args, model, device, eval_dataset, eval_dataloader, eval_examples, eval_features)
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
                                logger.info("!!! Best dev %s (lr=%s, epoch=%d): p_c: %.2f, r_c: %.2f, f1_c: %.2f, p_i: %.2f, r_i: %.2f, f1_i: %.2f, best_na_thresh: %.5f" %
                                            # logger.info("!!! Best dev %s (lr=%s, epoch=%d): p_c: %.2f, r_c: %.2f, f1_c: %.2f, best_na_thresh: %.10f" %
                                            # (args.eval_metric, str(lr), epoch, result["prec_c"], result["recall_c"], result["f1_c"], result["best_na_thresh"]))
                                            (args.eval_metric, str(lr), epoch, result["prec_c"], result["recall_c"], result["f1_c"], result["prec_i"], result["recall_i"], result["f1_i"], result["best_na_thresh"]))
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
            eval_examples = read_ace_examples(input_file=args.test_file, is_training=False)
            gold_examples = read_ace_examples(input_file=args.gold_file, is_training=False)
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                query_templates=query_templates,
                nth_query=args.nth_query,
                is_training=False)
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
            model = BertForQuestionAnswering.from_pretrained(args.model_dir)
        else:
            model = BertForQuestionAnswering_withIfTriggerEmbedding.from_pretrained(args.model_dir)
        if args.fp16:
            model.half()
        model.to(device)

        # 这里的preds是make_predictions()函数output的结果，并不是通过treshold之后的结果
        result, preds = evaluate(args, model, device, eval_dataloader, eval_examples, gold_examples, eval_features, pred_only=True)

        with open(os.path.join(args.model_dir, "test_results.txt"), "w") as writer:
            for key in result:
                writer.write("%s = %s\n" % (key, str(result[key])))
        with open(os.path.join(args.model_dir, "arg_predictions.json"), "w") as writer:
            for key in preds:
                writer.write(json.dumps(preds[key], default=int) + "\n")

        ### old

        # na_prob_thresh = 1.0
        # if args.version_2_with_negative:
        #     eval_result_file = os.path.join(args.output_dir, "eval_results.txt")
        #     if os.path.isfile(eval_result_file):
        #         with open(eval_result_file) as f:
        #             for line in f.readlines():
        #                 if line.startswith('best_f1_thresh'):
        #                     na_prob_thresh = float(line.strip().split()[-1])
        #                     logger.info("na_prob_thresh = %.6f" % na_prob_thresh)

        # result, preds, _ = \
        #     evaluate(args, model, device, eval_dataset,
        #              eval_dataloader, eval_examples, eval_features,
        #              na_prob_thresh=na_prob_thresh,
        #              pred_only=args.eval_test)
        # with open(os.path.join(args.output_dir, "predictions.json"), "w") as writer:
        #     writer.write(json.dumps(preds, indent=4) + "\n")


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
    args = parser.parse_args()

    if max_seq_length != args.max_seq_length: max_seq_length = args.max_seq_length

    main(args)
