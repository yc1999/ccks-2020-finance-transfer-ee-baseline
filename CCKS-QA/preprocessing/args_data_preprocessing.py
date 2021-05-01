# -*- coding:utf-8 -*-
"""
@Project ：eeqa_yc 
@File    ：args_data_preprocessing.py
@IDE     ：PyCharm 
@Author  ：yc1999
@Date    ：2021/4/18 15:00 
"""

import json
import random
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re
import math

def save_pickle(data, file_path):
    """
    保存成pickle文件

    :param data:
    :param file_path:
    :return:
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def change_format(filename):
    """
    对CCKS数据集的格式进行修改，修改成eeqa中的数据格式

    :param filename: 需要转换的文件的文件名
    :return:
    """
    converted_file = open(Path.cwd().parent / "dataset" / "test_trans.json", "w", encoding="utf-8")

    with open(filename, mode="r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            events = line["events"]
            content = line["content"]
            for event in events:
                type = event["type"]
                mentions = event["mentions"]
                argument_lst = []
                trigger = []
                for argument in mentions:
                    word, span, role = argument["word"], argument["span"], argument["role"]
                    span[1] = span[1] - 1  # 考虑数据的构造需要减去一位
                    if role != "trigger":
                        span.append(role)
                        argument_lst.append(span)
                    else:
                        span.append(type)  # 这里拼接的是事件类型
                        trigger.append(span)
                newLine = {}
                newLine["sentence"] = content
                newLine["s_start"] = 0
                newLine["event"] = []
                newLine["event"].extend(trigger)
                newLine["event"].extend(argument_lst)
                newLine["event"] = [newLine["event"]]
                # 写入新的文件
                converted_file.write(json.dumps(newLine, ensure_ascii=False) + "\n")  # ensure_ascii=False 是处理中文时，必须写的


def train_dev_split(filename, dev_prop=0.2,
                    shuffle=True, save=True, seed=1,
                    data_dir=Path.cwd().parent / "dataset", task_type="base"):
    """
    划分train set和dev set

    :param filename: 文件名
    :param dev_prop: 验证集的比例
    :param shuffle: 是否随机化
    :param save: 是否保存划分的文件
    :param seed: 是否使用随机数种子
    :return:
    """
    data = []
    with open(filename, "r") as f:
        # lines = f.readlines()
        for line in tqdm(f.readlines(), desc="train_val_split"):
            line = json.loads(line)
            data.append(line)
    print(len(data))
    N = len(data)
    dev_size = int(N * dev_prop)
    if shuffle:
        random.seed(seed)
        random.shuffle(data)  # 确实需要洗动一下
    dev = data[:dev_size]
    train = data[dev_size:]

    # 混洗train数据集
    if shuffle:
        random.seed(seed)
        random.shuffle(train)

    if save:
        train_path = data_dir / f"train.{task_type}.pkl"
        dev_path = data_dir / f"dev.{task_type}.pkl"
        save_pickle(data=train, file_path=train_path)
        save_pickle(data=dev, file_path=dev_path)
    return train, dev


def get_test(filename, data_dir=Path.cwd().parent / "dataset", task_type="base"):
    test = []
    with open(filename, "r") as f:
        for line in tqdm(f.readlines(), desc="test"):
            line = json.loads(line)
            test.append(line)
    test_path = data_dir / f"test.{task_type}.pkl"
    save_pickle(data=test, file_path=test_path)


def merge_data(task_type):
    """
    将train_base和test_base的数据融合到一起

    :param task_type: 任务类型
    :return:
    """
    data_dir = Path.cwd().parent / "dataset"

    merge_file = open(data_dir / f"{task_type}_merge.json", "w")
    with open(data_dir / f"train_{task_type}.json","r") as f:
        for line in tqdm(f):
            line = json.loads(line)
            merge_file.write(json.dumps(line, ensure_ascii=False) + "\n")
    with open(data_dir / f"test_{task_type}.json","r") as f:
        for line in tqdm(f):
            line = json.loads(line)
            merge_file.write(json.dumps(line, ensure_ascii=False) + "\n")

def get_few_shot_data():
    """

    :return:
    """
    
    # 得到每个事件对应的论元
    arg_dict = dict()
    with open(Path.cwd().parent / "question_templates" / "chinese-description.csv", "r", encoding='gbk') as f:
        for line in f:
            event_arg, query = line.strip().split(",")
            event_type, arg_name = event_arg.split("_")

            if event_type not in arg_dict:
                arg_dict[event_type] = set()
            if arg_name not in arg_dict[event_type] and arg_name != "trigger":
                arg_dict[event_type].add(arg_name)


    class_list = ["收购","担保","中标","签署合同","判决"]   # 要求必须是完整的
    k_list = [1,2,5,7,9]

    merge_dataset = []
    merge_file = Path.cwd().parent / "dataset" / "trans_merge.json"
    with open(merge_file, "r") as f:
        for line in f:
            line = json.loads(line)
            merge_dataset.append(line)

    # 洗乱数据集
    random.seed(10)
    random.shuffle(merge_dataset)

    # 先在文件中找到九组，然后将它们装起来
    candidate_support_set = dict()
    for cls in class_list:
        candidate_support_set[cls] = []
        count = 0
        for event in merge_dataset:
            # 需要包含所有的argument
            if event["event"][0][0][2] == cls and cls == "收购":
                cur_arg_num = len(event["event"][0])
                if cur_arg_num >= 7:
                    candidate_support_set[cls].append(event)
                    merge_dataset.remove(event)
                    count += 1
                    if count == 45:
                        break
            elif event["event"][0][0][2] == cls:
                cur_arg_num = len(event["event"][0])
                arg_num = len(arg_dict[event["event"][0][0][2]])+1
                if cur_arg_num >= arg_num-1:
                    candidate_support_set[cls].append(event)
                    merge_dataset.remove(event)
                    count += 1
                    if count == 45:
                        break

    # 接下来开始写入文件
    data_dir = Path.cwd().parent / "dataset"
    for k in k_list:
        subdir = data_dir / f"{k}-shot"
        subdir.mkdir(exist_ok=False)
        for group in range(5):
            file_name = subdir / f"group-{group}.json"
            f = open(file_name,"w")
            dataset = []
            for cls in candidate_support_set:
                for event in candidate_support_set[cls][group*k:(group+1)*k]:
                    dataset.append(event)
                    f.write(json.dumps(event,ensure_ascii=False)+"\n")
            save_pickle(dataset,subdir / f"group-{group}.pkl")

    # 保存验证集
    dev_filename = data_dir / "few-shot-dev.json"
    dev = open(dev_filename,"w")
    dataset = []
    for event in merge_dataset:
        dataset.append(event)
        dev.write(json.dumps(event,ensure_ascii=False)+"\n")
    save_pickle(dataset,data_dir / "few-shot-dev.pkl")

def stat_arg_lenth():
    """
    统计所有论元的长度，用来确定fin_arg_qa_thresh.py中的超参数——max_answer_length的大小

    :return:
    """
    merge_file = Path.cwd().parent / "dataset" / "trans_merge.json"
    arg_lenth_dict = dict()
    with open(merge_file, "r") as f:
        for line in f:
            event = json.loads(line)
            for arg in event["event"][0]:
                lenth = arg[1] - arg[0] + 1
                if lenth not in arg_lenth_dict:
                    arg_lenth_dict[lenth] = 1
                else:
                    arg_lenth_dict[lenth] += 1
    sorted_dict = dict(sorted(arg_lenth_dict.items(),key=lambda x:x[0]))
    print(sorted_dict)

def get_mean_res():
    output_dir = Path.cwd().parent / "output/fin_args_qa_thresh_output/few-shot"
    k_lst = [1, 5, 9]
    for k in k_lst:
        groups_dir = output_dir / f"{k}-shot"
        groups = groups_dir.glob("group-[0-9]")
        f1_c = np.array([], dtype=np.float64)
        f1_i = np.array([], dtype=np.float64)
        for group in groups:
            with open(group/"eval_results.txt") as f:
                for line in f:
                    check_c = re.compile(r'(?<=f1_c = )\d+\.?\d*')
                    check_i = re.compile(r'(?<=f1_i = )\d+\.?\d*')
                    if(check_c.findall(line)):
                        f1_c = np.append(f1_c,float(check_c.findall(line)[0]))
                    if(check_i.findall(line)):
                        f1_i = np.append(f1_i,float(check_i.findall(line)[0]))
        # f1_c.astype(np.float)
        print(f1_c, f1_i)
        print(f"f1_c--mean: {f1_c.mean()}, f1_c--dev: {f1_c.std()}\n f1_i--mean: {f1_i.mean()}, f1_i--dev: {f1_i.std()}")

def analyze_trigger():
    data_dir = Path.cwd().parent.parent / "dataset"
    dataset = open(data_dir / "test_base.json","r")
    cnt = 0
    span_len_dict = dict()

    for idx, line in enumerate(dataset):
        example = json.loads(line)
        events = example['events']
        event_type_dict = dict()

        for event in events:
            type = event["type"]
            if type not  in event_type_dict:
                event_type_dict[type] = []

            for arg in event["mentions"]:
                if arg["role"] == "trigger":
                    span = arg["span"]
                    span_len = span[1]-span[0]
                    if span_len not in span_len_dict:
                        span_len_dict[span_len] = 0
                    span_len_dict[span_len] += 1
                    if span not in event_type_dict[type]:
                        event_type_dict[type].append(span)

        for type in event_type_dict.keys():
            if len(event_type_dict[type]) > 1:
                #print(f"example-{idx} contains multiple triggers for event——{type}")
                cnt = cnt+1
    print(cnt)
    print(span_len_dict)

if __name__ == "__main__":
    # Step 1:
    # filename = Path.cwd().parent.parent / "dataset" / "test_trans.json"
    # change_format(filename)

    # Step 2:
    # merge_data(task_type="trans")

    # Step 3:
    # filename = Path.cwd().parent / "dataset" / "base_merge.json"
    # train_dev_split(filename,task_type="base")

    # Step 4:
    # filename = Path.cwd().parent / "dataset" / "test_base.json"
    # get_test(filename)

    # Step 5:
    #get_few_shot_data()

    # Step 6: 查看argument的长度
    #stat_arg_lenth()

    # Step 7: 统计每个文件下的eval_results的结果，然后平均
    get_mean_res()

    # Step 8: 分析数据集中，一个事件的触发词是否都一样，这里需要用到原始数据集！
    # analyze_trigger()
    pass
