# -*- coding:utf-8 -*-
"""
@Project ：eeqa_yc 
@File    ：trigger_data_preprocessing.py
@IDE     ：PyCharm 
@Author  ：yc1999
@Date    ：2021/4/27 15:08
@Desc    ：创建fin_trigger_qa.py文件需要的数据集，文件全都放在 dataset/trigger 下面
"""
import json
import random
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re
import math

# 需要创建的文件的父文件夹
parent_dir = Path.cwd().parent / "dataset/trigger"
parent_dir.mkdir(parents=True,exist_ok=True)

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
    对CCKS数据集的格式进行修改，trigger需要的格式。

    :param filename: 需要转换的文件的文件名
    :return:
    """
    # 原始文件的目录
    raw_file_path = Path.cwd().parent.parent / "dataset" / filename
    converted_file = open(parent_dir / filename, "w", encoding="utf-8")

    with open(raw_file_path, mode="r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            id = line["id"]
            events = line["events"]
            content = line["content"]

            # 创建一个事件类型字典
            event_type_dict = dict()

            for event in events:
                type = event["type"]
                if type not in event_type_dict:
                    event_type_dict[type] = []
                mentions = event["mentions"]
                trigger = []
                for argument in mentions:
                    word, span, role = argument["word"], argument["span"], argument["role"]
                    span[1] = span[1] - 1  # 考虑数据的构造需要减去一位
                    if role == "trigger" :
                        span.append(type)
                        trigger.append(span)
                # 还要注意去重，有trigger重叠情况
                if trigger not in event_type_dict[type]:
                    event_type_dict[type].append(trigger)

            # 开始写入新行
            for type in event_type_dict:
                newLine = {}
                newLine["id"] = id
                newLine["sentence"] = content
                newLine["s_start"] = 0
                newLine["event"] = event_type_dict[type]
                # 写入新的文件
                converted_file.write(json.dumps(newLine, ensure_ascii=False) + "\n")  # ensure_ascii=False 是处理中文时，必须写的

def merge_data(task_type):
    """
    将train_base和test_base的数据融合到一起

    :param task_type: 任务类型
    :return:
    """

    merge_file = open(parent_dir / f"{task_type}_merge.json", "w")
    with open(parent_dir / f"train_{task_type}.json","r") as f:
        for line in tqdm(f):
            line = json.loads(line)
            merge_file.write(json.dumps(line, ensure_ascii=False) + "\n")
    with open(parent_dir / f"test_{task_type}.json","r") as f:
        for line in tqdm(f):
            line = json.loads(line)
            merge_file.write(json.dumps(line, ensure_ascii=False) + "\n")

def train_dev_split(filename, dev_prop=0.2,
                    shuffle=True, save=True, seed=1,
                    data_dir=parent_dir, task_type="base"):
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
    train_json_path = open(data_dir / f"train.{task_type}.json", "w")
    dev_json_path = open(data_dir / f"dev.{task_type}.json", "w")
    for line in dev:
        dev_json_path.write(json.dumps(line,ensure_ascii=False)+"\n")
    for line in train:
        train_json_path.write(json.dumps(line,ensure_ascii=False)+"\n")

    return train, dev

def get_few_shot_data():
    """

    :return:
    """

    # 得到每个事件对应的论元
    class_list = ["收购","担保","中标","签署合同","判决"]   # 要求必须是完整的
    k_list = [1,3,5,7,9]

    merge_dataset = []
    merge_file = parent_dir / "trans_merge.json"
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
            if event["event"][0][0][2] == cls:
                candidate_support_set[cls].append(event)
                merge_dataset.remove(event)
                count += 1
                if count == 45:
                    break

    # 接下来开始写入文件
    data_dir = parent_dir
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

def get_mean_res():
    output_dir = Path.cwd().parent / "output/fin_args_qa_thresh_output/trigger/few-shot"
    print(output_dir)
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

if __name__ == "__main__":
    # Step 1: 转换数据格式
    # change_format("test_base.json")

    # Step 2: 合并数据
    # merge_data(task_type="trans")

    # Step 3: 数据及划分
    # filename = parent_dir / "base_merge.json"
    # train_dev_split(filename,task_type="base")

    # Step 4：生成few-shot数据集
    # get_few_shot_data()

    # Step 5: 得到结果均值
    get_mean_res()

    # pass