import torch
import time
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss
from pybert.train.trainer import Trainer
from torch.utils.data import DataLoader
from pybert.io.utils import collate_fn
from pybert.io.bert_processor import BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.configs.basic_config import config  # 直接引入了一个字典对象
from pybert.model.bert_for_multi_label import BertForMultiLable, BertForMultiLable_Fewshot
from pybert.preprocessing.preprocessor import ChinesePreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport, F1Score
from pybert.callback.optimizater.adamw import AdamW
from pybert.callback.lr_schedulers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
import os

warnings.filterwarnings("ignore")


def run_train(args):
    # --------- data
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels(args.task_type)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(config['data_dir'] / f"{args.data_name}.train.{args.task_type}.pkl")
    train_examples = processor.create_examples(lines=train_data,
                                               example_type=f'train_{args.task_type}',
                                               cached_examples_file=config[
                                                    'data_dir'] / f"cached_train_{args.task_type}_examples_{args.arch}")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                    'data_dir'] / "cached_train_{}_features_{}_{}".format(
                                                   args.task_type, args.train_max_seq_len, args.arch
                                               ))
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    valid_data = processor.get_dev(config['data_dir'] / f"{args.data_name}.valid.{args.task_type}.pkl")
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type=f'valid_{args.task_type}',
                                               cached_examples_file=config[
                                                'data_dir'] / f"cached_valid_{args.task_type}_examples_{args.arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_features_file=config[
                                                'data_dir'] / "cached_valid_{}_features_{}_{}".format(
                                                   args.task_type,args.eval_max_seq_len, args.arch
                                               ))
    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size,
                                  collate_fn=collate_fn)

    # ------- model
    logger.info("initializing model")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = BertForMultiLable.from_pretrained(args.resume_path, num_labels=len(label_list))
    else:
        if args.task_type == 'trans':
            model = BertForMultiLable_Fewshot.from_pretrained(Path('pybert/output/checkpoints/bert/base'), num_labels=len(label_list))
            #model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list))
        else:
            model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list))
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    # 下面是optimizer和scheduler的设计
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ---- callbacks
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch) # TODO: 理解train_monitor的作用，感觉就是一个用来绘图的东西，用于记录每一个epoch中得到的结果
    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'],mode=args.mode,
                                       monitor=args.monitor,arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(args= args,model=model,logger=logger,criterion=BCEWithLogLoss(),optimizer=optimizer,
                      scheduler=scheduler,early_stopping=None,training_monitor=train_monitor,
                      model_checkpoint=model_checkpoint,
                      batch_metrics=[AccuracyThresh(thresh=0.5)],   # 作用于batch之上的metrics，在每次loss.backward()之后都会执行计算，记得区分它与loss
                      epoch_metrics=[AUC(average='micro', task_type='binary'),  # 作用于epoch之上的metrics
                                     MultiLabelReport(id2label=id2label),
                                     F1Score(task_type='binary', average='micro', search_thresh=True)]) # TODO: 考虑是否应该使用F1-score替代指标
    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader)

def run_test(args):
    # TODO: 对训练集使用micro F1-score进行结果评测
    from pybert.io.task_data import TaskData
    from pybert.test.predictor import Predictor
    data = TaskData()
    ids,targets, sentences = data.read_data(raw_data_path=config['test_path'],
                                        preprocessor=ChinesePreProcessor(),
                                        is_train=True)  # 设置为True
    lines = list(zip(sentences, targets))
    #print(ids,sentences)
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)
    label_list = processor.get_labels(args.task_type)
    id2label = {i: label for i, label in enumerate(label_list)}

    test_data = processor.get_test(lines=lines)
    test_examples = processor.create_examples(lines=test_data,
                                              example_type=f'test_{args.task_type}',
                                              cached_examples_file=config[
                                            'data_dir'] / f"cached_test_{args.task_type}_examples_{args.arch}")
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=args.eval_max_seq_len,
                                              cached_features_file=config[
                                            'data_dir'] / "cached_test_{}_features_{}_{}".format(
                                                  args.task_type,args.eval_max_seq_len, args.arch
                                              ))
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size,
                                 collate_fn=collate_fn)
    model = None
    if args.task_type == 'base':
        model = BertForMultiLable.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))
    else:
        # model = BertForMultiLable.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))
        model = BertForMultiLable_Fewshot.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))

    # ----------- predicting
    logger.info('model predicting....')
    predictor = Predictor(model=model,
                          logger=logger,
                          n_gpu=args.n_gpu)
    result = predictor.predict(data=test_dataloader)    # 感觉这个变量名叫all_logits可能更好
    # TODO: 计算F1-score，这个功能模块需要用代码测试一下~
    f1_metric = F1Score(task_type='binary', average='micro', search_thresh=True)
    all_logits = torch.tensor(result,dtype=torch.float)  # 转换成tensor
    all_labels = torch.tensor(targets,dtype=torch.long)  # 转换成tensor
    f1_metric(all_logits, all_labels)  # 会自动打印结果
    print(f1_metric.value())
    # 将结果写入一个文件之中
    with open('test_output/test.log','a+') as f:
        f.write(str(f1_metric.value())+"\n")
    thresh = f1_metric.thresh

    ids = np.array(ids)
    df1 = pd.DataFrame(ids,index=None)
    df2 = pd.DataFrame(result,index=None)
    all_df = pd.concat([df1, df2],axis=1)
    if args.task_type == 'base':
        all_df.columns = ['id','zy','gfgqzr','qs','tz','ggjc']
    else:
        all_df.columns = ['id','sg','pj','zb','qsht','db']
    for column in all_df.columns[1:]:
        all_df[column] = all_df[column].apply(lambda x: 1 if x>thresh else 0)
    # all_df['zy'] = all_df['zy'].apply(lambda x: 1 if x>thresh else 0)
    # all_df['gfgqzr'] = all_df['gfgqzr'].apply(lambda x: 1 if x>thresh else 0)
    # all_df['qs'] = all_df['qs'].apply(lambda x: 1 if x>thresh else 0)
    # all_df['tz'] = all_df['tz'].apply(lambda x: 1 if x>thresh else 0)
    # all_df['ggjc'] = all_df['ggjc'].apply(lambda x: 1 if x>thresh else 0)
    all_df.to_csv(f"test_output/{args.task_type}/cls_out.csv",index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str) # 使用的预训练语言模型
    parser.add_argument("--do_data", action='store_true')   # 进行数据切分
    parser.add_argument("--do_train", action='store_true')  # 进行模型训练
    parser.add_argument("--do_test", action='store_true')   # 进行模型推断
    parser.add_argument("--save_best", action='store_true') # 保留最好的模型
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='ccks', type=str)    # 数据集的名字
    parser.add_argument("--mode", default='min', type=str)  # 设置monitor关注的角度
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--task_type", default='base', type=str)

    parser.add_argument("--epochs", default=4, type=int)
    parser.add_argument("--resume_path", default='', type=str)  # 恢复路径，从pretrained model中载入模型
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--valid_size", default=0.2, type=float)    # 验证集的大小
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ') # 表示是否按照序列的长度排序
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)   # gradient_accumulation_steps的大小，用于解决内存小，无法使用大batch_size的问题
    parser.add_argument("--train_batch_size", default=8, type=int)  # 训练集batch_size
    parser.add_argument('--eval_batch_size', default=8, type=int)   # 测试集batch_size
    parser.add_argument("--train_max_seq_len", default=256, type=int)   # 训练集sequence的最大长度
    parser.add_argument("--eval_max_seq_len", default=256, type=int)    # 测试集sequence的最大长度
    parser.add_argument('--loss_scale', type=float, default=0)  # TODO: 理解loss scale的作用
    parser.add_argument("--warmup_proportion", default=0.1, type=float) # 用于learning rate上的warmup proportion
    parser.add_argument("--weight_decay", default=0.01, type=float) # TODO: 理解weight decay的含义
    parser.add_argument("--adam_epsilon", default=1e-8, type=float) # adam优化器的参数
    parser.add_argument("--grad_clip", default=1.0, type=float) # TODO: 理解grad clip的含义
    parser.add_argument("--learning_rate", default=2e-5, type=float)    # 学习率
    parser.add_argument('--seed', type=int, default=42) # 随机数种子
    parser.add_argument('--fp16', action='store_true')  # TODO: 理解fp16是什么
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    args = parser.parse_args()
    # 初始化日志记录器logger
    config['log_dir'].mkdir(exist_ok=True)   # 源代码没有写这句代码
    init_logger(log_file=config['log_dir'] / f'{args.arch}-{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}.log')
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch / args.task_type # 重新调整输出的位置
    config['checkpoint_dir'].mkdir(exist_ok=True)
    BASE_DIR = Path('pybert')
    config['raw_data_path'] = BASE_DIR / f'dataset/train_{args.task_type}_sample.csv'
    config['test_path'] = BASE_DIR / f'dataset/test_{args.task_type}.csv'
    config['figure_dir'] = config['figure_dir'] / f'{args.task_type}'
    config['figure_dir'].mkdir(exist_ok=True)
    # 动态修改文件路径
    # BASE_DIR = Path('pybert')
    # if args.task_type == 'trans':
    #     config['raw_data_path'] = BASE_DIR / 'dataset/train_trans_sample.csv'
    #     config['test_path'] = BASE_DIR / 'dataset/test_trans.csv'
    #     config['figure_dir'] = config['figure_dir'] / f'{args.task_type}'
    #     config['figure_dir'].mkdir(exist_ok=True)
    # elif args.task_type == 'base':
    #     config['raw_data_path'] = BASE_DIR / 'dataset/train_base_sample.csv'
    #     config['test_path'] = BASE_DIR / 'dataset/test_base.csv'
    #     config['figure_dir'] = config['figure_dir'] / f'{args.task_type}'
    #     config['figure_dir'].mkdir(exist_ok=True)
    # else:
    #     raise ValueError(f"Invalid task_type {args.task_type}")

    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)  # 一个方法设置所有的seed
    logger.info("Training/evaluation parameters %s", args)
    if args.do_data:
        from pybert.io.task_data import TaskData
        data = TaskData()
        ids, targets, sentences = data.read_data(raw_data_path=config['raw_data_path'],preprocessor=ChinesePreProcessor(),is_train=True)
        data.train_val_split(X=sentences, y=targets, shuffle=True, stratify=False,
                             valid_size=args.valid_size, data_dir=config['data_dir'],
                             data_name=args.data_name, task_type=args.task_type) # 增加了task_type参数
    if args.do_train:
        run_train(args)

    if args.do_test:
        run_test(args)


if __name__ == '__main__':
    main()
