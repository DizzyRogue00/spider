import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from word2vec import get_time_diff
from tensorboardX import SummaryWriter
from classify import config,BertForSentenceClassification
from overload import overload
from transformers import BertTokenizer
import utils_bert
import os
import logging
import copy

#initial default:xavier
def init_network(model,method='xavier',exclude='embedding',seed=1223):
    for name,w in model.named_parameters():
        if exclude not in name:
            dimensions=w.dim()
            if 'weight' in name:
                if dimensions<2:
                    nn.init.normal_(w)
                else:
                    if method =='xavier':
                        nn.init.xavier_normal_(w)
                    elif method=='kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w,0)
            else:
                pass

@overload
def train(config,model,train_iter,train_original):#(config,model,train_iter,dev_iter,test_iter)
    start_time=time.time()
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    total_batch=0
    #dev_best_loss=float('inf')
    train_best_loss=float('inf')
    last_improve=0
    flag=False
    writer=SummaryWriter(log_dir=config.log_path+'/'+time.strftime('%m-%d_%H.%M',time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch+1,config.num_epochs))

        for i,(train_data,labels) in enumerate(train_iter):
            output=model(train_data)#[batch_size,num_classes]
            model.zero_grad()
            loss=F.cross_entropy(output,labels)
            loss.backward()
            optimizer.step()
            if total_batch%1==0:
                true=labels.data.cpu()
                pred=torch.max(output.data,1)[1].cpu()
                train_batch_acc=metrics.accuracy_score(true,pred)
                train_batch_f1_score=metrics.f1_score(true,pred)
                train_acc,train_loss,train_report,train_confusion=evaluate(config,model,train_original)
                if train_loss<=train_best_loss:
                    train_best_loss=train_loss
                    torch.save(model.state_dict(),config.save_path)
                    improve='*'
                    last_improve=total_batch
                else:
                    improve=''
                time_dif=get_time_diff(start_time)
                msg = 'Iter: {0:>6},  Train Batch Loss: {1:>5.3},  Train Batch Acc: {2:>6.2%}, Train Batch F1 Score:{3:>5.3}  Val Loss: {4:>5.2},  Val Acc: {5:>6.2%},Val Report:{6},Val Confusion Matrix:{7}  Time: {8} {9}'
                print(msg.format(total_batch,loss.item(),train_batch_acc,train_batch_f1_score,train_loss,train_acc,train_report,train_confusion,time_dif,improve))
                writer.add_scalar('Batch Loss/Train',loss.item(),total_batch)
                writer.add_scalar('Batch Accu/Train',train_batch_acc,total_batch)
                writer.add_scalar('Loss/Train',train_loss,total_batch)
                writer.add_scalar('Accu/Train',train_acc,total_batch)
                model.train()
            total_batch+=1
            if total_batch-last_improve >config.require_improvement:
                print("No optimization for a long time, auto-stopping")
                flag=True
                break
            print(i,flag)

        if flag:
            break
    writer.close()

'''
def evaluate(config,model,data_iter,test=False):
    model.eval()
    loss_total=0
    predict_all=np.array([],dtype=int)
    labels_all=np.array([],dtype=int)
    with torch.no_grad():
        for data,labels in data_iter:
            output=model(data)
            loss=F.cross_entropy(output,labels)
            loss_total+=loss
            labels=labels.data.cpu().numpy()
            pred=torch.max(output.data,1)[1].cpu.numpy()
            labels_all=np.append(labels_all,labels)
            predict_all=np.append(predict_all,pred)
    acc=metrics.accuracy_score(labels_all,predict_all)
    if test:
        report=metrics.classification_report(labels_all,predict_all,target_names=config.class_list,digits=4)
        confusion=metrics.confusion_matrix(labels_all,predict_all)
        return acc,loss_total/len(data_iter),report,confusion
    return acc,loss_total/len(data_iter)
'''
@overload
def evaluate(config,model,data_iter):
    model.eval()
    loss_total=0
    predict_all=np.array([],dtype=int)
    labels_all=np.array([],dtype=int)
    with torch.no_grad():
        for data,labels in data_iter:
            output=model(data)
            loss=F.cross_entropy(output,labels)
            loss_total+=loss
            labels = labels.data.cpu().numpy()
            pred = torch.max(output.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, pred)
    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return acc, loss_total/len(data_iter),report,confusion

@overload
def train(config):
    start_time=time.time()
    model=BertForSentenceClassification(config,config.pretrained_model_dir)
    if os.path.exists(config.save_path):
        load_params=torch.load(config.save_path)
        model.load_state_dict(load_params)
        logging.info("Successfully load the existed model. Begin to train the model")
    model = model.to(config.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    model.train()
    bert_tokrnize=BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader=utils_bert.LoadDataset(
        vocab_path=config.bert_vocab_path,
        tokenizer=bert_tokrnize,
        batch_size=config.batch_size,
        max_sen_len=config.pad_size,
        split_sep='\t',
        max_position_embeddings=config.bert_max_position_embeddings,
        pad_index=config.bert_pad_token_id
        )
    train_data=data_loader.data_process(config.train_path)#[([...],1,12),([...],0,32)]
    train_iter=utils_bert.build_iter(train_data,config)
    dev_iter=copy.deepcopy(train_iter)
    time_diff=get_time_diff(start_time)
    print("Time usage:",time_diff)

    total_batch=0
    #dev_best_loss=float('inf')
    train_best_loss=float('inf')
    last_improve=0
    flag=False
    max_acc=0

    writer=SummaryWriter(log_dir=config.log_path+'/'+time.strftime('%m-%d_%H.%M',time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        losses=0
        for i, (sample, labels) in enumerate(train_iter):
            sample=sample.to(config.device)
            labels=labels.to(config.device)
            loss,logits=model(
                input_ids=sample,
                token_type_ids=None,
                position_ids=None,
                labels=labels
            ) #logits: [batch_size,num_classes]
            #model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            loss_temp = F.cross_entropy(logits, labels)
            loss_temp.backward()
            optimizer.step()
            losses+=loss.item()
            train_batch_acc_evaluate=(logits.argmax(1)==labels).float().mean()
            if total_batch % 1 == 0:
                true = labels.data.cpu()
                pred = torch.max(logits.data, 1)[1].cpu()
                train_batch_acc = metrics.accuracy_score(true, pred)
                train_batch_f1_score = metrics.f1_score(true, pred)
                train_acc_evaluate,train_acc, train_loss, train_report, train_confusion = evaluate(config, model, dev_iter,label=None)
                if train_loss <= train_best_loss:
                    train_best_loss = train_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                if train_acc_evaluate>max_acc:
                    max_acc=train_acc_evaluate
                    torch.save(model.state_dict(),config.save_path)
                    improve+='+'
                else:
                    improve+=''
                time_dif = get_time_diff(start_time)
                msg = 'Epoch:{0},Batch:{1}/{2},Iter: {3:>6},Train Batch Loss Evaluate:{4:>5.3},Train Batch Loss: {5:>5.3},Train Batch Acc Evaluate:{6:>6.2%},Train Batch Acc: {7:>6.2%}, Train Batch F1 Score:{8:>5.3},Val Loss Evaluate:{9:5.2},Val Loss: {10:>5.2},Val Acc Evaluate:{11:>6.2%},  Val Acc: {12:>6.2%},Val Report:{13},Val Confusion Matrix:{14}  Time: {15} {16}'
                print(msg.format(epoch,i,len(train_iter),total_batch,loss.item(),loss_temp.item(),train_batch_acc_evaluate.item(),train_batch_acc, train_batch_f1_score, losses,train_loss, train_acc_evaluate,train_acc,
                                 train_report, train_confusion, time_dif, improve))
                logging.info(msg.format(epoch, i, len(train_iter), total_batch, loss.item(), loss_temp.item(),
                                 train_batch_acc_evaluate.item(), train_batch_acc, train_batch_f1_score, losses,
                                 train_loss, train_acc_evaluate, train_acc,
                                 train_report, train_confusion, time_dif, improve))
                writer.add_scalar('Batch Loss/Train', loss_temp.item(), total_batch)
                writer.add_scalar('Batch Accu/Train', train_batch_acc, total_batch)
                writer.add_scalar('Loss/Train', train_loss, total_batch)
                writer.add_scalar('Accu/Train', train_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping")
                flag = True
                break

        if flag:
            break
    writer.close()

@overload
def evaluate(config,model,data_iter,label=None):
    model.eval()
    loss_total=0
    predict_all=np.array([],dtype=int)
    labels_all=np.array([],dtype=int)
    with torch.no_grad():
        acc_sum=0.0
        num_data=0
        for data,labels in data_iter:
            data=data.to(config.device)
            labels=data.to(config.device)
            output=model(data,labels=label)
            acc_sum+=(output.argmax(1)==labels).float().sum().item()
            num_data+=len(labels)
            loss=F.cross_entropy(output,labels)
            loss_total+=loss
            labels = labels.data.cpu().numpy()
            pred = torch.max(output.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, pred)
    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    acc_evaluate=acc_sum/num_data
    return acc_evaluate,acc, loss_total/len(data_iter),report,confusion