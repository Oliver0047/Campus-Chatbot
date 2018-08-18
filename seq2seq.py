# -*- coding: UTF-8 -*-
#autor:Oliver
import os
import random
import sys
import time
import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import sys
sys.path.append('.')
USE_CUDA = torch.cuda.is_available()#如果有GPU可以使用，那么使用GPU计算
EOS_token = 1#结束符
SOS_token = 2#开始符
f=open('data/enc.vocab','r',encoding='utf-8')
enc_vocab=f.readlines()
enc_len=len(enc_vocab)#编码表长度
f.flush()
f.close()
f=open('data/dec.vocab','r',encoding='utf-8')
dec_vocab=f.readlines()
dec_len=len(dec_vocab)#解码表长度
f.flush()
f.close()
del(enc_vocab)#消去变量
del(dec_vocab)

class EncoderRNN(nn.Module):#编码器
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size#输入大小，指问句中每个字或词的索引的one-hot编码维度，即编码表的大小
        self.hidden_size = hidden_size#隐含层大小
        self.n_layers = n_layers#RNN层数

        self.embedding = nn.Embedding(input_size, hidden_size)#形成词向量
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)#门控循环神经网络

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden


class Attn(nn.Module):#注意力机制
    def __init__(self, method, hidden_size, max_length):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        attn_energies = Variable(torch.zeros(seq_len)) 
        if USE_CUDA: attn_energies = attn_energies.cuda()

        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])#计算权重

        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)#利用softmax将权重归一化

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))#torch.dot指各个元素相乘然后相加，和numpy不同
            return energy

class AttnDecoderRNN(nn.Module):#加入了注意力机制的解码器
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, self.max_length)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):

        word_embedded = self.embedding(word_input).view(1, 1, -1) #解码器输入转词向量

        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)#将词向量与上一个背景向量连接
        rnn_output, hidden = self.gru(rnn_input, last_hidden)#rnn_output相当于当下解码器输出的上下文环境

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)#利用这个上下文环境计算新的背景向量权重
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))#形成新的背景向量

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)     
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))#根绝输入输出的上下文环境计算解码器当下的输出
        return output, context, hidden, attn_weights


class seq2seq(nn.Module):
    def __init__(self):
        super(seq2seq, self).__init__()
        self.max_epoches = 5000#最大训练次数
        self.batch_index = 0#从第0个问答序列开始
        self.GO_token = 2
        self.EOS_token = 1
        
        self.input_size = 1500#编码器词表大小
        self.output_size = 1500#解码器词表大小
        self.hidden_size = 1024
        self.max_length = 15#句长
        self.show_epoch = 100#每训练一百次显示一次训练数据
        self.use_cuda = USE_CUDA
        self.model_path = "./model/"
        self.n_layers = 1
        self.dropout_p = 0.05
        self.beam_search = True#使用束搜索
        self.top_k = 5#选择可能性最大的5个序列
        self.alpha = 0.5#惩罚因子
	
        self.enc_vec = []#编码表
        self.dec_vec = []#解码表

        # 初始化encoder和decoder
        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.n_layers)
        self.decoder = AttnDecoderRNN('general', self.hidden_size, self.output_size, self.n_layers, self.dropout_p, self.max_length)

        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        #设置优化器
        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())
        #设置损失函数
        self.criterion = nn.NLLLoss()

    def loadData(self):#导入编码数据和解码数据
        with open("./data/enc.vec") as enc:
            line = enc.readline()
            while line:
                self.enc_vec.append(line.strip().split())
                line = enc.readline()

        with open("./data/dec.vec") as dec:
            line = dec.readline()
            while line:
                self.dec_vec.append(line.strip().split())
                line = dec.readline()

    def next(self, batch_size, eos_token=1, go_token=2, shuffle=False):#取一份数据
        inputs = []
        targets = []

        if shuffle:#随机选择一行数据
            ind = random.choice(range(len(self.enc_vec)))
            enc = [self.enc_vec[ind]]
            dec = [self.dec_vec[ind]]
        else:#按顺序选择一个batch数据
            if self.batch_index+batch_size >= len(self.enc_vec):
                enc = self.enc_vec[self.batch_index:]
                dec = self.dec_vec[self.batch_index:]
                self.batch_index = 0
            else:
                enc = self.enc_vec[self.batch_index:self.batch_index+batch_size]
                dec = self.dec_vec[self.batch_index:self.batch_index+batch_size]
                self.batch_index += batch_size
        for index in range(len(enc)):
            #限制长度
            enc = enc[0][:self.max_length] if len(enc[0]) > self.max_length else enc[0]
            dec = dec[0][:self.max_length] if len(dec[0]) > self.max_length else dec[0]

            enc = [int(i) for i in enc]
            dec = [int(i) for i in dec]
            dec.append(eos_token)#为解码数据添加结束符

            inputs.append(enc)
            targets.append(dec)

        inputs = Variable(torch.LongTensor(inputs)).transpose(1, 0).contiguous()#封装为变量，并保证在一个内存块上
        targets = Variable(torch.LongTensor(targets)).transpose(1, 0).contiguous()
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
        return inputs, targets

    def train(self):#训练
        self.loadData()
        try:#如果有已知模型，就在已知模型上继续训练
            self.load_state_dict(torch.load(self.model_path+'params.pkl'))
        except Exception as e:
            print(e)
            print("No model!")
        loss_track = []

        for epoch in range(self.max_epoches):
            start = time.time()
            inputs, targets = self.next(1, shuffle=False)#取出一份数据
            loss, logits = self.step(inputs, targets, self.max_length)#返回损失值和输出
            loss_track.append(loss)
            _,v = torch.topk(logits, 1)#取出可能性最高的输出
            pre = v.cpu().data.numpy().T.tolist()[0][0]
            tar = targets.cpu().data.numpy().T.tolist()[0]
            stop = time.time()
            if epoch % self.show_epoch == 0:
                print("-"*50)
                print("epoch:", epoch)
                print("    loss:", loss)
                print("    target:%s\n    output:%s" % (tar, pre))
                print("    per-time:", (stop-start))
                torch.save(self.state_dict(), self.model_path+'params.pkl')

    def step(self, input_variable, target_variable, max_length):#一份数据前向传播，反向传播，参数更新
        teacher_forcing_ratio = 0.1
        clip = 5.0#梯度裁剪，防止梯度爆炸，这是RNN经常会出现的问题
        loss = 0 
        #每次训练将梯度归零
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)#编码

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden 
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        decoder_outputs = []
        use_teacher_forcing = random.random() < teacher_forcing_ratio#随机切换方式
        use_teacher_forcing = True
        if use_teacher_forcing:#使用正确的标签数据作为下一次解码器输入
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)#解码
                loss += self.criterion(decoder_output, target_variable[di])#累计损失
                decoder_input = target_variable[di]
                decoder_outputs.append(decoder_output.unsqueeze(0))
        else:#使用当下解码器输出作为下一次解码器输入
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_outputs.append(decoder_output.unsqueeze(0))
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                decoder_input = Variable(torch.LongTensor([[ni]]))
                if USE_CUDA: decoder_input = decoder_input.cuda()
                if ni == EOS_token: break
        loss.backward()#梯度反向传播
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)#梯度裁剪
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        self.encoder_optimizer.step()#参数优化
        self.decoder_optimizer.step()
        decoder_outputs = torch.cat(decoder_outputs, 0)#解码器输出
        return loss.data[0] / target_length, decoder_outputs

    def input_deal(self, input_vec):#将编码器输入向量限制长度，并封装为变量
        inputs = []
        enc = input_vec[:self.max_length] if len(input_vec) > self.max_length else input_vec#向量限制长度
        inputs.append(enc)
        inputs = Variable(torch.LongTensor(inputs)).transpose(1, 0).contiguous()#封装为变量
        if USE_CUDA:
            inputs = inputs.cuda()
        return inputs
    
    def prepare(self):
        try:
            self.load_state_dict(torch.load(self.model_path+'params.pkl'))#如果有模型就加载
        except Exception as e:
            print(e)
            print("No model!")
        # 加载字典
        self.str_to_vec = {}
        with open("./data/enc.vocab") as enc_vocab:
            for index,word in enumerate(enc_vocab.readlines()):
                self.str_to_vec[word.strip()] = index

        self.vec_to_str = {}
        with open("./data/dec.vocab") as dec_vocab:
            for index,word in enumerate(dec_vocab.readlines()):
                self.vec_to_str[index] = word.strip()
        
    
    def predict_one(self,data):
        # 字符串转向量
        segement = jieba.lcut(data.strip())
        input_vec = [self.str_to_vec.get(i, 3) for i in segement]
        input_vec = self.input_deal(input_vec)#向量处理

        samples = self.beamSearchDecoder(input_vec)#得到概率top5的结果
        samples.sort(key=lambda x:-x[3])
        sample=samples[0]#取出概率最大的序列结果
        outstrs = []
        for i in sample[0]:
            if i == 1:
                break
            outstrs.append(self.vec_to_str.get(i, "Un"))#序列转字符
        if ("Un" in outstrs) or ("__UNK__" in outstrs):
            return "风太大，我听不见><"
        return "".join(outstrs)
    
    def predict(self):#预测
        try:
            self.load_state_dict(torch.load(self.model_path+'params.pkl'))#如果有模型就加载
        except Exception as e:
            print(e)
            print("No model!")
        loss_track = []

        # 加载字典
        str_to_vec = {}
        with open("./data/enc.vocab",encoding='utf-8') as enc_vocab:
            for index,word in enumerate(enc_vocab.readlines()):
                str_to_vec[word.strip()] = index

        vec_to_str = {}
        with open("./data/dec.vocab",encoding='utf-8') as dec_vocab:
            for index,word in enumerate(dec_vocab.readlines()):
                vec_to_str[index] = word.strip()

        while True:
            input_strs = input(">> ")
            # 字符串转向量
            segement = jieba.lcut(input_strs)
            input_vec = [str_to_vec.get(i, 3) for i in segement]
            input_vec = self.input_deal(input_vec)#向量处理

            # 选择序列输出方式
            if self.beam_search:#采用beam search
                samples = self.beamSearchDecoder(input_vec)#得到概率top5的结果
                samples.sort(key=lambda x:-x[3])
                sample=samples[0]#取出概率最大的序列结果
                outstrs = []
                for i in sample[0]:
                    if i == 1:
                        break
                    outstrs.append(vec_to_str.get(i, "Un"))#序列转字符
                print("小电 > ", "".join(outstrs))
            else:#普通的序列输出
                logits = self.normal_search(input_vec)#按照每个时刻选择最高概率的字符输出，得到最终序列
                _,v = torch.topk(logits, 1)
                pre = v.cpu().data.numpy().T.tolist()[0][0]
                outstrs = []
                for i in pre:
                    if i == 1:
                        break
                    outstrs.append(vec_to_str.get(i, "Un"))
                print("小电 > ", "".join(outstrs))

    def normal_search(self, input_variable):#按照每个时刻选择最高概率的字符输出，得到最终序列
        input_length = input_variable.size()[0]

        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
        decoder_outputs = []

        for i in range(self.max_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output.unsqueeze(0))
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]])) #使用当下解码器输出作为下一次解码器输入
            if USE_CUDA: decoder_input = decoder_input.cuda()
            if ni == EOS_token: break

        decoder_outputs = torch.cat(decoder_outputs, 0)
        return decoder_outputs

    def tensorToList(self, tensor):#tensor转list
        return tensor.cpu().data.numpy().tolist()[0]

    def beamSearchDecoder(self, input_variable):#Beam Search算法
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        topk = decoder_output.data.topk(self.top_k)#输入开始符，得到前5大概率的输出字符以及对应信息
        samples = [[] for i in range(self.top_k)]
        dead_k = 0
        final_samples = []
        for index in range(self.top_k):#储存前5大概率的输出字符，以及对应的分数，背景向量等
            topk_prob = topk[0][0][index]
            topk_index = int(topk[1][0][index])
            samples[index] = [[topk_index], topk_prob, 0, 0, decoder_context, decoder_hidden, decoder_attention, encoder_outputs]

        for _ in range(self.max_length):
            tmp = []
            for index in range(len(samples)):
                tmp.extend(self.beamSearchInfer(samples[index], index))#对每个储存的字符序列，继续预测下一个输出字符，保留前5大概率的字符输出
            samples = []

            # 筛选出topk
            df = pd.DataFrame(tmp)#封装成数据帧格式
            df.columns = ['sequence', 'pre_socres', 'fin_scores', "ave_scores", "decoder_context", "decoder_hidden", "decoder_attention", "encoder_outputs"]
            sequence_len = df.sequence.apply(lambda x:len(x))#取出序列长度
            df['ave_scores'] = df['fin_scores'] / sequence_len#计算平均分
            df = df.sort_values('ave_scores', ascending=False).reset_index().drop(['index'], axis=1)#根据平均分从大到小排序
            df = df[:(self.top_k-dead_k)]#最多取5个带结束符的序列
            for index in range(len(df)):
                group = df.ix[index]#取出序列已经对应信息
                if group.tolist()[0][-1] == 1:#如果该序列的结尾是结束符
                    final_samples.append(group.tolist())#那就加入最终输出序列组中
                    df = df.drop([index], axis=0)#舍弃该序列
                    dead_k += 1#表示需要的序列数量减一
                    #print("drop {}, {}".format(group.tolist()[0], dead_k))
            samples = df.values.tolist()
            if len(samples) == 0:#如果已经没有序列了，那就可以结束了
                break

        if len(final_samples) < self.top_k:
            final_samples.extend(samples[:(self.top_k-dead_k)])#如果最终序列的数量不够，那就取几个概率较大的补上
        return final_samples

    def beamSearchInfer(self, sample, k):#计算已知序列的下一个输出字符，并计算分数
        samples = []
        decoder_input = Variable(torch.LongTensor([[sample[0][-1]]]))
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
        sequence, pre_scores, fin_scores, ave_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs = sample
        decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

        # choose topk
        topk = decoder_output.data.topk(self.top_k)
        for k in range(self.top_k):
            topk_prob = topk[0][0][k]#取出该字符概率
            topk_index = int(topk[1][0][k])#取出该字符索引
            pre_scores += topk_prob#分数累加
            fin_scores = pre_scores - (k - 1 ) * self.alpha#加入惩罚因子
            #数据更新
            samples.append([sequence+[topk_index], pre_scores, fin_scores, ave_scores, decoder_context, decoder_hidden, decoder_attention, encoder_outputs])
        return samples

    def retrain(self):#从头开始训练
        try:
            os.remove(self.model_path)
        except Exception as e:
            pass
        self.train()

if __name__ == '__main__':
    seq = seq2seq()
    if sys.argv[1] == 'train':#训练模式
        seq.train()
    elif sys.argv[1] == 'predict':#预测模式
        seq.predict()
    elif sys.argv[1] == 'retrain':#从头开始训练
        seq.retrain()
