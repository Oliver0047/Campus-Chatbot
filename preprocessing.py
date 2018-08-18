# -*- coding: UTF-8 -*-
#autor:Oliver
import jieba

class preprocessing():
    __PAD__ = 0#填充符
    __EOS__ = 1#结束符
    __GO__ = 2#开始符
    __UNK__ = 3#未知符
    vocab = ['__PAD__', '__EOS__', '__GO__','__UNK__']
    def __init__(self):
        self.encoderFile = "./data/question.txt"#问题
        self.decoderFile = "./data/answer.txt"#回答
        self.savePath = './data/'#储存路径
        jieba.load_userdict("./data/supplementvocab.txt")#选择jieba的中文分词字典
    
    def wordToVocabulary(self, originFile, vocabFile, segementFile):
        vocabulary = []
        sege = open(segementFile, "w",encoding='utf-8')
        with open(originFile, 'r',encoding='utf-8') as en:
            for sent in en.readlines():
                if "enc" in segementFile:
                    words = jieba.lcut(sent.strip())#jieba分词，返回列表
                    print(words)
                else:
                    words = jieba.lcut(sent.strip())
                vocabulary.extend(words)#初步形成字典
                for word in words:#储存每行分词结果
                    sege.write(word+" ")
                sege.write("\n")
        sege.close()

        # 去重并存入词典
        vocab_file = open(vocabFile, "w",encoding='utf-8')
        _vocabulary = list(set(vocabulary))
        _vocabulary.sort(key=vocabulary.index)
        _vocabulary = self.vocab + _vocabulary#加入特殊符号形成最终字典
        if "enc" in segementFile:
            print('encode_vocab_length: ',len(_vocabulary))
        else:
            print('decode_vocab_length: ',len(_vocabulary))
        for index, word in enumerate(_vocabulary):
            vocab_file.write(word+"\n")
        vocab_file.close()

    def toVec(self, segementFile, vocabFile, doneFile):
        word_dicts = {}
        vec = []
        with open(vocabFile, "r",encoding='utf-8') as dict_f:#将字典封装成索引词表
            for index, word in enumerate(dict_f.readlines()):
                word_dicts[word.strip()] = index

        f = open(doneFile, "w",encoding='utf-8')
        #如果单独或者连续输入未知符号，则回答未知符号
        if "enc.vec" in doneFile:
            f.write("3 3 3 3\n")
            f.write("3\n")
        elif "dec.vec" in doneFile:
            f.write(str(word_dicts.get("other", 3))+"\n")
            f.write(str(word_dicts.get("other", 3))+"\n")
        with open(segementFile, "r",encoding='utf-8') as sege_f:
            for sent in sege_f.readlines():
                sents = [i.strip() for i in sent.split(" ")[:-1]]
                vec.extend(sents)
                for word in sents:
                    f.write(str(word_dicts.get(word))+" ")#将字词转为索引号
                f.write("\n")
        f.close()
            

    def main(self):
        # 获得字典
        self.wordToVocabulary(self.encoderFile, self.savePath+'enc.vocab', self.savePath+'enc.segement')
        self.wordToVocabulary(self.decoderFile, self.savePath+'dec.vocab', self.savePath+'dec.segement')
        # 转向量
        self.toVec(self.savePath+"enc.segement", 
                   self.savePath+"enc.vocab", 
                   self.savePath+"enc.vec")
        self.toVec(self.savePath+"dec.segement", 
                   self.savePath+"dec.vocab", 
                   self.savePath+"dec.vec")


if __name__ == '__main__':
    pre = preprocessing()
    pre.main()
