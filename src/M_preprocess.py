import os
import re
import pickle
import pandas as pd
import feather
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm_notebook as tqdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold,KFold

# 读取数据-------------------------------------------------------
# 正常读取数据
train_df = pd.read_csv('../../data/raw_data/train_set.csv')
test_df = pd.read_csv('../../data/raw_data/test_set.csv')

# 分开读取数据
def gen_csv_feather(path, path_new):
    f = open(path)
    reader = pd.read_csv(f, sep=',', iterator=True)
    loop = True
    chunkSize = 10000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df = pd.concat(chunks, ignore_index=True)
    print(df.count())
    feather.write_dataframe(df, path_new)

gen_csv_feather("../data/train_set.csv", "../data/train_set.feather")
gen_csv_feather("../data/test_set.csv", "../data/test_set.feather")

# 分词，格式的处理，一般处理成[[][]]貌似
# 类别和编码的处理

# 数据探索 --------------------------------------------------------
# 看分类个数
train_df['class'].value_counts().plot.bar(figsize=(10,6), fontsize=20, rot=0)

# 句子长度分布

# 基础特征处理部分 -------------------------------------------------
# 处理低idf 词


# 特征部分
# tfidf特征 重要的一元特征 tfidf怎么用到NN的？
vec = TfidfVectorizer(ngram_range=(1,1),min_df=5, max_df=0.8,use_idf=1,smooth_idf=1, sublinear_tf=1)
trainX = vec.fit_transform(train['word_seg'])

ch2 = SelectKBest(chi2, k=150000)
trainX = ch2.fit_transform(trainX,label)
feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
# 其他ngram 的tfidf词
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])

# tf-idf 并且降维
vectorizer = TfidfVectorizer(ngram_range=(1,2),
                             stop_words=sw_list,
                             sublinear_tf=True,
                             use_idf=True,
                             norm='l2',
                             max_features=10000
                             )
svd = TruncatedSVD(n_components=n_dim)
lsa = make_pipeline(vectorizer, svd)
train_X = lsa.fit_transform(train_X)
test_X = lsa.transform(test_X)
os.makedirs('../../data/feature/', exist_ok=True)
with open('../../data/feature/make_pipeline_{}_{}.pkl'.format(feature, n_dim), 'wb') as f:
    pickle.dump(lsa, f)
# 保存两个tfidf特征
with open('../../data/feature/train_x_{}_{}.pkl'.format(feature, n_dim), 'wb') as f:
    pickle.dump(train_X, f)
with open('../../data/feature/test_x_{}_{}.pkl'.format(feature, n_dim), 'wb') as f:
    pickle.dump(test_X, f)


# 处理低频词
def construct_dict(df, d_type='word'):
    # 构建词库
    word_dict = {}
    corput = df.word_seg if d_type == 'word' else df.article
    for line in tqdm(corput):
        for e in line.strip().split():
            word_dict[e] = word_dict.get(e, 0) + 1
    return word_dict
char_dict = construct_dict(train_df, d_type='char')
# 过滤词库中低频词
char_stop_word = [e for e in char_dict if char_dict[e] <=2]
# 过滤低频词
def filter_low_freq(df):
    min_freq = 2
    word_seg_mf2 = []
    char_mf2 = []
    for w in tqdm(df.word_seg):
        word_seg_mf2.append(' '.join([e for e in w.split() if word_dict[e] > min_freq]))
    for w in tqdm(df.article):
        char_mf2.append(' '.join([e for e in w.split() if char_dict[e] > min_freq]))
    df['word_mf2'] = word_seg_mf2
    df['char_mf2'] = char_mf2
filter_low_freq(train_df)


# 构建过滤了低频次和低idf 的词库
corpus = []
# 所有词
for doc in train['word_seg'].tolist() + test['word_seg'].tolist():
    doc = doc.split()
    corpus.append(doc)

def make_idf_vocab(train_data):
    # 词频大的tfidf 词库
    if os.path.exists('./new_data/vocab.pkl'):
        vocab = pickle.load(open('./new_data/vocab.pkl','rb'))
    else:
        word_to_doc = {}
        idf = {}
        total_doc_num = float(len(train_data))

        for doc in train_data:
            for word in set(doc):
                if word not in word_to_doc.keys():
                    word_to_doc[word] = 1
                else:
                    word_to_doc[word] += 1

        for word in word_to_doc.keys():
            # 只算词频大的的词的idf ?
            if word_to_doc[word] > 10:
                idf[word] = np.log(total_doc_num/(word_to_doc[word]+1))

        sort_idf = sorted(idf.items(),key=lambda x:x[1])# 按idf 排序
        vocab = [x[0] for x in sort_idf]
        pickle.dump(vocab,open('./new_data/vocab.pkl','wb'))
    return vocab
vocab = make_idf_vocab(corpus)
# 过滤词
vocab_dict = {w:1 for w in vocab}
def filter_word(x):
    x = x.split()
    x = [w for w in x if w in vocab_dict.keys()]
    return ' '.join(x)

train['word_seg'] = train['word_seg'].map(lambda x : filter_word(x))
test['word_seg'] = test['word_seg'].map(lambda x : filter_word(x))



# 词频特征
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
cv = CountVectorizer(max_features=500000,max_df=0.7, min_df=3, lowercase=False ,ngram_range=(2,3))
trainX_cv = cv.fit_transform(train['word_seg'])
testX_cv = cv.transform(test['word_seg'])

# 这些统计的数值特征要归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)


# w2v特征 -----------------------------------
x_train = train['word_seg']
# x_train 是分词后的某一列特征
w2v=Word2Vec(size=300)
w2v.build_vocab(x_train)
w2v.train(x_train,total_examples=w2v.corpus_count,epochs=10)

# doc2vec特征
def buildDocVector(text,size):
    vec=np.zeros(size).reshape((1,size))
    count=0
    for word in text:
        try:
            vec+=w2v[word].reshape((1,size))
            count+=1
        except KeyError:
            continue
    if count !=0:
        vec/=count
    return vec

# train_vec=np.concatenate([buildDocVector(z,300) for z in x_train])
# train_vec=scale(train_vec)
# w2v.train(x_test,total_examples=w2v.corpus_count,epochs=10)
# test_vec=np.concatenate([buildDocVector(z,300) for z in x_test])
# test_vec=scale(test_vec)
logging.info("w2v处理特征完成")


# 训练词向量，保存模型
def train_w2v_model(type='article', min_freq=5, size=100):
    sentences = []
    if type == 'char':
        corpus = pd.concat((train_df['article'], test_df['article']))
    elif type == 'word':
        corpus = pd.concat((train_df['word_seg'], test_df['word_seg']))
    for e in tqdm(corpus):
        sentences.append([i for i in e.strip().split() if i])
    print('训练集语料:', len(corpus))
    print('总长度: ', len(sentences))
    model = Word2Vec(sentences, size=size, window=5, min_count=min_freq)
    model.itos = {}
    model.stoi = {}
    model.embedding = {}
    print('保存模型...')
    # model.wv.vocab.keys() 所有的词
    for k in tqdm(model.wv.vocab.keys()):
        model.itos[model.wv.vocab[k].index] = k
        model.stoi[k] = model.wv.vocab[k].index
        model.embedding[model.wv.vocab[k].index] = model.wv[k]
    model.save('../../data/word2vec-models/word2vec.{}.{}d.mfreq{}.model'.format(type, size, min_freq))
    return model
model = train_w2v_model(type='char', size=100)
model = train_w2v_model(type='word', size=100)


# 数据增强，只对少类别增强了一部分
# 随机数添加数据
import random
def run_enhance():
    # 最大的类别出现了多少次
    max_len = train_df['class'].value_counts().values[0]
    enhance_df = train_df.copy()[['article', 'word_seg', 'class', 'c_numerical']]
    # c是各个类别
    for c in tqdm(enhance_df['class'].value_counts().index):
        # 筛选出等于c的那些行
        c_data = enhance_df[enhance_df['class'] == c]
        # 类别少的那些类做才做增强
        if len(c_data) * 2 < max_len:
            for a, b, c_n in zip(c_data['word_seg'].values, c_data['article'].values, c_data['c_numerical'].values):
                a_lst = a.split()
                b_lst = b.split()
                random.shuffle(a_lst)
                random.shuffle(b_lst)
                a_str = ' '.join(a_lst)
                b_str = ' '.join(b_lst)
                enhance_df.loc[enhance_df.shape[0]+1] = {'article': b_str, 'word_seg': a_str, 'class': c, 'c_numerical':c_n}
    return enhance_df
enhance_df = run_enhance()
enhance_df.to_csv('../../data/Enhance.train.csv')
enhance_df['class'].value_counts().plot.bar(figsize=(10,6), fontsize=20, rot=0)