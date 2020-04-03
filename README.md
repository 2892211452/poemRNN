# poemRNN

项目的数据加载是自己写的,主要使用jieba分词以及sklearn还有numpy等库实现,
将结巴分词后然后使用sklearn建立搭建单词和one-hot向量之间的转化关系
然后使用numpy进行数据的处理,例如维度变化等等.



# 1.0版本
## 写诗的主要思路就是rnn生成
细节
- 设立有单词end,作为结束词,但是因为我的数据样本太小,而且语句太短了,导致基本大部分的词都指向了end词, loss降低不下来.这就十分尴尬.于是我将这个end禁用了,
个人任务如果改变样本就还是可行的.loss也降下来啦.


# 2.0版本
## 添加word2vct以及attention机制
- 使用embeding后效果很好,loss降低的很快.
- 使用attention, 