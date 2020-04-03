from sklearn import preprocessing
import numpy as np




from sklearn.feature_extraction.text import CountVectorizer
import jieba



#生成映射从单元字词到向量
def getOneHotDict(letter_list):
    enc = preprocessing.OneHotEncoder()
    enc.fit(letter_list)
    return enc

#将字词转化为热度向量
def letter_to_encode(letter, enc):
    ans = enc.transform([[letter]]).toarray()
    ans = np.array(ans)
    return ans

#将热度向量转为字词
def decode_to_letter(decode, enc):
    ans = enc.inverse_transform(decode)
    ans = np.array(ans)
    return ans


#将句子转为向量
def senten_to_encode(senten, enc):
    list_of_letter = []
    data = jieba.cut(senten)
    for item in data:
        item = letter_to_encode(item, enc)
        item= item.reshape(-1)
        list_of_letter.append(item)

    list_of_letter = np.array(list_of_letter)
    return list_of_letter

#将句子转为向量,但是将会前移一位
def get_encode_by_one(senten, enc):
    list_of_letter = []
    data = jieba.cut(senten)
    count = 0
    for item in data:
        if count == 0:

            count =count+1
            continue
        item = letter_to_encode(item, enc)
        item= item.reshape(-1)
        list_of_letter.append(item)
    
    #list_of_letter.append('end')
    item = letter_to_encode('end', enc)
    item= item.reshape(-1)
    list_of_letter.append(item)
    list_of_letter = np.array(list_of_letter)
    return list_of_letter




#将向量转为句子
def decode_to_senten(decode, enc):
    ans = enc.inverse_transform(decode)
    ans = np.array(ans)
    return ans






if __name__ == "__main__":

    letter_list = []
    data = jieba.cut("end网易是中国领先的互联网技术公司，为用户提供免费邮箱、游戏、搜索引擎服务，开设新闻、娱乐、体育等30多个内容频道，及博客、视频、论坛等互动交流，网聚人的力量, 网易评论不错哦")
    
    for temp in data:
        letter_list.append([temp])


    enc = getOneHotDict(letter_list)


    senten = get_encode_by_one('网易是公司', enc)
    print(senten)

    




    
    ans =  decode_to_senten(senten, enc)
    print(ans)