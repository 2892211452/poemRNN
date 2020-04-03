from data import *
from keras.utils import to_categorical

def get_all_senten():
    f = open("/home/lwl/code/python/test/test1/data.txt")             # 返回一个文件对象  
    line = f.readline()             # 调用文件的 readline()方法  
    ans =[]
    while line:  
        # print(line, end = '')　　　# 在 Python 3中使用  
        line = f.readline()  
        line = line.replace('　　', '')
        data = jieba.cut(line)
        list1 = []
        for item in data:
            list1.append(item)
        if (len(list1)>1):
            list1.pop()
            temp = ""
            for i in list1:
                temp = temp + i
            ans.append(temp)

    return ans

def get_all_word():
    f = open("/home/lwl/code/python/test/test1/data.txt")             # 返回一个文件对象  
    line = f.readline()             # 调用文件的 readline()方法  
    ans =[]
    ans = np.array(ans)
    while line:  
        # print(line, end = '')　　　# 在 Python 3中使用  
        line = f.readline()  
        line = line.replace('　　', '')
        data = jieba.cut(line)
        list1 = []
        for item in data:
            list1.append(item)
        if (len(list1)>1):
            list1.pop()
            list1 = np.array(list1)
            ans = np.append(ans,list1)
    ans = np.append(ans, ['end'])
    temp =ans.reshape(-1, 1)
    return temp



# 搭建神经网络
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.embed = nn.Embedding(176, 100)
        input_size = 100


        self.hidden_size = hidden_size

        self.i2h = nn.Linear( input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear( input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        temp =  np.argmax(np.array(input)).astype(int)
        temp = torch.tensor(temp)
        input = self.embed(temp)
        input = input.view(1, -1)

        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
    
    def seqForward(self, input,target):

        criterion = nn.NLLLoss()

        learning_rate = 0.01

        target_line_tensor = target.reshape(-1, 1, 175)

        hidden = rnn.initHidden()

        rnn.zero_grad()

        loss = 0
        input_line_tensor = input.reshape(-1, 1, 175)
        
        for i in range(input_line_tensor.size(0)-1):
            output, hidden = rnn(input_line_tensor[i].float(), hidden)
        
            l = criterion(output,  torch.max(target_line_tensor[i], 1)[1])
            if i == (input_line_tensor.size(0)-2):

                loss += l

        loss.backward()

        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)

        return output, loss.item() / input_line_tensor.size(0)




    def initHidden(self):
        return torch.zeros(1, self.hidden_size)





#我们一共174个中文词加上end就是175
N_letter = 175
rnn = RNN(N_letter, 328, N_letter)
print(rnn)                




def train( input_line_tensor, target_line_tensor):


    criterion = nn.NLLLoss()

    learning_rate = 0.01

    target_line_tensor = target_line_tensor.reshape(-1, 1, 175)

    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0
    input_line_tensor = input_line_tensor.reshape(-1, 1, 175)
    
    for i in range(input_line_tensor.size(0)-1):
        output, hidden = rnn(input_line_tensor[i].float(), hidden)
    
        l = criterion(output,  torch.max(target_line_tensor[i], 1)[1])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)






if __name__ == "__main__":
    words = get_all_word()
    print(words)
    enc = getOneHotDict(words)# dict
        
    if True:
   
        sentens  = get_all_senten()

        for step in range(200):
            loss = 0
            for item in sentens:
            
                encode = senten_to_encode(item, enc)
                encode = torch.tensor(encode)

                target = get_encode_by_one(item, enc)
                target = torch.tensor(target)
                if encode.size(0) <=1:
                    continue
            #     out , l = train(encode, target)
                out , l = rnn.seqForward(encode, target)
                loss+=l
            print(loss)
        torch.save(rnn, './model1.h5') 
    else:
        model = torch.load('./model1.h5')
        hidden = model.initHidden()
        start = '问'
        input = letter_to_encode(start,enc)
        input = torch.tensor(input, requires_grad=True)
        input = input.reshape(1, -1)
        num = 10
        print(start, end="")
        
        for i in range(num):
            input = input.detach().numpy()
            output , hidden = model(input, hidden)
            input=output
            ch = input.data
            ch = ch.reshape(-1)
            ch = np.array(ch)
            ans = np.argmax(ch)
            temp = np.zeros((1, 175))
            temp[0][ans] = 1
            ans = decode_to_letter(temp, enc)
            print(ans[0][0], end='')
        
