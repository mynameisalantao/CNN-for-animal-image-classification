#------------------------------ Import Module --------------------------------#
import numpy as np
import cv2
import os
import tensorflow as tf
import math
import matplotlib.pyplot as plt 

#---------------------------------- Reminded ---------------------------------#
# windows读取文件可以用\，但在字符串里面\被作为转义字符使用
# 那么python在描述路径时有两种方式：
#  'd:\\a.txt'，转义的方式
#  r'd:\a.txt'，声明字符串不需要转义
# 推荐使用此写法“/"，可以避免很多异常

#--------------------------------- Parameter ---------------------------------#
image_heigh=60            # 統一圖片高度
image_width=60            # 統一圖片寬度
data_number=1000          # 每種類動物要取多少筆data來train
data_test_number=400      # 取多少筆testing data來算正確率
race=10                   # 總共分為10種動物
batch_size=50             # 多少筆data一起做訓練
layer1_node=60            # 第1層的節點數
layer2_node=60            # 第2層的節點數
layer3_node=1024          # 第3層的節點數
layer4_node=100           # 第4層的節點數
output_node=race          # 輸出層的節點數(輸出)
epoch_num=12              # 執行多少次epoch
record_train_accuracy=[]  # 紀錄每次epoch的訓練正確率
record_test_accuracy=[]   # 紀錄每次epoch的測試正確率
record_xentropy=[]        # 紀錄每次epoch的cross entropy

#---------------------------------- Function ---------------------------------#
# 讀取圖片
def read_image(path,data_number):
    imgs = os.listdir(path)      # 獲得該路徑下所有的檔案名稱
    #total_image=np.zeros([len(imgs),image_heigh,image_width,3])
    total_image=np.zeros([data_number,image_heigh,image_width,3])   
    # 依序將每張圖片儲存進矩陣total_image當中
    #for num_image in range(0,len(imgs)):
    for num_image in range(0,data_number):
        filePath=path+'//'+imgs[num_image]    # 圖片路徑
        cv_img=cv2.imread(filePath)  # 取得圖片
        total_image[num_image,:,:,:] = cv2.resize(cv_img, (image_heigh, image_width), interpolation=cv2.INTER_CUBIC)  # resize並且存入total_image當中      
    return total_image

# 產生Weight參數
def weight_generate(shape):
    initialize=tf.truncated_normal(shape,stddev=1/math.sqrt(float(image_heigh*image_width)))
    return tf.Variable(initialize)

# 產生Bias參數
def bias_generate(shape):
    initialize=tf.truncated_normal(shape,stddev=1/math.sqrt(float(image_heigh*image_width)))
    return tf.Variable(initialize)
    
# Convoluation
def conv(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# Max Pooling
def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

def get_batch(training_data_index,get_batch_number,training_data):
    # training_data_index是存資料在training_data的index和其對應的label號碼
    # get_batch_number為要取第幾個batch的資料
    temp_data=training_data_index[int(get_batch_number*batch_size):int((get_batch_number+1)*batch_size),0]
    batch_data=training_data[temp_data.astype(int),:]
    temp_label=training_data_index[int(get_batch_number*batch_size):int((get_batch_number+1)*batch_size),1]
    batch_label=np.eye(race)[temp_label.astype(int),:]
    return batch_data,batch_label

#-------------------------------- Input Data ---------------------------------#
# 建立training data的index
training_data_index=np.zeros([data_number*race,2])     # 第1行為data的index,第2行為其對應的label
training_data_index[:,0]=np.linspace(0,data_number*race-1,data_number*race)   # data的index先按順序

# 建立testing data的index
testing_data_index=np.zeros([data_test_number*race,2])     # 第1行為data的index,第2行為其對應的label
testing_data_index[:,0]=np.linspace(0,data_test_number*race-1,data_test_number*race)   # data的index先按順序

# 傳入training data
training_data=np.zeros([data_number*race,image_heigh,image_width,3]) # training data
#path=os.getcwd()            # 取得目前jupyter所在的位置(路徑)
path=r'/home/alantao/deep learning/DL HW2'
training_data_path=path+'/animal-10/train'     # 取得training data所在的路徑
file_name = os.listdir(training_data_path)
for file_num in range(0,race):
    filePath=training_data_path+'//'+file_name[file_num]    # 資料路徑
    training_data[file_num*data_number:file_num*data_number+data_number,:,:,:]=read_image(filePath,data_number)
    training_data_index[file_num*data_number:file_num*data_number+data_number,1]=file_num  # 放入其對應的種族(0~9)
    
# 傳入testing data
testing_data=np.zeros([data_test_number*race,image_heigh,image_width,3]) # training data
#path=os.getcwd()            # 取得目前jupyter所在的位置(路徑)
testing_data_path=path+'/animal-10/val'     # 取得testing data所在的路徑
file_name = os.listdir(testing_data_path)
for file_num in range(0,race):
    filePath=testing_data_path+'//'+file_name[file_num]    # 資料路徑
    testing_data[file_num*data_test_number:(file_num+1)*data_test_number,:,:,:]=read_image(filePath,data_test_number)
    testing_data_index[file_num*data_test_number:(file_num+1)*data_test_number,1]=file_num  # 放入其對應的種族(0~9)    

# 修改資料型態
#training_data=training_data.reshape([-1,image_heigh*image_width,3])  # 把每個顏色的2為圖片拉長
training_data=training_data.reshape([-1,image_heigh*image_width*3])  # 把每個顏色的2為圖片(連同RGB)拉長
testing_data=testing_data.reshape([-1,image_heigh*image_width*3])  # 把每個顏色的2為圖片(連同RGB)拉長
    
#----------------------------------- CNN -------------------------------------# 
# 建立Session
sess=tf.InteractiveSession()    
    
# 輸入點設置 data 與 label
images_placeholder=tf.placeholder(tf.float32,shape=(None,image_heigh*image_width*3))
label_placeholder=tf.placeholder(tf.float32,shape=(None,race))
x_image=tf.reshape(images_placeholder,[-1,image_heigh,image_width,3])  # 轉回圖片的size

## 建立網路
# 第1層 Convolution
W1=weight_generate([4,4,3,layer1_node])  # convolution的patch為3*3,輸入3channal(RGB),輸出layer1_node個feature map
b1=bias_generate([layer1_node])
hidden1=conv(x_image,W1)+b1     # x_image用W1的patch做conv,接著再加上b1的偏差
hidden1=tf.nn.relu(hidden1)     # 通過 ReLU激活函數
hidden1=max_pooling(hidden1)    # 通過 Max pooling減少維度

# 第2層 Convolution
W2=weight_generate([4,4,layer1_node,layer2_node])  # convolution的patch為3*3,輸入layer1_node channal,輸出layer2_node個feature map
b2=bias_generate([layer2_node])
hidden2=conv(hidden1,W2)+b2     # hidden1用W2的patch做conv,接著再加上b2的偏差
hidden2=tf.nn.relu(hidden2)     # 通過 ReLU激活函數
hidden2=max_pooling(hidden2)    # 通過 Max pooling減少維度

# 將第2層的輸出拉平(目前有layer2_node張feature map,每張大小為(image_heigh/4)*(image_width/4))
hidden2_flat=tf.reshape(hidden2,[-1,int((image_heigh/4)*(image_width/4)*layer2_node)]) # 拉平

# 第3層 Fully connected
W3=weight_generate([int((image_heigh/4)*(image_width/4)*layer2_node),layer3_node])  # 因為經過2次Max pooling,feature map會是原圖片的1/4倍
b3=bias_generate([layer3_node]) 
hidden3=tf.matmul(hidden2_flat,W3)+b3  # hidden3_flat用W3矩陣相乘,接著再加上b3的偏差
hidden3=tf.nn.relu(hidden3)         # 通過 ReLU激活函數

# 第4層 Fully connected
W4=weight_generate([layer3_node,layer4_node])  
b4=bias_generate([layer4_node]) 
hidden4=tf.matmul(hidden3,W4)+b4  # hidden4用W4矩陣相乘,接著再加上b4的偏差
hidden4=tf.nn.relu(hidden4)         # 通過 ReLU激活函數

# 第5層 Fully connected
W5=weight_generate([layer4_node,output_node])  
b5=bias_generate([output_node]) 
output=tf.matmul(hidden4,W5)+b5          # hidden4用W5矩陣相乘,接著再加上b5的偏差
output=tf.nn.softmax(output)         # 通過 softmax激活函數

# 評估模型
cross_entropy=-tf.reduce_sum(label_placeholder*tf.log(output))
training_method=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 用Adam做參數修正,學習率10^-4,最小化Loss function=Cross entropy
accuracy_judge=tf.equal(tf.argmax(output,1),tf.argmax(label_placeholder,1))  # 輸出的機率最大者是否與label標記者相等
accuracy=tf.reduce_mean(tf.cast(accuracy_judge,'float'))  # 轉為float並且做平均(多筆data)

# 激活模型
Loss_record=[]  # 紀錄Loss

sess.run(tf.global_variables_initializer())      # 激活所有變數
for epoch_times in range(0,epoch_num):       # 要執行多次epoch
    print('epoch times=',epoch_times)
    for batch_times in range(0,int(data_number*race/batch_size)):  # 全部的資料可以分成多少個batch
        get_x,get_y=get_batch(training_data_index,batch_times,training_data)  # 取得一個batch的資料(data與label)
        # 做training
        training_method.run(feed_dict={images_placeholder:get_x,label_placeholder:get_y})
        
        
        
        # 計算正確率(每個batch都會算一次training正確率)
        training_accuracy=accuracy.eval(feed_dict={images_placeholder:get_x,label_placeholder:get_y})
        print('training_accuracy=',training_accuracy)
        #record_train_accuracy.append(training_accuracy)
        
        
        # 計算正確率(testing data的正確率)
        #get_x=testing_data   # 全部的testing data
        #temp_label=testing_data_index[:,1]
        #get_y=np.eye(race)[temp_label.astype(int),:]
        #testing_accuracy=accuracy.eval(feed_dict={images_placeholder:get_x,label_placeholder:get_y})
        #record_test_accuracy.append(testing_accuracy)
        
        # 印出經過此batch後的 training accuracy 和 testing accuracy
        #print('training accuracy=',training_accuracy,', testing accuracy=',testing_accuracy)
        
        
    # 計算正確率(每個epoch都會算一次training正確率)
    #temp_data=training_data_index[:,0]     # 獲得training data的index
    #get_x=training_data[temp_data.astype(int),:]    # 全部的training data
    #temp_label=training_data_index[:,1]
    #get_y=np.eye(race)[temp_label.astype(int),:]
    #training_accuracy=accuracy.eval(feed_dict={images_placeholder:get_x,label_placeholder:get_y})
    #record_train_accuracy.append(training_accuracy)
    
    # 觀察Loss
    #temp_data=training_data_index[:,0]     # 獲得training data的index
    #get_x=training_data[temp_data.astype(int),:]    # 全部的training data
    #temp_label=training_data_index[:,1]
    #get_y=np.eye(race)[temp_label.astype(int),:]
    #Loss=cross_entropy.eval(feed_dict={images_placeholder:get_x,label_placeholder:get_y})
    #Loss=Loss/data_number
    #print('Loss=',Loss)
    #Loss_record.append(Loss)
    
        
    # 計算正確率(epoch的正確率)
    #get_x=testing_data   # 全部的testing data
    #temp_label=testing_data_index[:,1]
    #get_y=np.eye(race)[temp_label.astype(int),:]
    #testing_accuracy=accuracy.eval(feed_dict={images_placeholder:get_x,label_placeholder:get_y})
    #record_test_accuracy.append(testing_accuracy)
        
    # 印出經過此batch後的 training accuracy 和 testing accuracy
    #print('training accuracy=',training_accuracy,', testing accuracy=',testing_accuracy)
   
    # 計算正確率(每個種類的正確率)
    #race_accuracy=[]
    #for file_num in range(0,race):
    #    get_x=testing_data[file_num*data_test_number:(file_num+1)*data_test_number,:]   # 某種類testing data
    #    temp_label=testing_data_index[file_num*data_test_number:(file_num+1)*data_test_number,1]
    #    get_y=np.eye(race)[temp_label.astype(int),:]
    #    testing_accuracy=accuracy.eval(feed_dict={images_placeholder:get_x,label_placeholder:get_y})
    #    race_accuracy.append(testing_accuracy)  
    #print(race_accuracy)
        

    # 每做完1次epoch就做shuffle
    np.random.shuffle(training_data_index)     # shuffle

# 找出辨識錯誤與正確的圖片
correct_and_error_image_index=np.zeros([race,2])  # 每個種族當中挑出分類正確與錯誤的index
for file_num in range(0,race):
    print('Race=',file_num)
    filePath=training_data_path+'//'+file_name[file_num] 
    imgs = os.listdir(filePath)          # 取得該路徑下的所有圖片
    # 找正確的
    print('found right')
    for test_num in range(0,data_test_number):
        get_x=testing_data[file_num*data_test_number+test_num,:]   # 取得該筆data
        get_x=get_x.reshape([-1,10800])
        temp_label=testing_data_index[file_num*data_test_number+test_num,1]  # 其所對應的label
        get_y=np.eye(race)[temp_label.astype(int),:]       # label轉成one-hot vector
        get_y=get_y.reshape([-1,10])

        out=output.eval(feed_dict={images_placeholder:get_x,label_placeholder:get_y})
        #print(out)
        find_label = np.argmax(out)
        #print(find_label)

        if find_label==temp_label:   # 為其所對應的真實label
            correct_and_error_image_index[file_num,0] # 紀錄正確
            imagepath=filePath+'//'+imgs[test_num]
            print('currect_image=',imagepath)
            break
        
    # 找錯誤的
    print('found error')
    for test_num in range(0,data_test_number):
        get_x=testing_data[file_num*data_test_number+test_num,:]   # 取得該筆data
        get_x=get_x.reshape([-1,10800])
        temp_label=testing_data_index[file_num*data_test_number+test_num,1]  # 其所對應的label
        get_y=np.eye(race)[temp_label.astype(int),:]       # label轉成one-hot vector
        get_y=get_y.reshape([-1,10])
        out=output.eval(feed_dict={images_placeholder:get_x,label_placeholder:get_y})
        #print(out)
        find_label = np.argmax(out)
        #print(find_label)

        
        if find_label!=temp_label: # 並非其所對應的真實label
            correct_and_error_image_index[file_num,1] # 紀錄錯誤
            imagepath=filePath+'//'+imgs[test_num]
            print('error_image=',imagepath)
            print('error to',find_label)
            break
        
            
    




    
# 印出結果
#plt.figure(2)
#plt.plot(record_train_accuracy)
#plt.plot(record_test_accuracy)
#plt.xlabel('Number of epoch')
#plt.ylabel('Accuracy')
#plt.show() 

#plt.figure(2)
#plt.plot(Loss_record)
#plt.xlabel('Number of batch')
#plt.ylabel('Cross entropy')
#plt.show() 

    

    


#cv2.imshow('123', k1)
#cv2.waitKey(0)
    
