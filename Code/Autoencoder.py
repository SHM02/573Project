import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
import tensorflow.python.debug as tf_debug
ratings = pd.read_csv('C:/Users/aishwaryamohan/Desktop/Ripple/RippleNet/data/movie/ratings.dat', sep="::", header = None, engine='python')
ratings_pivot = pd.pivot_table(ratings[[0,1,2]],
          values=2, index=0, columns=1 ).fillna(0)
X_train, X_test = train_test_split(ratings_pivot, train_size=0.8)
# Initialise node counts of layers
nc_ip = 3706  
nc_h1  = 2048
nc_h2  = 1024
nc_h3  = 700
nc_h4  = 700
nc_h5  = 1024
nc_h6  = 2048 
nc_op = 3706  

h1_vals = {'weights':tf.Variable(tf.random_normal([nc_ip,nc_h1])),'biases':tf.Variable(tf.random_normal([nc_h1]))  }
h2_vals = {'weights':tf.Variable(tf.random_normal([nc_h1, nc_h2])),'biases':tf.Variable(tf.random_normal([nc_h2]))  }
h3_vals = {'weights':tf.Variable(tf.random_normal([nc_h2, nc_h3])),'biases':tf.Variable(tf.random_normal([nc_h3]))  }
h4_vals = {'weights':tf.Variable(tf.random_normal([nc_h3, nc_h4])),'biases':tf.Variable(tf.random_normal([nc_h4]))  }
h5_vals = {'weights':tf.Variable(tf.random_normal([nc_h4, nc_h5])),'biases':tf.Variable(tf.random_normal([nc_h5]))  }
h6_vals = {'weights':tf.Variable(tf.random_normal([nc_h5, nc_h6])),'biases':tf.Variable(tf.random_normal([nc_h6]))  }

op_layer_values = {
'weights':tf.Variable(tf.random_normal([nc_h6,nc_op])),'biases':tf.Variable(tf.random_normal([nc_op])) }

input_layer = tf.placeholder('float', [None, 3706])
input_layer_const = tf.fill( [tf.shape(input_layer)[0], 1] ,1.0  )
input_layer_concat =  tf.concat([input_layer, input_layer_const], 1)
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer,h1_vals['weights']), h1_vals['biases']))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,h2_vals['weights']),h2_vals['biases']))
layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2,h3_vals['weights']),h3_vals['biases']))
layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3,h4_vals['weights']),h4_vals['biases']))
layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4,h5_vals['weights']),h5_vals['biases']))
layer_6 = tf.nn.sigmoid(tf.add(tf.matmul(layer_5,h6_vals['weights']),h6_vals['biases']))
output_layer = tf.matmul(layer_6,op_layer_values['weights'])  + op_layer_values['biases']
# true_output retains the original shape for eror calculation
true_output = tf.placeholder(tf.float32, [None, 3706])
# Use only non zero ratings in output_true for loss computation
mask=tf.where(tf.equal(true_output,0.0), tf.zeros_like(true_output), true_output) # zero value indices in the training set (no ratings)
num_labels=tf.cast(tf.count_nonzero(mask),dtype=tf.float32) # non zero value count in the train set
bool_mask=tf.cast(mask,dtype=bool) 
output_layer1=tf.where(bool_mask, output_layer, tf.zeros_like(output_layer))
print((true_output))
print(output_layer)
print(input_layer)
meansqr = tf.div(tf.reduce_sum(tf.square(tf.subtract(output_layer1,true_output))),num_labels)
learning_rate = 0.01   # model learning rate
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(meansqr)

init =tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size = 100
num_epochs = 20
tot_usr_cnt = X_train.shape[0]


import matplotlib.pyplot as plt1
ae_train=[]
ae_test=[]
ae_loss=[]
# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0    # initialize error to 0
    
    for i in range(int(tot_usr_cnt/batch_size)):
        epoch_x = X_train[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, meansqr], feed_dict={input_layer: epoch_x,true_output: epoch_x})
        epoch_loss += c
        
    output_train = sess.run(output_layer,               feed_dict={input_layer:X_train})
    output_test = sess.run(output_layer,                   feed_dict={input_layer:X_test})
        
    print('MSE train', MSE(output_train, X_train),'MSE test', MSE(output_test, X_test))
    print('Epoch', epoch, '/', num_epochs, 'loss:',epoch_loss)
    
    ae_train.append((MSE(output_train, X_train)))
    ae_test.append((MSE(output_test, X_test)))
    ae_loss.append(epoch_loss)
    num_points = len(ae_train)
    x_axis = []
    for i in range(0,num_points):
        x_axis.append(i)
print(num_points)
    
print((ae_train))
print(ae_test)
print(ae_loss)


import sys
import operator
#  Display results
num_user_ub = 20
users = ((X_test.iloc[:20, :0]).index.tolist())
for i in range(0, num_user_ub):
    sample_user = X_test.iloc[i, :]
    print("USER ID -", users[i])

    print("ORIGINAL RATING:")
    print(list(sample_user[:20]))
    # get the predicted ratings
    sample_user_pred = sess.run(output_layer, feed_dict={input_layer: [sample_user]})
    np.set_printoptions(threshold=np.inf)
    my_list = (sample_user_pred.tolist()[0])
    prep = ['%.2f' % elem for elem in my_list]
    print("PREDICTED RATINGS:")
    print(prep[:20])
    dic = {}
    for i in range(len(sample_user_pred[0])):
        dic[i] = prep[i]

    sorted_dic = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)

    sample_user_list = list(sample_user)
    result_list = []
    for j in range(0, len(sorted_dic)):
        if (len(result_list) == 50): #prev 10
            break
        if (sample_user_list[sorted_dic[j][0]] == 0):
            result_list.append(sorted_dic[j])
    print("TOP-10 RECOMMENDED MOVIES")
    print(result_list)
