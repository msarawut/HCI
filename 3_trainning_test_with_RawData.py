from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import xlrd

from keras.models import Sequential
from keras.layers import Dense, Activation

from time import time
start_time = time()

# prepare training data
OutputUnit = 256 # 256, 1024, 4096, 16384 cluster number
BatchSize = 128
Epochs = 100 # 100, 200, 300
cluster_fileName='BY_RX_64b8b_t10k_32_64_32' #### write output name

book = xlrd.open_workbook('./input/SePH_MIR_query_raw.xlsx')
query_train = book.sheet_by_name('tQY_raw') # training query
query_test = book.sheet_by_name('QBY_raw') # testing query

prob_train = np.loadtxt('./output/2_calc_groundTruth_prob/64b8b/probBX_64b8b_t10k.txt') # prob training -> RXY
prob_test = np.loadtxt('./output/2_calc_groundTruth_prob/64b8b/probBX_64b8b_q.txt') # prob query-> RXY

y_train = np.array(prob_train) 
y_test = np.array(prob_test)

x_train = np.array([[query_train.cell_value(r, c) for c in range(query_train.ncols)] for r in range(query_train.nrows)])
x_test = np.array([[query_test.cell_value(r, c) for c in range(query_test.ncols)] for r in range(query_test.nrows)])

##### network definition
model = Sequential()
model.add(Dense(32, input_dim = x_train.shape[1], kernel_initializer='he_uniform'))
model.add(Activation('relu'))

model.add(Dense(64, kernel_initializer='he_uniform'))
model.add(Activation('relu'))

model.add(Dense(32, kernel_initializer='he_uniform'))
model.add(Activation('relu'))

model.add(Dense(OutputUnit, kernel_initializer='he_uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=Epochs, batch_size=BatchSize)

# testing
start_predict_time = time()
p_test = model.predict(x_test)
end_predict_time = time()
### sorting
start_sort_time = time()
p_idx = np.argsort(-p_test)
clusterB_all=p_idx
end_sort_time = time()

################
predict_time_taken = end_predict_time - start_predict_time # time_taken is in seconds
hours, rest = divmod(predict_time_taken,3600)
minutes, seconds = divmod(rest, 60)
print("Predict time took %d hours %d minutes %f seconds" %(hours,minutes,seconds)) 

sort_time_taken = end_sort_time - start_sort_time # time_taken is in seconds
hours, rest = divmod(sort_time_taken,3600)
minutes, seconds = divmod(rest, 60)
print("Sort time took %d hours %d minutes %f seconds" %(hours,minutes,seconds)) 

f_clusterB_all = './measure/learned_'  + cluster_fileName  + '.txt'
np.savetxt(f_clusterB_all, clusterB_all, delimiter=' ',fmt='%d') 

end_time = time()
all_time_taken = end_time - start_time # time_taken is in seconds
hours, rest = divmod(all_time_taken,3600)
minutes, seconds = divmod(rest, 60)
print("All time took %d hours %d minutes %f seconds" %(hours,minutes,seconds)) 
