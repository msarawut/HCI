from scipy.spatial.distance import cdist
import numpy as np
import xlrd
from time import time
start_time = time()
bits_n='064b' # have to change to bit number
full_bit=64

book = xlrd.open_workbook('./input/SePH_MIR_64b.xlsx')
QL = book.sheet_by_name('Q_labels') # query label
RL = book.sheet_by_name('R_labels') # retrieval label

query_size = 836    
QX = book.sheet_by_name('Q_BX') #change query Q_BX or Q_BY    
RX = book.sheet_by_name('R_BY') # change dataset  R_BX and R_BY 
    
qX_data = np.array([[QX.cell_value(r, c) for c in range(QX.ncols)] for r in range(QX.nrows)])       
rX_data = np.array([[RX.cell_value(r, c) for c in range(RX.ncols)] for r in range(RX.nrows)])
qL_data = np.array([[QL.cell_value(r, c) for c in range(QL.ncols)] for r in range(QL.nrows)])       
rL_data = np.array([[RL.cell_value(r, c) for c in range(RL.ncols)] for r in range(RL.nrows)]) 
  
num_query = qX_data.shape[0]
num_retrieval = rX_data.shape[0]
    
q_X=np.zeros((num_query,full_bit),dtype=int) # query size
r_X=np.zeros((num_retrieval,full_bit),dtype=int) # database size  
for c in range(num_query): # query size
    qx=np.array(list(qX_data[c][0]))         
    q_X[c]=qx
    
for c in range(num_retrieval): # database size 
    rx=np.array(list(rX_data[c][0]))      
    r_X[c]=rx
        

def mAP(g_candidate_size): ## compute mAP # compute mAP
    tmap=0
    for iter in range(query_size):  # query_size -> all_query   
        tsum = np.sum(gnd[iter])
        #print('tsum: ', tsum)
        count_truth=0    
        sum_ap=0
        for can_i in range(g_candidate_size): # all_candidate        
            ap=0       
            if gnd[iter][ind_hamming[iter][can_i]]==1: # if candidate = answer in ground-truth
                count_truth+=1 # index of ground-truth                               
                ap=count_truth/(can_i+1) # compute AP of each answer (can_i+1 index of candidate)
                sum_ap=sum_ap+ap # sum AP for each query
        if tsum>g_candidate_size:
            tsum=g_candidate_size
        sum_ap=sum_ap/tsum # if ground-truth more than candidate divide by candidate
        tmap=tmap+sum_ap
    tmap=tmap/query_size   
    return tmap

MAP50_start_time = time()
        
gnd = (np.dot(qL_data, rL_data.transpose()) > 0).astype(np.integer) # check ground-truth
hamming = cdist(q_X, r_X, 'hamming')
ind_hamming = np.argsort(hamming) #+1

print('candidate_size: 50')
print('map: ', mAP(50))

end_time = time()
time_taken = end_time - MAP50_start_time # time_taken is in seconds
hours, rest = divmod(time_taken,3600)
minutes, seconds = divmod(rest, 60)  
print("MAP@50 %d hours %d minutes %f seconds" %(hours,minutes,seconds)) 


# compute mAP    
print('candidate_size: 1')
print('map: ', mAP(1))
print('candidate_size: 5')
print('map: ', mAP(5))
print('candidate_size: 10')
print('map: ', mAP(10))
print('candidate_size: 15')
print('map: ', mAP(15))
print('candidate_size: 20')
print('map: ', mAP(20))
print('candidate_size: 25')
print('map: ', mAP(25))
print('candidate_size: 30')
print('map: ', mAP(30))
print('candidate_size: 35')
print('map: ', mAP(35))
print('candidate_size: 40')
print('map: ', mAP(40))
print('candidate_size: 45')
print('map: ', mAP(45))
print('candidate_size: 50')
print('map: ', mAP(50))

  
end_time = time()
time_taken = end_time - start_time # time_taken is in seconds
hours, rest = divmod(time_taken,3600)
minutes, seconds = divmod(rest, 60)  
print("This took %d hours %d minutes %d seconds" %(hours,minutes,seconds)) 
           
       