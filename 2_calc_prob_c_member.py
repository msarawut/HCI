import numpy as np
import xlrd
import time

start_time = time.time()
print("--- %s start time ---" % (start_time)) 

cluster_n=256 # 256, 1024, 4096, 16384 
cluster_fileName='64b8b_t10k' # training set= 64b8b_t10k or query set=64b8b_q 

if __name__=='__main__':  
    
    book = xlrd.open_workbook('./input/SePH_MIR_64b8b.xlsx')
    R_BX = book.sheet_by_name('RBX_dec')
    R_BY = book.sheet_by_name('RBY_dec')
    T_labels = book.sheet_by_name('Train_labels') #  Train_labels or  Q_labels for training label
    R_labels = book.sheet_by_name('R_labels')   
   
    R_BX_data = np.array([[R_BX.cell_value(r, c) for c in range(R_BX.ncols)] for r in range(R_BX.nrows)])
    R_BY_data = np.array([[R_BY.cell_value(r, c) for c in range(R_BY.ncols)] for r in range(R_BY.nrows)])
    
    T_labels_data = np.array([[T_labels.cell_value(r, c) for c in range(T_labels.ncols)] for r in range(T_labels.nrows)])
    R_labels_data = np.array([[R_labels.cell_value(r, c) for c in range(R_labels.ncols)] for r in range(R_labels.nrows)])
        
    num_train = T_labels_data.shape[0]    
    result_prob_BX=np.zeros((num_train,cluster_n),dtype=float)
    result_prob_BY=np.zeros((num_train,cluster_n),dtype=float)
     
    Dot_qL_rL = (np.dot(T_labels_data, R_labels_data.transpose()) > 0).astype(np.integer) 
    groundTruth_countMember=np.zeros((num_train,1),dtype=int)
     
    idxCount_X = np.loadtxt('./output/1_calc_clusterMember/64b8b/clusterCountBX_64b8b.txt', dtype=int) 
    idxCount_Y = np.loadtxt('./output/1_calc_clusterMember/64b8b/clusterCountBY_64b8b.txt', dtype=int) 
    
    count_mem_X=np.array(idxCount_X)
    count_mem_Y=np.array(idxCount_Y)
    for cm in range(count_mem_X.shape[0]): # num_train, Train_no
        if count_mem_X[cm]==0:
            count_mem_X[cm]=1
        if count_mem_Y[cm]==0:
            count_mem_Y[cm]=1
            
    for t in range(num_train): # num_train
        count_gnd = np.asarray(np.where(Dot_qL_rL[t] == 1)) 
        print('t: ', t)
        for sq in range(len(count_gnd[0])):
            temp_BX=R_BX_data[count_gnd[0][sq]].astype(np.integer) 
            result_prob_BX[t][temp_BX]=result_prob_BX[t][temp_BX].astype(np.float32)+ 1 
            temp_BY =R_BY_data[count_gnd[0][sq]].astype(np.integer)            
            result_prob_BY[t][temp_BY]=result_prob_BY[t][temp_BY].astype(np.float32)+ 1 
         
        result_prob_BX[t]=np.divide(result_prob_BX[t],count_mem_X) #normalize
        result_prob_BY[t]=np.divide(result_prob_BY[t],count_mem_Y) #normalize
        
        result_prob_BX[t] =  result_prob_BX[t]/np.sum(result_prob_BX[t])
        result_prob_BY[t] =  result_prob_BY[t]/np.sum(result_prob_BY[t])
    
    f_result_probBX = './output/2_calc_groundTruth_prob/64b8b/probBX_'  + cluster_fileName + '.txt'
    np.savetxt(f_result_probBX, result_prob_BX, delimiter=' ',fmt='%f')
    
    f_result_probBY = './output/2_calc_groundTruth_prob/64b8b/probBY_'  + cluster_fileName + '.txt'
    np.savetxt(f_result_probBY, result_prob_BY, delimiter=' ',fmt='%f')
    
    end_time = time.time()
    time_taken = end_time - start_time # time_taken is in seconds
    hours, rest = divmod(time_taken,3600)
    minutes, seconds = divmod(rest, 60)
    print("time took %d hours %d minutes %f seconds" %(hours,minutes,seconds))
        
    
