import numpy as np
from scipy.spatial.distance import cdist # for compute hamming distance
import xlrd
from time import time
start_time = time()

full_bit=64 # 32, 16
bits_n='064b' # 032b, 016b have to change to bit number

c_amount=50  ##### how many first cluster we want 

def hamming_distance(q, r): # input matrix (query, dataset)
    h = cdist(q, r, 'hamming')
    return h

def reorder(q, r, clusterMember): # input matrix (query, clusterMember)   
    g_r = np.array(r)
    g_clusterMember = np.array(clusterMember)
    num_clusterMember = g_clusterMember.shape[0]   
    m_q=np.zeros((1,full_bit),dtype=int)
    c_member_b=np.zeros((num_clusterMember,full_bit),dtype=int)
    
    m_q[0]=q
    for i in range(num_clusterMember): # get data full bit
        c_member_b[i]=g_r[g_clusterMember[i]]
    Hamming_matrix=hamming_distance(m_q,c_member_b)    
    m_idx=np.argsort(Hamming_matrix)
    g_clusterMember=g_clusterMember[m_idx]
    return g_clusterMember

def mAP_cluster(q_i, g_candidate_size,g_candidate_member,g_gnd): ## compute mAP 
    tmap=0
    tsum = np.sum(g_gnd[q_i])
    count_truth=0
    sum_ap=0
    for can_i in range(g_candidate_size): # all_candidate 
        ap=0          
        if g_gnd[q_i][g_candidate_member[0][can_i]]==1: # if candidate = answer in ground-truth
            count_truth+=1 # index of ground-truth                               
            ap=count_truth/(can_i+1) # compute AP of each answer (can_i+1 index of candidate)
            sum_ap=sum_ap+ap # sum AP for each query
    if tsum>g_candidate_size:
        tsum=g_candidate_size      
    sum_ap=sum_ap/tsum # if ground-truth more than candidate divide by candidate
    tmap=tmap+sum_ap               
    return tmap
   
if __name__=='__main__':
    
    book = xlrd.open_workbook('./input/SePH_MIR_64b.xlsx')      
    QB = book.sheet_by_name('Q_BX') #change query Q_BX or Q_BY    
    RB = book.sheet_by_name('R_BY') # change dataset  R_BX or R_BY
    
    QB_data = np.array([[QB.cell_value(r, c) for c in range(QB.ncols)] for r in range(QB.nrows)])       
    RB_data = np.array([[RB.cell_value(r, c) for c in range(RB.ncols)] for r in range(RB.nrows)])
    QL = book.sheet_by_name('Q_labels') # query label
    RL = book.sheet_by_name('R_labels') # retrieval label
    qL_data = np.array([[QL.cell_value(r, c) for c in range(QL.ncols)] for r in range(QL.nrows)])       
    rL_data = np.array([[RL.cell_value(r, c) for c in range(RL.ncols)] for r in range(RL.nrows)])
    gnd = (np.dot(qL_data, rL_data.transpose()) > 0).astype(np.float32) # check ground-truth
    
    Input_NN_bucket = np.loadtxt('./measure/learned_BX_RY_64b8b_t10k_32_64_32.txt', dtype=int) # query -> BXY
    idxCount = np.loadtxt('./output/1_calc_clusterMember/64b8b/clusterCountBY_64b8b.txt', dtype=int) # dataset-> RXY
    npzfile = np.load('./output/1_calc_clusterMember/64b8b/clusterMemberBY_64b8b.npz') # dataset -> RXY    
    Ints_clustMem = np.array(npzfile['cluster_menber'])
    
    num_query = QB_data.shape[0]
    num_retrieval = RB_data.shape[0]
      
    q_X=np.zeros((num_query,full_bit),dtype=int) # query size
    r_X=np.zeros((num_retrieval,full_bit),dtype=int) # database size  
    for c in range(num_query): # query size
        temp_Q=str(QB_data[c][0])       
        temp_Q=temp_Q.replace(' ', '')
        qx=np.array(list(temp_Q))         
        q_X[c]=qx
    
    for c in range(num_retrieval): # database size 
        temp_R=str(RB_data[c][0])       
        temp_R=temp_R.replace(' ', '')        
        rx=np.array(list(temp_R))      
        r_X[c]=rx
    
    print('\nLoading input files completed')    
    print('First amount cluster= ', c_amount)
    
    mAP_no_rank = np.zeros((num_query,2), dtype=float)
    mAP_rank = np.zeros((num_query,2), dtype=float)
    
    sum_mAP_no_rank=0
    sum_mAP_rank=0
    sum_c_n=0
    #pick candidate    
    for i in range(num_query): #  how many query     
        count_candidate=0         
        c_n=0
        for c in range(c_amount) :
            c_n=c_n+idxCount[Input_NN_bucket[i,c]]            
        c_m = np.zeros((1,c_n), dtype=int)        
       
        for j in range(c_amount) :        
            if idxCount[Input_NN_bucket[i,j]] > 0:
                dataMember_inBucket=Ints_clustMem[Input_NN_bucket[i,j]]                
                for c in range(len(dataMember_inBucket)):                    
                    c_m[0][count_candidate]=dataMember_inBucket[c]                    
                    count_candidate+=1
                    
        mAP_no_rank[i][0]=mAP_cluster(i,c_n,c_m,gnd) ### no-ranking  
        mAP_no_rank[i][1]=c_n
                
        ####### reorder               
        dataMember_re=reorder(q_X[i], r_X, c_m[0]) ### re-ranking
        mAP_rank[i][0]=mAP_cluster(i,c_n,dataMember_re,gnd)  
        mAP_rank[i][1]=c_n
        
        sum_mAP_no_rank=sum_mAP_no_rank+mAP_no_rank[i][0]
        sum_mAP_rank=sum_mAP_rank+mAP_rank[i][0]
        sum_c_n=sum_c_n+c_n
                     
    print('Avg mAP no rank= ', sum_mAP_no_rank/num_query)
    print('Avg mAP rank= ',sum_mAP_rank/num_query)
    print('Avg candidate= ', sum_c_n/num_query)
    
    end_time = time()
    time_taken = end_time - start_time # time_taken is in seconds
    hours, rest = divmod(time_taken,3600)
    minutes, seconds = divmod(rest, 60)  
    print("This took %d hours %d minutes %d seconds" %(hours,minutes,seconds)) 
   