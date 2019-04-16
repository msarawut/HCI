import numpy as np
from scipy.spatial.distance import cdist # for compute hamming distance
import xlrd
from time import time
start_time = time()

query_size=836 
full_bit=64 # 32, 16 -> original file -> full file
bits_n='064b' # 032b, 016b have to change to bit number

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

def mAP(g_query_size, g_candidate_size,g_candidate_member,g_gnd): ## compute mAP 
    tmap=0
    for iter in range(g_query_size):  # query_size -> all_query                
        tsum = np.sum(g_gnd[iter])
        count_truth=0
        sum_ap=0
        for can_i in range(g_candidate_size): # all_candidate 
            ap=0          
            if g_gnd[iter][g_candidate_member[iter][can_i]]==1: # if candidate = answer in ground-truth
                count_truth+=1 # index of ground-truth                               
                ap=count_truth/(can_i+1) # compute AP of each answer (can_i+1 index of candidate)
                sum_ap=sum_ap+ap # sum AP for each query
        if tsum>g_candidate_size:
            tsum=g_candidate_size      
        sum_ap=sum_ap/tsum # if ground-truth more than candidate divide by candidate
        tmap=tmap+sum_ap               
    tmap=tmap/g_query_size    
    
    return tmap
   
if __name__=='__main__':

    candidate_size1= 1  
    candidate_size5= 5 
    candidate_size10= 10
    candidate_size15= 15 
    candidate_size20= 20 
    candidate_size25= 25 
    candidate_size30= 30 
    candidate_size35= 35 
    candidate_size40= 40 
    candidate_size45= 45 
    candidate_size50= 50 
   
    
    candidate_member1 = np.zeros((query_size,candidate_size1), dtype=int)
    candidate_member5 = np.zeros((query_size,candidate_size5), dtype=int)
    candidate_member10 = np.zeros((query_size,candidate_size10), dtype=int)
    candidate_member15 = np.zeros((query_size,candidate_size15), dtype=int)
    candidate_member20 = np.zeros((query_size,candidate_size20), dtype=int)
    candidate_member25 = np.zeros((query_size,candidate_size25), dtype=int)
    candidate_member30 = np.zeros((query_size,candidate_size30), dtype=int)
    candidate_member35 = np.zeros((query_size,candidate_size35), dtype=int)
    candidate_member40 = np.zeros((query_size,candidate_size40), dtype=int)
    candidate_member45 = np.zeros((query_size,candidate_size45), dtype=int)
    candidate_member50 = np.zeros((query_size,candidate_size50), dtype=int)
    
    
    book = xlrd.open_workbook('./input/SePH_MIR_64b.xlsx') #### full bit   
    QB = book.sheet_by_name('Q_BX') #change query Q_BX or Q_BY    
    RB = book.sheet_by_name('R_BY') # change database  
    
    QB_data = np.array([[QB.cell_value(r, c) for c in range(QB.ncols)] for r in range(QB.nrows)])       
    RB_data = np.array([[RB.cell_value(r, c) for c in range(RB.ncols)] for r in range(RB.nrows)])
    QL = book.sheet_by_name('Q_labels') # query label
    RL = book.sheet_by_name('R_labels') # retrieval label
    qL_data = np.array([[QL.cell_value(r, c) for c in range(QL.ncols)] for r in range(QL.nrows)])       
    rL_data = np.array([[RL.cell_value(r, c) for c in range(RL.ncols)] for r in range(RL.nrows)])
    gnd = (np.dot(qL_data, rL_data.transpose()) > 0).astype(np.integer) # check ground-truth
    
    Input_NN_bucket = np.loadtxt('./measure/learned_BX_RY_64b8b_t10k_32_64_32.txt', dtype=int) # query -> BXY
    idxCount = np.loadtxt('./output/1_calc_clusterMember/64b8b/clusterCountBY_64b8b.txt', dtype=int) # database-> RXY
    npzfile = np.load('./output/1_calc_clusterMember/64b8b/clusterMemberBY_64b8b.npz') # database -> RXY    
    Ints_clustMem = np.array(npzfile['cluster_menber'])
    
    num_query = QB_data.shape[0]
    num_retrieval = RB_data.shape[0]
        
    q_X=np.zeros((num_query,full_bit),dtype=int) # query size
    r_X=np.zeros((num_retrieval,full_bit),dtype=int) # database size  
    for c in range(num_query): # query size
        qx=np.array(list(QB_data[c][0]))         
        q_X[c]=qx
    
    for c in range(num_retrieval): # database size 
        rx=np.array(list(RB_data[c][0]))      
        r_X[c]=rx
      
    print('\nLoading input files completed')
    
    #pick candidate
    for i in range(query_size): #  how many query
       
        count_candidate=0
        Is_full=0 
       
        for j in range(Input_NN_bucket.shape[1]) : # how many candidate bucket
            
            if idxCount[Input_NN_bucket[i,j]] > 0: 
                if Is_full==1:
                    break
                dataMember_inBucket=Ints_clustMem[Input_NN_bucket[i,j]]
                # reorder               
                dataMember_re=reorder(q_X[i], r_X, dataMember_inBucket)              
                for c in range(len(dataMember_inBucket)):
                    
                    if count_candidate < candidate_size1:
                         candidate_member1[i][count_candidate]=dataMember_re[0][c]  
                    if count_candidate < candidate_size5:
                         candidate_member5[i][count_candidate]=dataMember_re[0][c]  
                    if count_candidate < candidate_size10:
                         candidate_member10[i][count_candidate]=dataMember_re[0][c]  
                    if count_candidate < candidate_size15:
                         candidate_member15[i][count_candidate]=dataMember_re[0][c]  
                    if count_candidate < candidate_size20:
                         candidate_member20[i][count_candidate]=dataMember_re[0][c]
                    if count_candidate < candidate_size25:
                         candidate_member25[i][count_candidate]=dataMember_re[0][c]
                    if count_candidate < candidate_size30:
                         candidate_member30[i][count_candidate]=dataMember_re[0][c]
                    if count_candidate < candidate_size35:
                         candidate_member35[i][count_candidate]=dataMember_re[0][c]
                    if count_candidate < candidate_size40:
                         candidate_member40[i][count_candidate]=dataMember_re[0][c]
                    if count_candidate < candidate_size45:
                         candidate_member45[i][count_candidate]=dataMember_re[0][c]
                    if count_candidate < candidate_size50:
                         candidate_member50[i][count_candidate]=dataMember_re[0][c]
                        
                    count_candidate+=1
                                     
                if count_candidate >= candidate_size50:
                    Is_full=1
                elif  len(dataMember_inBucket) >= candidate_size50:
                    Is_full=1             
    
    print('\ncollect data completed')
    # compute mAP    
    mAP1=mAP(query_size,candidate_size1,candidate_member1,gnd)
    print('candidate_size: 1')
    print('map: ', mAP1)
    mAP5=mAP(query_size,candidate_size5,candidate_member5,gnd)
    print('candidate_size: 5')
    print('map: ', mAP5)
    mAP10=mAP(query_size,candidate_size10,candidate_member10,gnd)
    print('candidate_size: 10')
    print('map: ', mAP10)
    mAP15=mAP(query_size,candidate_size15,candidate_member15,gnd)
    print('candidate_size: 15')
    print('map: ', mAP15)
    mAP20=mAP(query_size,candidate_size20,candidate_member20,gnd)
    print('candidate_size: 20')
    print('map: ', mAP20)
    mAP25=mAP(query_size,candidate_size25,candidate_member25,gnd)
    print('candidate_size: 25')
    print('map: ', mAP25)
    mAP30=mAP(query_size,candidate_size30,candidate_member30,gnd)
    print('candidate_size: 30')
    print('map: ', mAP30)
    mAP35=mAP(query_size,candidate_size35,candidate_member35,gnd)
    print('candidate_size: 35')
    print('map: ', mAP35)
    mAP40=mAP(query_size,candidate_size40,candidate_member40,gnd)
    print('candidate_size: 40')
    print('map: ', mAP40)
    mAP45=mAP(query_size,candidate_size45,candidate_member45,gnd)
    print('candidate_size: 45')
    print('map: ', mAP45)
    mAP50=mAP(query_size,candidate_size50,candidate_member50,gnd)
    print('candidate_size: 50')
    print('map: ', mAP50)
            
    end_time = time()
    time_taken = end_time - start_time # time_taken is in seconds
    hours, rest = divmod(time_taken,3600)
    minutes, seconds = divmod(rest, 60)  
    print("This took %d hours %d minutes %d seconds" %(hours,minutes,seconds)) 
   