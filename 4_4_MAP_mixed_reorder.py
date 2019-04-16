# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:24:27 2018
@author: wut-pc
"""
import numpy as np
from scipy.spatial.distance import cdist # for compute hamming distance
import xlrd
from time import time
start_time = time()

query_size= 836 
full_bit=64 # 32, 16
bits_n='064b' # 032b, 016b have to change to bit number

Top_k=50 ### MAP@R change 1-50

t_c=0 # no need to change
pr=np.zeros((1,5),dtype=float)

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
    
    if num_clusterMember > Top_k and Top_k > 0:        
        top_dataMember=np.array(g_clusterMember[0,0:Top_k])
    else:
        top_dataMember=np.array(g_clusterMember[0,0:num_clusterMember])
    
    return top_dataMember

def c_pr_mAP(query_n,g_candidate_member,g_gnd): ## compute mAP 
    tp=0
    tr=0
    tsum = np.sum(g_gnd[query_n])         
    count_truth=0
    sum_ap=0
    tmap=0
    count_mem=g_candidate_member.shape[0]
    
    for can_i in range(count_mem): # all_candidate                       
        if g_gnd[query_n][g_candidate_member[can_i]]==1: # if candidate = answer in ground-truth
            count_truth+=1 # count correct answer 
            ap=count_truth/(can_i+1) # compute AP of each answer (can_i+1 index of candidate)
            sum_ap=sum_ap+ap # sum AP for each query
            
    tp=count_truth/count_mem # all correct answer divide by candidate size
    tr=count_truth/tsum 
    
    if tsum<=count_mem:
            tsum=tsum
    elif tsum>count_mem:
            tsum=count_mem      
    sum_ap=sum_ap/tsum # if ground-truth more than candidate divide by candidate
    tmap=tmap+sum_ap
    
    return tp,tr,tmap
   
if __name__=='__main__':
         
    book = xlrd.open_workbook('./input/SePH_MIR_64b.xlsx')        
    QB = book.sheet_by_name('Q_BX') #change query Q_BX or Q_BY    
    RB = book.sheet_by_name('R_BY') # change dataset, R_BX or R_BY 
    
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
    start_time = time()
    
    for q in range(query_size): # how many query        
        count_candidate=0
        Is_full=0  
        merge_dataMember=""
        tmp_merge_dataMember=""
        for j in range(Input_NN_bucket.shape[1]) : # how many candidate bucket
           
            if Is_full==1:
                break            
            elif idxCount[Input_NN_bucket[q,j]] > 0: #and retrievedBuckets < CandidateInfo[2] : 
                 
                if len(tmp_merge_dataMember) == 0:
                    merge_dataMember=Ints_clustMem[Input_NN_bucket[q,j]]                                                                   
                else:          
                    merge_dataMember= np.concatenate((tmp_merge_dataMember,Ints_clustMem[Input_NN_bucket[q,j]]))
                tmp_merge_dataMember=merge_dataMember
                
                if count_candidate>=Top_k :
                    Is_full=1 
                elif idxCount[Input_NN_bucket[q,j]] >=Top_k:
                    Is_full=1
                   
                count_candidate=count_candidate + idxCount[Input_NN_bucket[q,j]]
                                
            
        if len(tmp_merge_dataMember) == 0:
            tp=0
            tr=0
            mAP=0
        else:
            dataMember_re=reorder(q_X[q], r_X, tmp_merge_dataMember)  #### ranking
            tp,tr,mAP=  c_pr_mAP(q,dataMember_re,gnd)   #### ranking
            #tp,tr,mAP=  c_pr_mAP(q,tmp_merge_dataMember,gnd)   #### no-ranking
            
        pr[t_c][0]=t_c+1
        pr[t_c][1]=pr[t_c][1]+tp
        pr[t_c][2]=pr[t_c][2]+tr
        pr[t_c][3]=pr[t_c][3]+mAP
        pr[t_c][4]=pr[t_c][4]+count_candidate
    pr[t_c][1]=pr[t_c][1]/query_size
    pr[t_c][2]=pr[t_c][2]/query_size
    pr[t_c][3]=pr[t_c][3]/query_size
    pr[t_c][4]=pr[t_c][4]/query_size 
    
    print('Top_k: ', Top_k)
    print('Precision: ', pr[t_c][1])
    print('Recall: ', pr[t_c][2])
    print('mAP: ', pr[t_c][3])
    print('candidate: ', pr[t_c][4])
 
    end_time = time()
    time_taken = end_time - start_time # time_taken is in seconds
    hours, rest = divmod(time_taken,3600)
    minutes, seconds = divmod(rest, 60)  
    print("This took %d hours %d minutes %f seconds" %(hours,minutes,seconds)) 
   