import numpy as np
import xlrd
import time
start_time = time.time()
print("--- %s start time ---" % (start_time)) 
cluster_n=256 # 256, 1024, 4096, 16384
## select bit
start_p=2 # start at 2
end_p=10 # if we want 10 type 12
cluster_fileName='64b8b'

if __name__=='__main__':  
    
    book = xlrd.open_workbook('./input/SePH_MIR_64b8b.xlsx')   
    R_BX = book.sheet_by_name('R_BX')
    R_BY = book.sheet_by_name('R_BY')   
    R_BX_data = np.array([[R_BX.cell_value(r, c) for c in range(R_BX.ncols)] for r in range(R_BX.nrows)])
    R_BY_data = np.array([[R_BY.cell_value(r, c) for c in range(R_BY.ncols)] for r in range(R_BY.nrows)])   
    
    result_clusterCountBX=np.zeros((cluster_n,1),dtype=int)
    result_clusterCountBY=np.zeros((cluster_n,1),dtype=int)
    num_retrieval = R_BX_data.shape[0]
    
    result_DecBX=np.zeros((num_retrieval),dtype=int)
    result_DecBY=np.zeros((num_retrieval),dtype=int)   
            
    for iter in range(num_retrieval):     
        print('iter: ', iter)      
        temp_R_BX=str(R_BX_data[iter])  
        temp_R_BY=str(R_BY_data[iter]) 
        temp_R_BX=temp_R_BX.replace(' ', '')
        temp_R_BY=temp_R_BY.replace(' ', '')     
        
               
        DecBX=int(temp_R_BX[start_p:end_p], base=2) 
        DecBY=int(temp_R_BY[start_p:end_p], base=2)
        result_DecBX[iter]=DecBX
        result_DecBY[iter]=DecBY
        
        result_clusterCountBX[DecBX]=result_clusterCountBX[DecBX].astype(np.integer)+1
        result_clusterCountBY[DecBY]=result_clusterCountBY[DecBY].astype(np.integer)+1  
    
    with open('./output/1_calc_clusterMember/64b8b/clusterMemberBX_'  + cluster_fileName + '.txt', 'a') as fx:
        for c in range(cluster_n):            
            for r in range(num_retrieval):
                if result_DecBX[r]==c:            
                    fx.write(str(r) + ' ') 
            fx.write('\n')
        fx.close()
            
    with open('./output/1_calc_clusterMember/64b8b/clusterMemberBY_'  + cluster_fileName + '.txt', 'a') as fy:
        for c in range(cluster_n):
            for r in range(num_retrieval):                
                if result_DecBY[r]==c:            
                    fy.write(str(r) + ' ')                    
            fy.write('\n')           
        fy.close()       
        
    # write cluster BX and BY      
    Idxcluster_menberBX = []
    fBX = open('./output/1_calc_clusterMember/64b8b/clusterMemberBX_' + cluster_fileName + '.txt', 'r')
    for l in fBX.readlines():
        l = l.split()
        l=np.array(list(l)).astype(np.integer)  
        Idxcluster_menberBX += [l]
    fBX.close()
    np.savez('./output/1_calc_clusterMember/64b8b/clusterMemberBX_' + cluster_fileName , cluster_menber=Idxcluster_menberBX)
    
    Idxcluster_menberBY = []
    fBY = open('./output/1_calc_clusterMember/64b8b/clusterMemberBY_'  + cluster_fileName + '.txt', 'r')
    for l in fBY.readlines():
        l = l.split()
        l=np.array(list(l)).astype(np.integer)
        Idxcluster_menberBY += [l]
    fBY.close()
    np.savez('./output/1_calc_clusterMember/64b8b/clusterMemberBY_'  + cluster_fileName , cluster_menber=Idxcluster_menberBY)         

    f_result_clusterCountBX = './output/1_calc_clusterMember/64b8b/clusterCountBX_'  + cluster_fileName  + '.txt'
    np.savetxt(f_result_clusterCountBX, result_clusterCountBX, delimiter=' ',fmt='%d')
    
    f_result_clusterCountBY = './output/1_calc_clusterMember/64b8b/clusterCountBY_'  + cluster_fileName  + '.txt'
    np.savetxt(f_result_clusterCountBY, result_clusterCountBY, delimiter=' ',fmt='%d') 
       
    end_time = time.time()
    time_taken = end_time - start_time # time_taken is in seconds
    hours, rest = divmod(time_taken,3600)
    minutes, seconds = divmod(rest, 60)
    print("time took %d hours %d minutes %f seconds" %(hours,minutes,seconds))
        
    
