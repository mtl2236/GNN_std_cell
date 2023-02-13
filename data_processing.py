import sys
import os
import string
import numpy as np
#get all the subfolder names at base path
Base_path='/simulation/tianliang/Github/GNN_std_cell/original_data'
subfolderlist=[]
Cell_list=[]

#Success_fail_list=[]
seven_dimensional_list=[]
files = os.listdir(Base_path)

def get_file_list(folder, file_type_list):
    filelist = []  
    for dirpath,dirnames,filenames in os.walk(folder):
        for file in filenames:
            file_type = file.split('.')[-1]
            if(file_type in file_type_list):
                file_fullname = os.path.join(dirpath, file) #full name of files
                filelist.append(file_fullname)
    return filelist

def Calculate_PDP(path):
    file_type_list = ['lib'] 
    filelist=get_file_list(path+'/lib',file_type_list)
    #print(filelist)
    f=open(filelist[0],'r')
    Original_Lib=f.readlines()
    Cell_Begin_Str='cell '
    Cell_Begin_Line_Num=[]
    for i in range(len(Original_Lib)):
        if Cell_Begin_Str in Original_Lib[i]:
            Cell_Begin_Line_Num.append(i)
    Cell_Begin_Line_Num.append(len(Original_Lib)-1)
    #print(Cell_Begin_Line_Num)
    Cell_end_Line_Num=[]
    for j in range(len(Cell_Begin_Line_Num)-2):
        Cell_end_Line_Num.append(Cell_Begin_Line_Num[j+1]-1)
    Cell_end_Line_Num.append(Cell_Begin_Line_Num[j+2])
        #print(Cell_end_Line_Num)
    Cell_Begin_Line_Num.pop()  #delete the last line num
        #Extract cell name in every begin line
        # line_example=[]
        # line_example=Original_Lib[95].split(' ')
        # print(line_example)
    Cell_Name=[]
    for m in range(len(Cell_Begin_Line_Num)):
        str_split=[]
        str_split=Original_Lib[Cell_Begin_Line_Num[m]].split(' ')
        cell_name_str=str_split[3].strip('(')
        cell_name_str=cell_name_str.strip(')')
        Cell_Name.append(cell_name_str)
        #print(Cell_Name)
        #make pair for every cell begin line and end line
    Cell_Begin_End_Pair=[]
    for k in range(len(Cell_Begin_Line_Num)):
        Cell_Begin_End_Pair.append({'name':Cell_Name[k],'begin_line':Cell_Begin_Line_Num[k],'end_line':Cell_end_Line_Num[k],'original_PDP':0,'succeed_fail':1})
        #print(Cell_Begin_End_Pair)
        #walk through every cell contents
    for n in range(len(Cell_Begin_End_Pair)):
        begin_line=Cell_Begin_End_Pair[n]['begin_line']
        end_line=Cell_Begin_End_Pair[n]['end_line']
        PDP_rise_index='rise_power (power_template)'
        PDP_fall_index='fall_power (power_template)'     #temporarily only for combinational cell
        PDP_rise_rawdata=[]
        PDP_fall_rawdata=[]
            #data processing ,some parameters should consider about Len_of_Index1 and Len_of_Index2
        for p in range(begin_line,end_line):
            if PDP_rise_index in Original_Lib[p]:
                rise_rawdata0=[]
                rise_rawdata1=[]
                rise_rawdata0=Original_Lib[p+4].split(' ')
                rise_rawdata0.pop()
                del rise_rawdata0[0:12]
                a=rise_rawdata0[0].strip('"')
                a=a.strip(',')
                num0=float(a)
                b=rise_rawdata0[1].strip(',')
                b=b.strip('"')
                num1=float(b)
                    #print(num0)
                    #print(num1)
                rise_rawdata1=Original_Lib[p+5].split(' ')
                rise_rawdata1.pop()
                del rise_rawdata1[0:12]
                c=rise_rawdata1[0].strip('"')
                c=c.strip(',')
                num2=float(c)
                d=rise_rawdata1[1].strip(',')
                d=d.strip('"')
                num3=float(d)
                    #print(num2)
                    #print(num3)
                avg_rise_num=(num0+num1+num2+num3)/4
                    #print('avg!')
                    #print(avg_rise_num)
                PDP_rise_rawdata.append(avg_rise_num)
                    
                    #print("rise!")
                    #print(avg_rise_num)
                
            if PDP_fall_index in Original_Lib[p]:
                fall_rawdata0=[]
                fall_rawdata1=[]
                fall_rawdata0=Original_Lib[p+4].split(' ')
                fall_rawdata0.pop()
                del fall_rawdata0[0:12]
                a=fall_rawdata0[0].strip('"')
                a=a.strip(',')
                num0=float(a)
                b=fall_rawdata0[1].strip(',')
                b=b.strip('"')
                num1=float(b)
                    #print(num0)
                    #print(num1)
                fall_rawdata1=Original_Lib[p+5].split(' ')
                fall_rawdata1.pop()
                del fall_rawdata1[0:12]
                c=fall_rawdata1[0].strip('"')
                c=c.strip(',')
                num2=float(c)
                d=fall_rawdata1[1].strip(',')
                d=d.strip('"')
                num3=float(d)
                    #print(num2)
                    #print(num3)
                avg_fall_num=(num0+num1+num2+num3)/4
                    #print("fall!")
                    #print(avg_fall_num)
                PDP_fall_rawdata.append(avg_fall_num)
            #delete repeating elements
        PDP_rise_rawdata=list(set(PDP_rise_rawdata))
        PDP_fall_rawdata=list(set(PDP_fall_rawdata))
            # print('rise!')
            # print(PDP_rise_rawdata)
            # print('fall!')
            # print(PDP_fall_rawdata)
        PDP_rise=np.mean(PDP_rise_rawdata)
        PDP_fall=np.mean(PDP_fall_rawdata)
        original_PDP=(PDP_rise+PDP_fall)/2
        Cell_Begin_End_Pair[n]['original_PDP']=original_PDP
            #print(original_PDP)
            #print(PDP_rise_rawdata)
            #print(PDP_fall_rawdata)
        #print(Cell_Begin_End_Pair)
        if(original_PDP>1e+10):
            Cell_Begin_End_Pair[n]['succeed_fail']=0
    f.close()  
    return Cell_Begin_End_Pair

        #reading M for each cell from external file

    # f=open('M.txt','r+')
    # M_file=f.readlines()
    # strs=[]
    # for i in range(len(M_file)):
        # if(''!=M_file[i].split('\n')[0]):
            # strs.append(M_file[i].split('\n')[0])
        # #print(strs)
    # names=[]
    # M=[]
    # for j in range(len(strs)):
        # index=strs[j].find(' ')
        # names.append(strs[j][0:index])
        # M.append(strs[j][-1])
        # #print(names)
        # #print(M)
    # M_vector=[]
    # for k in range(len(names)):
        # M_vector.append({'name':names[k],'M':M[k]})
        # #print(M_vector) 
    # Equivalent_PDP=0
    # for u in range(len(M_vector)):
        # for v in range(len(Cell_Begin_End_Pair)):
            # if(M_vector[u]['name']==Cell_Begin_End_Pair[v]['name']):
                # Cell_Begin_End_Pair[v]['M']=M_vector[u]['M']
                # Cell_Begin_End_Pair[v]['equivalent PDP']=float(Cell_Begin_End_Pair[v]['original_PDP'])/float(Cell_Begin_End_Pair[v]['M'])
                    # #print(Cell_Begin_End_Pair[v])
                # if(Cell_Begin_End_Pair[v]['equivalent PDP']>100):
                    # Cell_Begin_End_Pair[v]['equivalent PDP']=10       #give this fail cell a big PDP but not infinite
                # Equivalent_PDP=Equivalent_PDP+Cell_Begin_End_Pair[v]['equivalent PDP']
    # Equivalent_PDP=Equivalent_PDP/len(M_vector)
        # #print(Equivalent_PDP)
        # #normlaize_value=1/(1+math.exp(-Equivalent_PDP))    #logistic function 
        # #print(normlaize_value)
    # f.close()
    # return Equivalent_PDP
    
  
    

for file in files:
    #得到该文件下所有目录的路径
    m = os.path.join(Base_path,file)
    #判断该路径下是否是文件夹
    if (os.path.isdir(m)):
        h = os.path.split(m)
        #print (h[1])
        subfolderlist.append(h[1])
#print (len(subfolderlist))
#print (subfolderlist)
#travel all the subfolders
for i in range(len(subfolderlist)):
    travel_dir=Base_path+'/'+subfolderlist[i]+'/liberate'
    #print(travel_dir)
    if(os.path.exists(travel_dir+'/lib')):
        feature=Calculate_PDP(travel_dir)
        
        f=open(travel_dir+'/tcl/char.tcl','r') #get VDD values
        char_tcl=f.readlines()
        VDD=char_tcl[8].strip().split(' ')[-1]
        VDD=str((float(VDD)-0.5)/(3-0.5))            #normalize
        #print(VDD)
        f.close()
        f=open(travel_dir+'/netlist/INVX1.scs','r') #get MU COX VTO VSS RCS RCD
        seven_dimensional=[]
        INVX_scs=f.readlines()
        MU_str=INVX_scs[14].split(' ')[7]
        MU=MU_str[3:]
        MU=str((float(MU)-5)/(100-5))             #normalize
        seven_dimensional.append(MU)
        COX_str=INVX_scs[14].split(' ')[8]
        COX=COX_str[4:]
        COX=str((float(COX)-50e-09)/(160e-09-50e-09))             #normalize
        seven_dimensional.append(COX)
        VTO_str=INVX_scs[14].split(' ')[13]
        VTO=VTO_str[4:]
        VTO=str((float(VTO)-0.3)/(1.3-0.3))             #normalize
        seven_dimensional.append(VTO)
        VSS_str=INVX_scs[14].split(' ')[21]
        VSS=VSS_str[4:]
        VSS=str((float(VSS)-0.06)/(0.3-0.06))              #normalize
        seven_dimensional.append(VSS)
        RCS_str=INVX_scs[14].split(' ')[28]
        RCS=RCS_str[4:]
        RCS=str((float(RCS)-500)/(3000-500))             #normalize 
        seven_dimensional.append(RCS)
        RCD_str=INVX_scs[14].split(' ')[29]
        RCD=RCD_str[4:]
        RCD=str((float(RCD)-500)/(3000-500))             #normalize
        seven_dimensional.append(RCD)
        seven_dimensional.append(VDD)
        if(seven_dimensional not in seven_dimensional_list): #remove redundant 7 variable parameters results
            seven_dimensional_list.append(seven_dimensional)
            Cell_list.append(feature)         #get each cell PDP and success or failure state info
#print(len(Cell_list))
#print(len(seven_dimensional_list))

f=open('INVX.txt','r')
INV_file=f.readlines()
f.close()
        
f=open('AND2.txt','r')
AND2_file=f.readlines()
f.close()  

f=open('BUF.txt','r')
BUF_file=f.readlines()
f.close()

f=open('NAND2.txt','r')
NAND2_file=f.readlines()
f.close()

f=open('XOR.txt','r')
XOR_file=f.readlines()
f.close()

f=open('OR.txt','r')
OR_file=f.readlines()
f.close()

for i in range(len(Cell_list)):
    INV_file[10]='                '+'[4, '+seven_dimensional_list[i][6]+', 0, 0, 0, 0, 0, 0, 0],'+'\n'
    INV_file[13]='                '+'[3, 0, -1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    INV_file[14]='                '+'[3, 0, 1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'    
    INV_file[17]='        y=torch.tensor(['+str(Cell_list[i][2]['succeed_fail'])+'])'+'\n'
    INV_file[18]='        '+'data'+str(6*i)+' = Data(x=x, y=y, edge_index=edge_index)'+'\n'
    INV_file[19]='        data_list.append(data'+str(6*i)+')'+'\n'
    
    AND2_file[12]='                '+'[4, '+seven_dimensional_list[i][6]+', 0, 0, 0, 0, 0, 0, 0],'+'\n'
    AND2_file[14]='                '+'[3, 0, 1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    AND2_file[15]=AND2_file[14]
    AND2_file[16]=AND2_file[14]
    AND2_file[17]='                '+'[3, 0, -1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    AND2_file[18]=AND2_file[17]
    AND2_file[19]=AND2_file[17]
    AND2_file[22]='        y=torch.tensor(['+str(Cell_list[i][0]['succeed_fail'])+'])'+'\n'
    AND2_file[23]='        '+'data'+str(6*i+1)+' = Data(x=x, y=y, edge_index=edge_index)'+'\n'
    AND2_file[24]='        data_list.append(data'+str(6*i+1)+')'+'\n'
    
    BUF_file[10]='                '+'[4, '+seven_dimensional_list[i][6]+', 0, 0, 0, 0, 0, 0, 0],'+'\n'
    BUF_file[13]='                '+'[3, 0, 1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    BUF_file[14]=BUF_file[13]
    BUF_file[15]='                '+'[3, 0, -1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    BUF_file[16]=BUF_file[15]
    BUF_file[19]='        y=torch.tensor(['+str(Cell_list[i][1]['succeed_fail'])+'])'+'\n'
    BUF_file[20]='        '+'data'+str(6*i+2)+' = Data(x=x, y=y, edge_index=edge_index)'+'\n'
    BUF_file[21]='        data_list.append(data'+str(6*i+2)+')'+'\n'
    
    NAND2_file[10]='                '+'[4, '+seven_dimensional_list[i][6]+', 0, 0, 0, 0, 0, 0, 0],'+'\n'
    NAND2_file[14]='                '+'[3, 0, -1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    NAND2_file[15]=NAND2_file[14]
    NAND2_file[16]='                '+'[3, 0, 1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    NAND2_file[17]=NAND2_file[16]
    NAND2_file[20]='        y=torch.tensor(['+str(Cell_list[i][3]['succeed_fail'])+'])'+'\n'
    NAND2_file[21]='        '+'data'+str(6*i+3)+' = Data(x=x, y=y, edge_index=edge_index)'+'\n'
    NAND2_file[22]='        data_list.append(data'+str(6*i+3)+')'+'\n'
    
    XOR_file[10]='                '+'[4, '+seven_dimensional_list[i][6]+', 0, 0, 0, 0, 0, 0, 0],'+'\n'
    XOR_file[14]='                '+'[3, 0, 1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    XOR_file[15]=XOR_file[14]
    XOR_file[16]=XOR_file[14]
    XOR_file[17]=XOR_file[14]
    XOR_file[18]=XOR_file[14]
    XOR_file[19]=XOR_file[14]
    XOR_file[20]='                '+'[3, 0, -1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    XOR_file[21]=XOR_file[20]
    XOR_file[22]=XOR_file[20]
    XOR_file[23]=XOR_file[20]
    XOR_file[24]=XOR_file[20]
    XOR_file[25]=XOR_file[20]
    XOR_file[28]='        y=torch.tensor(['+str(Cell_list[i][5]['succeed_fail'])+'])'+'\n'
    XOR_file[29]='        '+'data'+str(6*i+4)+' = Data(x=x, y=y, edge_index=edge_index)'+'\n'
    XOR_file[30]='        data_list.append(data'+str(6*i+4)+')'+'\n'
    
    OR_file[10]='                '+'[4, '+seven_dimensional_list[i][6]+', 0, 0, 0, 0, 0, 0, 0],'+'\n'
    OR_file[14]='                '+'[3, 0, 1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    OR_file[15]=OR_file[14]
    OR_file[16]=OR_file[14]
    OR_file[17]='                '+'[3, 0, -1, '+seven_dimensional_list[i][0]+', '+seven_dimensional_list[i][1]+', '+seven_dimensional_list[i][2]+', '+seven_dimensional_list[i][3]+', '+seven_dimensional_list[i][4]+', '+seven_dimensional_list[i][5]+'],'+'\n'
    OR_file[18]=OR_file[17]
    OR_file[19]=OR_file[17]
    OR_file[22]='        y=torch.tensor(['+str(Cell_list[i][4]['succeed_fail'])+'])'+'\n'
    OR_file[23]='        '+'data'+str(6*i+5)+' = Data(x=x, y=y, edge_index=edge_index)'+'\n'
    OR_file[24]='        data_list.append(data'+str(6*i+5)+')'+'\n'
    
    f=open('all.txt','a')
    f.writelines(INV_file)
    f.writelines(AND2_file)
    f.writelines(BUF_file)
    f.writelines(NAND2_file)
    f.writelines(XOR_file)
    f.writelines(OR_file)
    f.close()
