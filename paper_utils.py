
import sys, os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.cm as cm

from matplotlib.figure import Figure
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from stim_with_noise import Repetition_QEC_Stim_noise
from utils import correlation_matrix, data_analysis
from decoder import stim_decoder, corr_decoder

class paper_utils:

    def __init__(self) -> None:
        self.rounds = 10
        self.file_path = self.resource_path('Results/')

    def resource_path(self, relative_path):
        try: 
            base_path = sys.MEIPASS
        except:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    
    def get_file_name(self, distance, flag_length, initial_state):
        return f'distance-{distance}_rounds-{self.rounds}_flag_length-{flag_length}_initial_state-{initial_state}_index-0'
    
    def get_stab_data(self, distance, flag_length, initial_state):
        file_name = self.get_file_name(distance, flag_length, initial_state)
        return pd.read_csv( self.file_path + file_name + '.csv', index_col = 0, skiprows = [ 0 ] ).values.tolist( )

    def draw_ECDF_8(self):
        
        fig = Figure(figsize=(7, 4), dpi=150)
        distance = 9
        flag_length = 2
        initial_state = '0'
        datetime = self.resource_path(f'Calibration_data/properties_d{distance}_f{flag_length}_ini{initial_state}/')
        
        # Open the file for reading
        with open( self.file_path + f'/d{ distance }_f{ flag_length }_r{self.rounds}_ini{initial_state}_layout.txt' , 'r') as file:
            
            # Read each line in the file and add it to a list
            real_qubit_index = [ int( line.strip( ) ) for line in file.readlines( ) ] 

        # Stim 코드를 구성한다.
        stim_code_circ = Repetition_QEC_Stim_noise( distance, self.rounds, flag_length, datetime, real_qubit_index, initial_state)
        
        cx_list = [ ]
        reset_list = [ ]
        readout_list = [ ]
        single_list = [ ]
        
        for link in range( len( real_qubit_index ) - 1 ):
            cx_list.append( stim_code_circ.ecr_dic[ f'{(real_qubit_index[ link ],real_qubit_index[ link + 1 ])}' ][0] )
        
        for qubit in real_qubit_index:
            reset_list.append( stim_code_circ.reset_dic[ str(qubit) ][0] )
            readout_list.append( stim_code_circ.readout_dic[ str(qubit) ][0] )
            single_list.append( stim_code_circ.x_dic[ str(qubit) ][0])
        
        def ecdf(data):
            x = np.sort(data)
            n = x.size
            y = np.arange(1,n+1)/n
            return(x,y)

        color_map = [ 'crimson', 'magenta','navy']
        color_map_m = [ 'indianred','plum', '#336ea0' ]

        err = [cx_list,readout_list,single_list]
        mean = [np.mean(cx_list),np.mean(readout_list),np.mean(single_list)]

        error_type = ['ECR','Readout','SX']
        im = fig.add_subplot(111)
        for data in range(3):
            x,y = ecdf(err[data])
            
            im.plot(x,y,label = error_type[data] + f':{"{:.2e}".format(round(mean[data],6))}',linewidth = 3, color = color_map[data])
            
            mx,my = [mean[data],mean[data]],[0,1]
            im.plot(mx,my,'--',linewidth=2,color = color_map_m[data])

        #im.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
        im.legend(loc='lower center',fancybox=True, shadow=True, ncol=5, fontsize = 9)
        im.set_ylabel('ECDF',fontsize = 15)
        im.set_xlabel('Error rate',fontsize = 15)
        im.set_ylim(0,1)
        im.semilogx()
        return fig
    
    def draw_syndrome_graph_9(self):
        distance = 9
        flag_length = 2
        initial_state = '0' # 확인필요
        fig = Figure(figsize=(5, 4), dpi=150)
        
        # Open the file for reading
        #with open( '~/repetition_code/compiled_drawing/Results/d3_f0_r10_ini-_layout.txt' , 'r' ) as file:
        with open(self.file_path + f'd{distance}_f{flag_length}_r{self.rounds}_ini{initial_state}_layout.txt' , 'r' ) as file:
            # Read each line in the file and add it to a list
            real_qubit_index = [ int( line.strip( ) ) for line in file.readlines( ) ]
        properties = self.resource_path(f'Calibration_data/properties_d{distance}_f{flag_length}_ini{initial_state}/')
        stim_code_circ = Repetition_QEC_Stim_noise( distance, self.rounds, flag_length, properties, real_qubit_index, initial_state).generate_circuit()
        
        syndrome_g = stim_decoder(stim_code_circ).detector_error_model_to_nx_graph()
        
        pos = { }
        num_obs = stim_code_circ.num_detectors
        for node_num in syndrome_g.nodes( ):
            if node_num != num_obs:
                pos[ node_num ] = ( node_num % ( distance - 1 ), node_num // ( distance - 1 ) )
            else:
                pos[ node_num ] = ( -1, -1 )
        
        edgelist, edge_color = zip( *nx.get_edge_attributes( syndrome_g, 'weight' ).items( ) )
        
        # Draw graph
        im = fig.add_subplot(111)
        nodes = nx.draw_networkx_nodes( syndrome_g, pos, node_color = '#aaaaaa', ax = im, node_size=100 )
        edges = nx.draw_networkx_edges( syndrome_g, pos, edgelist = edgelist, edge_color = edge_color, edge_cmap = cm.coolwarm, ax = im )
        fig.colorbar( edges )
        im.set_xlabel('Space',fontsize=10)
        im.set_ylabel('Time',fontsize=10)

        S = [ ]
        T = [ ]
        ST = [ ]
        norm = np.sqrt(np.sum([v**2 for v in list(edge_color)]))
        for tuple in list(nx.get_edge_attributes( syndrome_g, 'weight' ).items( )):
            node1_x, node1_y = pos[tuple[0][0]][0], pos[tuple[0][0]][1]
            node2_x, node2_y = pos[tuple[0][1]][0], pos[tuple[0][1]][1]
            weight = tuple[1]
            
            if min(node1_x,node2_x) != -1:
                del_x = abs(node1_x -node2_x)
                del_y = abs(node1_y -node2_y)
                if del_x == 1 and del_y == 0:
                    S.append(weight)
                elif del_x == 0 and del_y ==1:
                    T.append(weight)
                elif del_x == 1 and del_y ==1:
                    ST.append(weight)
            else:
                S.append(weight)
                
        print_str = f'Mean weight: S type : {np.mean(S)}, T type : {np.mean(T)} , ST type : {np.mean(ST)}'
        print(print_str)
        return fig
        
    def draw_Correlation_matrix_0(self):
        distance = 9
        flag_length = 0
        initial_state = '1'
        syn_num = (distance-1)*(self.rounds+1)
        fig = Figure(figsize=(7, 6), dpi=150)

        tools = data_analysis(distance=distance, flag_length=flag_length)
        syn_data = tools.stab2syn( self.get_stab_data(distance,flag_length,initial_state), self.rounds )
        correlation_M_real = correlation_matrix(syn_data)
        p_matrix_real = correlation_M_real.p_matrix#[distance-1:syn_num,distance-1:syn_num]

        plot = fig.add_subplot(111)
        #colormapping = cm.ScalarMappable(cmap=cm.RdPu, norm=Normalize(vmin=0, vmax=0.3, clip=False))
        im = plot.imshow(p_matrix_real, cmap=cm.RdPu, vmin=0, vmax=0.03)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Space-Time)')
        plot.set_ylabel('Node index $j$ (Space-Time)')
        
        #x = np.arange((distance-1)/4 + (distance-3)/4,syn_num + (distance-1) ,(distance-1))
        x = np.arange(-0.5, syn_num, (distance-1)//2)
        #xtic = ['t'+str(int(i)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('t'+str(i//2))
        plot.set_xticks(ticks=x, labels=xtic)
        plot.set_yticks(x,xtic)
        return fig

    def draw_Correlation_matrix_1(self):
        distance = 9
        flag_length = 1
        initial_state = '1'
        syn_num = (distance-1)*(self.rounds+1)
        fig = Figure(figsize=(7, 6), dpi=150)

        tools = data_analysis(distance=distance, flag_length=flag_length)
        syn_data = tools.stab2syn( self.get_stab_data(distance,flag_length,initial_state), self.rounds )
        correlation_M_real = correlation_matrix(syn_data)
        p_matrix_real = correlation_M_real.p_matrix#[distance-1:syn_num,distance-1:syn_num]

        plot = fig.add_subplot(111)
        #colormapping = cm.ScalarMappable(cmap=cm.RdPu, norm=Normalize(vmin=0, vmax=0.3, clip=False))
        im = plot.imshow(p_matrix_real, cmap=cm.RdPu, vmin=0,vmax=0.08)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Space-Time)')
        plot.set_ylabel('Node index $j$ (Space-Time)')
        
        #x = np.arange((distance-1)/4 + (distance-3)/4,syn_num + (distance-1) ,(distance-1))
        x = np.arange(-0.5, syn_num, (distance-1)//2)
        #xtic = ['t'+str(int(i)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('t'+str(i//2))
        plot.set_xticks(ticks=x, labels=xtic)
        plot.set_yticks(x,xtic)
        return fig

    def draw_Correlation_matrix_2(self):
        distance = 9
        flag_length = 2
        initial_state = '1'
        syn_num = (distance-1)*(self.rounds+1)
        fig = Figure(figsize=(7, 6), dpi=150)

        tools = data_analysis(distance=distance, flag_length=flag_length)
        syn_data = tools.stab2syn( self.get_stab_data(distance,flag_length,initial_state), self.rounds )
        correlation_M_real = correlation_matrix(syn_data)
        p_matrix_real = correlation_M_real.p_matrix#[distance-1:syn_num,distance-1:syn_num]

        plot = fig.add_subplot(111)
        #colormapping = cm.ScalarMappable(cmap=cm.RdPu, norm=Normalize(vmin=0, vmax=0.3, clip=False))
        im = plot.imshow(p_matrix_real, cmap=cm.RdPu, vmin=0,vmax=0.2)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Space-Time)')
        plot.set_ylabel('Node index $j$ (Space-Time)')
        
        #x = np.arange((distance-1)/4 + (distance-3)/4,syn_num + (distance-1) ,(distance-1))
        x = np.arange(-0.5, syn_num, (distance-1)//2)
        #xtic = ['t'+str(int(i)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('t'+str(i//2))
        plot.set_xticks(ticks=x, labels=xtic)
        plot.set_yticks(x,xtic)
        return fig

    def draw_defect_syndrome_0(self):
        flag_length = 0
        cmap = ['royalblue','mediumorchid','mediumseagreen','salmon']
        fig = Figure(figsize=(12, 8), dpi=150)

        data_1 = {}
        width = 0.2
        initial_state_list = ['0', '1', '+', '-']
        plot = fig.add_subplot(111)
        for distance in [3,5,7,9]:
            
            syn_num_dic = {}
            
            for initial_state in initial_state_list:

                file_path = self.resource_path('Results/')
                file_name = f'distance-{distance}_rounds-{self.rounds}_flag_length-{flag_length}_initial_state-{initial_state}_index-0'
                self.stab_data = pd.read_csv( file_path + file_name + '.csv', index_col = 0, skiprows = [ 0 ] ).values.tolist( )
                tools = data_analysis(distance=distance, flag_length=flag_length)
                syn_data = tools.stab2syn( self.stab_data, self.rounds )

                for syn in syn_data:
                    if np.sum(syn[0]) not in syn_num_dic.keys():
                        syn_num_dic[np.sum(syn[0])] = syn[-1]
                    else:
                        syn_num_dic[np.sum(syn[0])] += syn[-1]

                                
            data_1[distance] = syn_num_dic

            if distance == 3:
                plot.bar(np.array(list(syn_num_dic.keys())) - 1.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            elif distance == 5:
                plot.bar(np.array(list(syn_num_dic.keys())) - 0.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            elif distance == 7:
                plot.bar(np.array(list(syn_num_dic.keys())) + 0.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            else:
                plot.bar(np.array(list(syn_num_dic.keys())) + 1.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            
        #plot.set_xticks(np.arange(0,26,2))
        #plot.set_xlim(-1,24)
        plot.set_xlabel('Number of defect syndromes')
        plot.set_ylabel('Pobability of samples')
        return fig

    def draw_defect_syndrome_1(self):
        flag_length = 1
        cmap = ['royalblue','mediumorchid','mediumseagreen','salmon']
        fig = Figure(figsize=(12, 8), dpi=150)

        data_1 = {}
        width = 0.2
        initial_state_list = ['0', '1', '+', '-']
        plot = fig.add_subplot(111)
        for distance in [3,5,7,9]:
            
            syn_num_dic = {}
            
            for initial_state in initial_state_list:

                file_path = self.resource_path('Results/')
                file_name = f'distance-{distance}_rounds-{self.rounds}_flag_length-{flag_length}_initial_state-{initial_state}_index-0'
                self.stab_data = pd.read_csv( file_path + file_name + '.csv', index_col = 0, skiprows = [ 0 ] ).values.tolist( )
                tools = data_analysis(distance=distance, flag_length=flag_length)
                syn_data = tools.stab2syn( self.stab_data, self.rounds )

                for syn in syn_data:
                    if np.sum(syn[0]) not in syn_num_dic.keys():
                        syn_num_dic[np.sum(syn[0])] = syn[-1]
                    else:
                        syn_num_dic[np.sum(syn[0])] += syn[-1]

                                
            data_1[distance] = syn_num_dic

            if distance == 3:
                plot.bar(np.array(list(syn_num_dic.keys())) - 1.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            elif distance == 5:
                plot.bar(np.array(list(syn_num_dic.keys())) - 0.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            elif distance == 7:
                plot.bar(np.array(list(syn_num_dic.keys())) + 0.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            else:
                plot.bar(np.array(list(syn_num_dic.keys())) + 1.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            
        #plot.set_xticks(np.arange(0,26,2))
        #plot.set_xlim(-1,24)
        plot.set_xlabel('Number of defect syndromes')
        plot.set_ylabel('Pobability of samples')
        return fig

    def draw_defect_syndrome_2(self):
        flag_length = 2
        cmap = ['royalblue','mediumorchid','mediumseagreen','salmon']
        fig = Figure(figsize=(12, 8), dpi=150)

        data_1 = {}
        width = 0.2
        initial_state_list = ['0', '1', '+', '-']
        plot = fig.add_subplot(111)
        for distance in [3,5,7,9]:
            
            syn_num_dic = {}
            
            for initial_state in initial_state_list:

                file_path = self.resource_path('Results/')
                file_name = f'distance-{distance}_rounds-{self.rounds}_flag_length-{flag_length}_initial_state-{initial_state}_index-0'
                self.stab_data = pd.read_csv( file_path + file_name + '.csv', index_col = 0, skiprows = [ 0 ] ).values.tolist( )
                tools = data_analysis(distance=distance, flag_length=flag_length)
                syn_data = tools.stab2syn( self.stab_data, self.rounds )

                for syn in syn_data:
                    if np.sum(syn[0]) not in syn_num_dic.keys():
                        syn_num_dic[np.sum(syn[0])] = syn[-1]
                    else:
                        syn_num_dic[np.sum(syn[0])] += syn[-1]

                                
            data_1[distance] = syn_num_dic

            if distance == 3:
                plot.bar(np.array(list(syn_num_dic.keys())) - 1.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            elif distance == 5:
                plot.bar(np.array(list(syn_num_dic.keys())) - 0.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            elif distance == 7:
                plot.bar(np.array(list(syn_num_dic.keys())) + 0.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            else:
                plot.bar(np.array(list(syn_num_dic.keys())) + 1.5*width, np.array(list(syn_num_dic.values()))/200000, label = f'distance:{distance}', width = width, color = cmap[(distance-3)//2])
            
        #plot.set_xticks(np.arange(0,26,2))
        #plot.set_xlim(-1,24)
        plot.set_xlabel('Number of defect syndromes')
        plot.set_ylabel('Pobability of samples')
        return fig

    def draw_Logical_error_rate_hardware_0(self):

        rounds_list = range( 1, 11 )
        distances = [ 3,5,7,9 ]
        fig = Figure(figsize=(10, 8), dpi=150)

        color_list_f = [ 'blue', 'green', 'red', 'grey' ]
        style = ['-s','-^','-o','-p']

        # data_path = f"./Paper_data2/correct_excel/"
        data_path = f'./Main_result/'
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
            
        shots = 50000
        
        flag_length = 0
            
        plot = fig.add_subplot(111)
        c = 0
        for distance in distances:
            
            # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
            file_name = f'distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                
            file = data_path + file_name
            
            df = pd.read_excel( file,sheet_name = ['real result' ], index_col = 0 )
            correct_nums = np.array( df[ 'real result' ].values.tolist( ) )
            correct_nums = 1 - correct_nums / (shots)
            
            # # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
            p_err = [np.mean( correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            p_err_var = [ np.std(correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            plot.errorbar( rounds_list, p_err, p_err_var, fmt=style[c], capsize = 8, capthick=2, color = color_list_f[c] ,label = rf'$[[{distance},1,{distance}]]$' +rf'$_{{f={flag_length}}}$', linewidth = 2, markersize = 10 )
            c += 1
        plot.semilogy( )
        plot.set_xlabel( "Syndrome extraction cycles", fontsize = 20 )
        plot.set_ylabel( "logical error rate $(p_L)$", fontsize = 20 )
        plot.legend( loc = "lower right", fontsize = 10 )
        return fig

    def draw_Logical_error_rate_hardware_1(self):

        rounds_list = range( 1, 11 )
        distances = [ 3,5,7,9 ]
        fig = Figure(figsize=(10, 8), dpi=150)

        color_list_f = [ 'blue', 'green', 'red', 'grey' ]
        style = ['-.s','-.^','-.o','-.p']

        # data_path = f"./Paper_data2/correct_excel/"
        data_path = f'./Main_result/'
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
            
        shots = 50000
        
        flag_length = 1
            
        plot = fig.add_subplot(111)
        c = 0
        for distance in distances:
            
            # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
            file_name = f'distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                
            file = data_path + file_name
            
            df = pd.read_excel( file,sheet_name = ['real result' ], index_col = 0 )
            correct_nums = np.array( df[ 'real result' ].values.tolist( ) )
            correct_nums = 1 - correct_nums / (shots)
            
            # # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
            p_err = [np.mean( correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            p_err_var = [ np.std(correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            plot.errorbar( rounds_list, p_err, p_err_var, fmt=style[c], capsize = 8, capthick=2, color = color_list_f[c] ,label = rf'$[[{distance},1,{distance}]]$' +rf'$_{{f={flag_length}}}$', linewidth = 2, markersize = 10 )
            c += 1
        plot.semilogy( )
        plot.set_xlabel( "Syndrome extraction cycles", fontsize = 20 )
        plot.set_ylabel( "logical error rate $(p_L)$", fontsize = 20 )
        plot.legend( loc = "lower right", fontsize = 10 )
        return fig

    def draw_Logical_error_rate_hardware_2(self):

        rounds_list = range( 1, 11 )
        distances = [ 3,5,7,9 ]
        fig = Figure(figsize=(10, 8), dpi=150)

        color_list_f = [ 'blue', 'green', 'red', 'grey' ]
        style = ['--s','--^','--o','--p']

        # data_path = f"./Paper_data2/correct_excel/"
        data_path = f'./Main_result/'
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
            
        shots = 50000
        
        flag_length = 2
            
        plot = fig.add_subplot(111)
        c = 0
        for distance in distances:
            
            # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
            file_name = f'distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                
            file = data_path + file_name
            
            df = pd.read_excel( file,sheet_name = ['real result' ], index_col = 0 )
            correct_nums = np.array( df[ 'real result' ].values.tolist( ) )
            correct_nums = 1 - correct_nums / (shots)
            
            # # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
            p_err = [np.mean( correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            p_err_var = [ np.std(correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            plot.errorbar( rounds_list, p_err, p_err_var, fmt=style[c], capsize = 8, capthick=2, color = color_list_f[c] ,label = rf'$[[{distance},1,{distance}]]$' +rf'$_{{f={flag_length}}}$', linewidth = 2, markersize = 10 )
            c += 1
        plot.semilogy( )
        plot.set_xlabel( "Syndrome extraction cycles", fontsize = 20 )
        plot.set_ylabel( "logical error rate $(p_L)$", fontsize = 20 )
        plot.legend( loc = "lower right", fontsize = 10 )
        return fig

    def draw_Logical_error_rate_sample_0(self):

        rounds_list = range( 1, 11 )
        distances = [ 3,5,7,9 ]
        fig = Figure(figsize=(10, 8), dpi=150)

        color_list_cor = [ 'navy', 'darkgreen', 'darkred', 'k' ]
        style = ['-s','-^','-o','-p']

        # data_path = f"./Paper_data2/correct_excel/"
        data_path = f'./Main_result/'
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
            
        shots = 50000
        
        flag_length = 0
            
        plot = fig.add_subplot(111)
        c = 0
        for distance in distances:
            
            # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
            file_name = f'distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                
            file = data_path + file_name
            
            df = pd.read_excel( file,sheet_name = ['real p result' ], index_col = 0 )
            correct_nums = np.array( df[ 'real p result' ].values.tolist( ) )
            correct_nums = 1 - correct_nums / (shots)
            
            # # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
            p_err = [np.mean( correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            p_err_var = [ np.std(correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            plot.errorbar( rounds_list, p_err, p_err_var, fmt=style[c], capsize = 8, capthick=2, color = color_list_cor[c] ,label = rf'$[[{distance},1,{distance}]]$' +rf'$_{{f={flag_length}}}$', linewidth = 2, markersize = 10 )
            c += 1
        plot.semilogy( )
        plot.set_xlabel( "Syndrome extraction cycles", fontsize = 20 )
        plot.set_ylabel( "logical error rate $(p_L)$", fontsize = 20 )
        plot.legend( loc = "lower right", fontsize = 10 )
        return fig
    
    def draw_Logical_error_rate_sample_1(self):

        rounds_list = range( 1, 11 )
        distances = [ 3,5,7,9 ]
        fig = Figure(figsize=(10, 8), dpi=150)

        color_list_cor = [ 'navy', 'darkgreen', 'darkred', 'k' ]
        style = ['-.s','-.^','-.o','-.p']

        # data_path = f"./Paper_data2/correct_excel/"
        data_path = f'./Main_result/'
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
            
        shots = 50000
        
        flag_length = 1
            
        plot = fig.add_subplot(111)
        c = 0
        for distance in distances:
            
            # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
            file_name = f'distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                
            file = data_path + file_name
            
            df = pd.read_excel( file,sheet_name = ['real p result' ], index_col = 0 )
            correct_nums = np.array( df[ 'real p result' ].values.tolist( ) )
            correct_nums = 1 - correct_nums / (shots)
            
            # # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
            p_err = [np.mean( correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            p_err_var = [ np.std(correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            plot.errorbar( rounds_list, p_err, p_err_var, fmt=style[c], capsize = 8, capthick=2, color = color_list_cor[c] ,label = rf'$[[{distance},1,{distance}]]$' +rf'$_{{f={flag_length}}}$', linewidth = 2, markersize = 10 )
            c += 1
        plot.semilogy( )
        plot.set_xlabel( "Syndrome extraction cycles", fontsize = 20 )
        plot.set_ylabel( "logical error rate $(p_L)$", fontsize = 20 )
        plot.legend( loc = "lower right", fontsize = 10 )
        return fig
    
    def draw_Logical_error_rate_sample_2(self):

        rounds_list = range( 1, 11 )
        distances = [ 3,5,7,9 ]
        fig = Figure(figsize=(10, 8), dpi=150)

        color_list_cor = [ 'navy', 'darkgreen', 'darkred', 'k' ]
        style = ['--s','--^','--o','--p']

        # data_path = f"./Paper_data2/correct_excel/"
        data_path = f'./Main_result/'
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
            
        shots = 50000
        
        flag_length = 2
            
        plot = fig.add_subplot(111)
        c = 0
        for distance in distances:
            
            # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
            file_name = f'distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                
            file = data_path + file_name
            
            df = pd.read_excel( file,sheet_name = ['real p result' ], index_col = 0 )
            correct_nums = np.array( df[ 'real p result' ].values.tolist( ) )
            correct_nums = 1 - correct_nums / (shots)
            
            # # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
            p_err = [np.mean( correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            p_err_var = [ np.std(correct_nums[ i, : ] ) for i in range( len( correct_nums ) ) ]
            plot.errorbar( rounds_list, p_err, p_err_var, fmt=style[c], capsize = 8, capthick=2, color = color_list_cor[c] ,label = rf'$[[{distance},1,{distance}]]$' +rf'$_{{f={flag_length}}}$', linewidth = 2, markersize = 10 )
            c += 1
        plot.semilogy( )
        plot.set_xlabel( "Syndrome extraction cycles", fontsize = 20 )
        plot.set_ylabel( "logical error rate $(p_L)$", fontsize = 20 )
        plot.legend( loc = "lower right", fontsize = 10 )
        return fig

    def draw_ECDF_3(self):
        distance = 3
        flag_length = 2
        initial_state = '-'

        fig = Figure(figsize=(7, 4), dpi=150)
        datetime = self.resource_path(f'Calibration_data/properties_d{distance}_f{flag_length}_ini{initial_state}/')
        
        # Open the file for reading
        with open( self.file_path + f'/d{ distance }_f{ flag_length }_r{self.rounds}_ini{initial_state}_layout.txt' , 'r') as file:
            
            # Read each line in the file and add it to a list
            real_qubit_index = [ int( line.strip( ) ) for line in file.readlines( ) ] 

        # Stim 코드를 구성한다.
        stim_code_circ = Repetition_QEC_Stim_noise( distance, self.rounds, flag_length, datetime, real_qubit_index, initial_state)
        
        cx_list = [ ]
        reset_list = [ ]
        readout_list = [ ]
        single_list = [ ]
        
        for link in range( len( real_qubit_index ) - 1 ):
            cx_list.append( stim_code_circ.ecr_dic[ f'{(real_qubit_index[ link ],real_qubit_index[ link + 1 ])}' ][0] )
        
        for qubit in real_qubit_index:
            reset_list.append( stim_code_circ.reset_dic[ str(qubit) ][0] )
            readout_list.append( stim_code_circ.readout_dic[ str(qubit) ][0] )
            single_list.append( stim_code_circ.x_dic[ str(qubit) ][0])
        
        def ecdf(data):
            x = np.sort(data)
            n = x.size
            y = np.arange(1,n+1)/n
            return(x,y)

        color_map = [ 'crimson', 'magenta','navy']
        color_map_m = [ 'indianred','plum', '#336ea0' ]

        err = [cx_list,readout_list,single_list]
        mean = [np.mean(cx_list),np.mean(readout_list),np.mean(single_list)]

        error_type = ['ECR','Readout','SX']
        im = fig.add_subplot(111)
        for data in range(3):
            x,y = ecdf(err[data])
            
            im.plot(x,y,label = error_type[data] + f':{"{:.2e}".format(round(mean[data],6))}',linewidth = 3, color = color_map[data])
            
            mx,my = [mean[data],mean[data]],[0,1]
            im.plot(mx,my,'--',linewidth=2,color = color_map_m[data])

        #im.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
        im.legend(loc='lower center',fancybox=True, shadow=True, ncol=5, fontsize = 9)
        im.set_ylabel('ECDF',fontsize = 15)
        im.set_xlabel('Error rate',fontsize = 15)
        im.set_ylim(0,1)
        im.semilogx()
        return fig

    def draw_ECDF_5(self):
        distance = 5
        flag_length = 2
        initial_state = '-'
        fig = Figure(figsize=(7, 4), dpi=150)
        datetime = self.resource_path(f'Calibration_data/properties_d{distance}_f{flag_length}_ini{initial_state}/')
        
        # Open the file for reading
        with open( self.file_path + f'/d{ distance }_f{ flag_length }_r{self.rounds}_ini{initial_state}_layout.txt' , 'r') as file:
            
            # Read each line in the file and add it to a list
            real_qubit_index = [ int( line.strip( ) ) for line in file.readlines( ) ] 

        # Stim 코드를 구성한다.
        stim_code_circ = Repetition_QEC_Stim_noise( distance, self.rounds, flag_length, datetime, real_qubit_index, initial_state)
        
        cx_list = [ ]
        reset_list = [ ]
        readout_list = [ ]
        single_list = [ ]
        
        for link in range( len( real_qubit_index ) - 1 ):
            cx_list.append( stim_code_circ.ecr_dic[ f'{(real_qubit_index[ link ],real_qubit_index[ link + 1 ])}' ][0] )
        
        for qubit in real_qubit_index:
            reset_list.append( stim_code_circ.reset_dic[ str(qubit) ][0] )
            readout_list.append( stim_code_circ.readout_dic[ str(qubit) ][0] )
            single_list.append( stim_code_circ.x_dic[ str(qubit) ][0])
        
        def ecdf(data):
            x = np.sort(data)
            n = x.size
            y = np.arange(1,n+1)/n
            return(x,y)

        color_map = [ 'crimson', 'magenta','navy']
        color_map_m = [ 'indianred','plum', '#336ea0' ]

        err = [cx_list,readout_list,single_list]
        mean = [np.mean(cx_list),np.mean(readout_list),np.mean(single_list)]

        error_type = ['ECR','Readout','SX']
        im = fig.add_subplot(111)
        for data in range(3):
            x,y = ecdf(err[data])
            
            im.plot(x,y,label = error_type[data] + f':{"{:.2e}".format(round(mean[data],6))}',linewidth = 3, color = color_map[data])
            
            mx,my = [mean[data],mean[data]],[0,1]
            im.plot(mx,my,'--',linewidth=2,color = color_map_m[data])

        #im.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
        im.legend(loc='lower center',fancybox=True, shadow=True, ncol=5, fontsize = 9)
        im.set_ylabel('ECDF',fontsize = 15)
        im.set_xlabel('Error rate',fontsize = 15)
        im.set_ylim(0,1)
        im.semilogx()
        return fig

    def draw_ECDF_7(self):
        distance = 7
        flag_length = 2
        initial_state = '-'
        fig = Figure(figsize=(7, 4), dpi=150)
        datetime = self.resource_path(f'Calibration_data/properties_d{distance}_f{flag_length}_ini{initial_state}/')
        
        # Open the file for reading
        with open( self.file_path + f'/d{ distance }_f{ flag_length }_r{self.rounds}_ini{initial_state}_layout.txt' , 'r') as file:
            
            # Read each line in the file and add it to a list
            real_qubit_index = [ int( line.strip( ) ) for line in file.readlines( ) ] 

        # Stim 코드를 구성한다.
        stim_code_circ = Repetition_QEC_Stim_noise( distance, self.rounds, flag_length, datetime, real_qubit_index, initial_state)
        
        cx_list = [ ]
        reset_list = [ ]
        readout_list = [ ]
        single_list = [ ]
        
        for link in range( len( real_qubit_index ) - 1 ):
            cx_list.append( stim_code_circ.ecr_dic[ f'{(real_qubit_index[ link ],real_qubit_index[ link + 1 ])}' ][0] )
        
        for qubit in real_qubit_index:
            reset_list.append( stim_code_circ.reset_dic[ str(qubit) ][0] )
            readout_list.append( stim_code_circ.readout_dic[ str(qubit) ][0] )
            single_list.append( stim_code_circ.x_dic[ str(qubit) ][0])
        
        def ecdf(data):
            x = np.sort(data)
            n = x.size
            y = np.arange(1,n+1)/n
            return(x,y)

        color_map = [ 'crimson', 'magenta','navy']
        color_map_m = [ 'indianred','plum', '#336ea0' ]

        err = [cx_list,readout_list,single_list]
        mean = [np.mean(cx_list),np.mean(readout_list),np.mean(single_list)]

        error_type = ['ECR','Readout','SX']
        im = fig.add_subplot(111)
        for data in range(3):
            x,y = ecdf(err[data])
            
            im.plot(x,y,label = error_type[data] + f':{"{:.2e}".format(round(mean[data],6))}',linewidth = 3, color = color_map[data])
            
            mx,my = [mean[data],mean[data]],[0,1]
            im.plot(mx,my,'--',linewidth=2,color = color_map_m[data])

        #im.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
        im.legend(loc='lower center',fancybox=True, shadow=True, ncol=5, fontsize = 9)
        im.set_ylabel('ECDF',fontsize = 15)
        im.set_xlabel('Error rate',fontsize = 15)
        im.set_ylim(0,1)
        im.semilogx()
        return fig

    def draw_ECDF_9(self):
        distance = 9
        flag_length = 2
        initial_state = '-'
        fig = Figure(figsize=(7, 4), dpi=150)
        datetime = self.resource_path(f'Calibration_data/properties_d{distance}_f{flag_length}_ini{initial_state}/')
        
        # Open the file for reading
        with open( self.file_path + f'/d{ distance }_f{ flag_length }_r{self.rounds}_ini{initial_state}_layout.txt' , 'r') as file:
            
            # Read each line in the file and add it to a list
            real_qubit_index = [ int( line.strip( ) ) for line in file.readlines( ) ] 

        # Stim 코드를 구성한다.
        stim_code_circ = Repetition_QEC_Stim_noise( distance, self.rounds, flag_length, datetime, real_qubit_index, initial_state)
        
        cx_list = [ ]
        reset_list = [ ]
        readout_list = [ ]
        single_list = [ ]
        
        for link in range( len( real_qubit_index ) - 1 ):
            cx_list.append( stim_code_circ.ecr_dic[ f'{(real_qubit_index[ link ],real_qubit_index[ link + 1 ])}' ][0] )
        
        for qubit in real_qubit_index:
            reset_list.append( stim_code_circ.reset_dic[ str(qubit) ][0] )
            readout_list.append( stim_code_circ.readout_dic[ str(qubit) ][0] )
            single_list.append( stim_code_circ.x_dic[ str(qubit) ][0])
        
        def ecdf(data):
            x = np.sort(data)
            n = x.size
            y = np.arange(1,n+1)/n
            return(x,y)

        color_map = [ 'crimson', 'magenta','navy']
        color_map_m = [ 'indianred','plum', '#336ea0' ]

        err = [cx_list,readout_list,single_list]
        mean = [np.mean(cx_list),np.mean(readout_list),np.mean(single_list)]

        error_type = ['ECR','Readout','SX']
        im = fig.add_subplot(111)
        for data in range(3):
            x,y = ecdf(err[data])
            
            im.plot(x,y,label = error_type[data] + f':{"{:.2e}".format(round(mean[data],6))}',linewidth = 3, color = color_map[data])
            
            mx,my = [mean[data],mean[data]],[0,1]
            im.plot(mx,my,'--',linewidth=2,color = color_map_m[data])

        #im.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
        im.legend(loc='lower center',fancybox=True, shadow=True, ncol=5, fontsize = 9)
        im.set_ylabel('ECDF',fontsize = 15)
        im.set_xlabel('Error rate',fontsize = 15)
        im.set_ylim(0,1)
        im.semilogx()
        return fig

    def draw_Correlation_matrix_3(self):
        distance = 3
        flag_length = 2
        initial_state = '-'
        syn_num = (distance-1)*(self.rounds+1)
        fig = Figure(figsize=(7, 6), dpi=150)

        tools = data_analysis(distance=distance, flag_length=flag_length)
        syn_data = tools.stab2syn( self.get_stab_data(distance,flag_length,initial_state), self.rounds )
        correlation_M_real = correlation_matrix(syn_data)
        p_matrix_real = correlation_M_real.p_matrix#[distance-1:syn_num,distance-1:syn_num]

        plot = fig.add_subplot(111)
        #colormapping = cm.ScalarMappable(cmap=cm.RdPu, norm=Normalize(vmin=0, vmax=0.3, clip=False))
        im = plot.imshow(p_matrix_real, cmap=cm.RdPu, vmin=0 ,vmax=0.2)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Space-Time)')
        plot.set_ylabel('Node index $j$ (Space-Time)')
        
        #x = np.arange((distance-1)/4 + (distance-3)/4,syn_num + (distance-1) ,(distance-1))
        x = np.arange(-0.5, syn_num, (distance-1)//2)
        #xtic = ['t'+str(int(i)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('t'+str(i//2))
        plot.set_xticks(ticks=x, labels=xtic)
        plot.set_yticks(x,xtic)
        return fig

    def draw_Correlation_matrix_5(self):
        distance = 5
        flag_length = 2
        initial_state = '-'
        syn_num = (distance-1)*(self.rounds+1)
        fig = Figure(figsize=(7, 6), dpi=150)

        tools = data_analysis(distance=distance, flag_length=flag_length)
        syn_data = tools.stab2syn( self.get_stab_data(distance,flag_length,initial_state), self.rounds )
        correlation_M_real = correlation_matrix(syn_data)
        p_matrix_real = correlation_M_real.p_matrix#[distance-1:syn_num,distance-1:syn_num]

        plot = fig.add_subplot(111)
        #colormapping = cm.ScalarMappable(cmap=cm.RdPu, norm=Normalize(vmin=0, vmax=0.3, clip=False))
        im = plot.imshow(p_matrix_real, cmap=cm.RdPu, vmin=0,vmax=0.2)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Space-Time)')
        plot.set_ylabel('Node index $j$ (Space-Time)')
        
        #x = np.arange((distance-1)/4 + (distance-3)/4,syn_num + (distance-1) ,(distance-1))
        x = np.arange(-0.5, syn_num, (distance-1)//2)
        #xtic = ['t'+str(int(i)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('t'+str(i//2))
        plot.set_xticks(ticks=x, labels=xtic)
        plot.set_yticks(x,xtic)
        return fig

    def draw_Correlation_matrix_7(self):
        distance = 7
        flag_length = 2
        initial_state = '-'
        syn_num = (distance-1)*(self.rounds+1)
        fig = Figure(figsize=(7, 6), dpi=150)

        tools = data_analysis(distance=distance, flag_length=flag_length)
        syn_data = tools.stab2syn( self.get_stab_data(distance,flag_length,initial_state), self.rounds )
        correlation_M_real = correlation_matrix(syn_data)
        p_matrix_real = correlation_M_real.p_matrix#[distance-1:syn_num,distance-1:syn_num]

        plot = fig.add_subplot(111)
        #colormapping = cm.ScalarMappable(cmap=cm.RdPu, norm=Normalize(vmin=0, vmax=0.3, clip=False))
        im = plot.imshow(p_matrix_real, cmap=cm.RdPu, vmin=0,vmax=0.2)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Space-Time)')
        plot.set_ylabel('Node index $j$ (Space-Time)')
        
        #x = np.arange((distance-1)/4 + (distance-3)/4,syn_num + (distance-1) ,(distance-1))
        x = np.arange(-0.5, syn_num, (distance-1)//2)
        #xtic = ['t'+str(int(i)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('t'+str(i//2))
        plot.set_xticks(ticks=x, labels=xtic)
        plot.set_yticks(x,xtic)
        return fig

    def draw_Correlation_matrix_9(self):
        distance = 9
        flag_length = 2
        initial_state = '-'
        syn_num = (distance-1)*(self.rounds+1)
        fig = Figure(figsize=(7, 6), dpi=150)

        tools = data_analysis(distance=distance, flag_length=flag_length)
        syn_data = tools.stab2syn( self.get_stab_data(distance,flag_length,initial_state), self.rounds )
        correlation_M_real = correlation_matrix(syn_data)
        p_matrix_real = correlation_M_real.p_matrix#[distance-1:syn_num,distance-1:syn_num]

        plot = fig.add_subplot(111)
        #colormapping = cm.ScalarMappable(cmap=cm.RdPu, norm=Normalize(vmin=0, vmax=0.3, clip=False))
        im = plot.imshow(p_matrix_real, cmap=cm.RdPu, vmin=0,vmax=0.2)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Space-Time)')
        plot.set_ylabel('Node index $j$ (Space-Time)')
        
        #x = np.arange((distance-1)/4 + (distance-3)/4,syn_num + (distance-1) ,(distance-1))
        x = np.arange(-0.5, syn_num, (distance-1)//2)
        #xtic = ['t'+str(int(i)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('t'+str(i//2))
        plot.set_xticks(ticks=x, labels=xtic)
        plot.set_yticks(x,xtic)
        return fig

    def draw_Correlation_matrix_invert_3(self):
        distance = 3
        flag_length = 2
        initial_state = '-'
        tools = data_analysis(distance=distance, flag_length=flag_length)
        fig = Figure(figsize=(7, 6), dpi=150)
        
        syn_num = (distance-1)*(self.rounds+1)
        syn_invert = tools.stab2syn_invert( self.get_stab_data(distance,flag_length,initial_state), self.rounds )

        correlation_M_real = correlation_matrix(syn_invert)
        p_matrix_real_invert = correlation_M_real.p_matrix
        
        plot = fig.add_subplot(111)
        
        im = plot.imshow(p_matrix_real_invert, cmap = cm.RdPu, vmin=0,vmax=0.2)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Time-Space)')
        plot.set_ylabel('Node index $j$ (Time-Space)')
        
        x = np.arange(-0.5,syn_num,(self.rounds+1)/2)
        #xtic = ['Sq'+str(int(i+1)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('sq'+str(i//2+1))
        plot.set_xticks(x,xtic)
        plot.set_yticks(x,xtic)
        #plt.title(f'd{distance}f{flag_length}ini{initial_state}_invert')
        return fig

    def draw_Correlation_matrix_invert_5(self):
        distance = 5
        flag_length = 2
        initial_state = '-'
        tools = data_analysis(distance=distance, flag_length=flag_length)
        fig = Figure(figsize=(7, 6), dpi=150)
        
        syn_num = (distance-1)*(self.rounds+1)
        syn_invert = tools.stab2syn_invert( self.get_stab_data(distance,flag_length,initial_state), self.rounds )

        correlation_M_real = correlation_matrix(syn_invert)
        p_matrix_real_invert = correlation_M_real.p_matrix
        
        plot = fig.add_subplot(111)
        
        im = plot.imshow(p_matrix_real_invert, cmap = cm.RdPu, vmin=0,vmax=0.2)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Time-Space)')
        plot.set_ylabel('Node index $j$ (Time-Space)')
        
        x = np.arange(-0.5,syn_num,(self.rounds+1)/2)
        #xtic = ['Sq'+str(int(i+1)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('sq'+str(i//2+1))
        plot.set_xticks(x,xtic)
        plot.set_yticks(x,xtic)
        #plt.title(f'd{distance}f{flag_length}ini{initial_state}_invert')
        return fig

    def draw_Correlation_matrix_invert_7(self):
        distance = 7
        flag_length = 2
        initial_state = '-'
        tools = data_analysis(distance=distance, flag_length=flag_length)
        fig = Figure(figsize=(7, 6), dpi=150)
        
        syn_num = (distance-1)*(self.rounds+1)
        syn_invert = tools.stab2syn_invert( self.get_stab_data(distance,flag_length,initial_state), self.rounds )

        correlation_M_real = correlation_matrix(syn_invert)
        p_matrix_real_invert = correlation_M_real.p_matrix
        
        plot = fig.add_subplot(111)
        
        im = plot.imshow(p_matrix_real_invert, cmap = cm.RdPu, vmin=0,vmax=0.2)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Time-Space)')
        plot.set_ylabel('Node index $j$ (Time-Space)')
        
        x = np.arange(-0.5,syn_num,(self.rounds+1)/2)
        #xtic = ['Sq'+str(int(i+1)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('sq'+str(i//2+1))
        plot.set_xticks(x,xtic)
        plot.set_yticks(x,xtic)
        #plt.title(f'd{distance}f{flag_length}ini{initial_state}_invert')
        return fig

    def draw_Correlation_matrix_invert_9(self):
        distance = 9
        flag_length = 2
        initial_state = '-'
        tools = data_analysis(distance=distance, flag_length=flag_length)
        fig = Figure(figsize=(7, 6), dpi=150)
        syn_num = (distance-1)*(self.rounds+1)
        syn_invert = tools.stab2syn_invert( self.get_stab_data(distance,flag_length,initial_state), self.rounds )

        correlation_M_real = correlation_matrix(syn_invert)
        p_matrix_real_invert = correlation_M_real.p_matrix
        
        plot = fig.add_subplot(111)
        
        im = plot.imshow(p_matrix_real_invert, cmap = cm.RdPu, vmin=0,vmax=0.2)
        fig.colorbar(im)
        fig.gca().invert_yaxis()
        plot.set_xlabel('Node index $i$ (Time-Space)')
        plot.set_ylabel('Node index $j$ (Time-Space)')
        
        x = np.arange(-0.5,syn_num,(self.rounds+1)/2)
        #xtic = ['Sq'+str(int(i+1)) for i in range(len(x))]
        xtic = []
        for i in range(len(x)):
            if i%2 == 0:
                xtic.append('')
            else:
                xtic.append('sq'+str(i//2+1))
        plot.set_xticks(x,xtic)
        plot.set_yticks(x,xtic)
        #plt.title(f'd{distance}f{flag_length}ini{initial_state}_invert')
        return fig

    def draw_defect_probability_real(self):
        distance = 9
        flag_length = 2
        initial_state = '-'
        syn_data = data_analysis(distance, flag_length).stab2syn(self.get_stab_data(distance,flag_length,initial_state), self.rounds)
        fig = Figure(figsize=(12, 8), dpi=150)
        
        data_det = { }
        shots = 50000
        for sample in syn_data:
            for r in np.arange(self.rounds+1):
                for data in np.arange(distance-1):
                    if (r,data) not in data_det.keys():
                        data_det[(r,data)] = sample[0][(distance-1)*r + data] * sample[-1] / (shots)
                    else:
                        data_det[(r,data)] += sample[0][(distance-1)*r + data] * sample[-1] / (shots)
        
        im = fig.add_subplot(111)
        for data in np.arange(distance-1):
            im.plot( np.arange(1,self.rounds+2), [ data_det[(r,data)] for r in np.arange(self.rounds+1)], color = 'grey')
        
        mean_data_det = [ np.mean([data_det[(r,data)] for data in np.arange(distance-1)]) for r in np.arange(self.rounds+1)]

        for data in np.arange(distance-1):
            im.plot( np.arange(1,self.rounds+2), mean_data_det, 'red' )
            
        im.set_xlabel('Syndrome rounds',fontsize = 15)
        im.set_ylabel('Detection probability',fontsize = 15)
        im.set_xticks(range(1,12))
        return fig

    def draw_Logical_error_rate_hardware_Z1_log(self):

        rounds_list = range( 1, 11 )
        distances = [ 3,5,7,9 ]
        fig = Figure(figsize=(10, 8), dpi=150)

        color_list_f = [ 'blue', 'green', 'red', 'grey' ]
        style = ['--s','--^','--o','--p']

        # data_path = f"./Paper_data2/correct_excel/"
        data_path = f'./Main_result/'
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
            
        shots = 50000
        
        flag_length = 2
            
        plot = fig.add_subplot(111)
        c = 0
        for distance in distances:
            
            # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
            file_name = f'distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                
            file = data_path + file_name
            
            df = pd.read_excel( file,sheet_name = ['real result' ], index_col = 0 )
            correct_nums = np.array( df[ 'real result' ].values.tolist( ) )
            correct_nums = 1 - correct_nums / (shots)
            
            # # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
            p_err_arr = np.array(correct_nums[:,1] )

            plot.plot( rounds_list, p_err_arr, style[c], color = color_list_f[c] ,label = rf'$[[{distance},1,{distance}]]$' +rf'$_{{f={flag_length}}}$', linewidth = 2, markersize = 10 )
            c += 1
        plot.semilogy( )
        plot.set_xlabel( "Syndrome extraction cycles", fontsize = 20 )
        plot.set_ylabel( "logical error rate $(p_L)$", fontsize = 20 )
        plot.legend( loc = "lower right", fontsize = 10 )
        return fig

    def draw_Logical_error_rate_hardware_Z1_linear(self):

        rounds_list = range( 1, 11 )
        distances = [ 3,5,7,9 ]
        fig = Figure(figsize=(10, 8), dpi=150)

        color_list_f = [ 'blue', 'green', 'red', 'grey' ]
        style = ['--s','--^','--o','--p']

        # data_path = f"./Paper_data2/correct_excel/"
        data_path = f'./Main_result/'
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
            
        shots = 50000
        
        flag_length = 2
            
        plot = fig.add_subplot(111)
        c = 0
        for distance in distances:
            
            # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
            file_name = f'distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                
            file = data_path + file_name
            
            df = pd.read_excel( file,sheet_name = ['real result' ], index_col = 0 )
            correct_nums = np.array( df[ 'real result' ].values.tolist( ) )
            correct_nums = 1 - correct_nums / (shots)
            
            # # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
            p_err_arr = np.array(correct_nums[:,1])

            plot.plot( rounds_list, p_err_arr, style[c], color = color_list_f[c] ,label = rf'$[[{distance},1,{distance}]]$' +rf'$_{{f={flag_length}}}$', linewidth = 2, markersize = 10 )
            c += 1

        plot.set_xlabel( "Syndrome extraction cycles", fontsize = 20 )
        plot.set_ylabel( "logical error rate $(p_L)$", fontsize = 20 )
        plot.legend( loc = "lower right", fontsize = 10 )
        return fig

    def draw_Logical_error_rate_hardware_X1_log(self):

        rounds_list = range( 1, 11 )
        distances = [ 3,5,7,9 ]
        fig = Figure(figsize=(10, 8), dpi=150)

        color_list_f = [ 'blue', 'green', 'red', 'grey' ]
        style = ['--s','--^','--o','--p']

        # data_path = f"./Paper_data2/correct_excel/"
        data_path = f'./Main_result/'
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
            
        shots = 50000
        
        flag_length = 2
            
        plot = fig.add_subplot(111)
        c = 0
        for distance in distances:
            
            # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
            file_name = f'distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                
            file = data_path + file_name
            
            df = pd.read_excel( file,sheet_name = ['real result' ], index_col = 0 )
            correct_nums = np.array( df[ 'real result' ].values.tolist( ) )
            correct_nums = 1 - correct_nums / (shots)
            
            # # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
            p_err_arr = np.array(correct_nums[:,3])

            plot.plot( rounds_list, p_err_arr, style[c], color = color_list_f[c] ,label = rf'$[[{distance},1,{distance}]]$' +rf'$_{{f={flag_length}}}$', linewidth = 2, markersize = 10 )
            c += 1
        plot.semilogy( )
        plot.set_xlabel( "Syndrome extraction cycles", fontsize = 20 )
        plot.set_ylabel( "logical error rate $(p_L)$", fontsize = 20 )
        plot.legend( loc = "lower right", fontsize = 10 )
        return fig

    def draw_Logical_error_rate_hardware_X1_linear(self):

        rounds_list = range( 1, 11 )
        distances = [ 3,5,7,9 ]
        fig = Figure(figsize=(10, 8), dpi=150)

        color_list_f = [ 'blue', 'green', 'red', 'grey' ]
        style = ['--s','--^','--o','--p']

        # data_path = f"./Paper_data2/correct_excel/"
        data_path = f'./Main_result/'
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
            
        shots = 50000
        
        flag_length = 2
            
        plot = fig.add_subplot(111)
        c = 0
        for distance in distances:
            
            # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
            file_name = f'distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                
            file = data_path + file_name
            
            df = pd.read_excel( file,sheet_name = ['real result' ], index_col = 0 )
            correct_nums = np.array( df[ 'real result' ].values.tolist( ) )
            correct_nums = 1 - correct_nums / (shots)
            
            # # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
            p_err_arr = np.array(correct_nums[:,3])

            plot.plot( rounds_list, p_err_arr, style[c], color = color_list_f[c] ,label = rf'$[[{distance},1,{distance}]]$' +rf'$_{{f={flag_length}}}$', linewidth = 2, markersize = 10 )
            c += 1

        plot.set_xlabel( "Syndrome extraction cycles", fontsize = 20 )
        plot.set_ylabel( "logical error rate $(p_L)$", fontsize = 20 )
        plot.legend( loc = "lower right", fontsize = 10 )
        return fig
    
    def draw_Sample_probability_hardware(self):

        data = {}
        data_1 = {}
        data_2 = {}
        rounds = 10
        fig = Figure(figsize=(16, 5), dpi=150)
        
        for flag_length in [0,1,2]:
            for distance in [3,5,7,9]:
                
                syn_num_dic = {}
                
                for initial_state in ['0','1','+','-']:
                    
                    stab_path = f'./Results/'
                    file_name = f'distance-{distance}_rounds-{rounds}_flag_length-{flag_length}_initial_state-{initial_state}_index-0'
                    stab_data = pd.read_csv( stab_path + file_name + '.csv', index_col = 0, skiprows = [ 0 ] ).values.tolist( )
                    syn_data = data_analysis(distance, flag_length).stab2syn(stab_data, rounds)

                    for syn in syn_data:
                        if np.sum(syn[0]) not in syn_num_dic.keys():
                            syn_num_dic[np.sum(syn[0])] = syn[-1]
                        else:
                            syn_num_dic[np.sum(syn[0])] += syn[-1]
                        
                        if np.sum(syn[0]) == 0:
                            if initial_state in ['0','+']:
                                if syn[1] == 1:
                                    if 'loc_err' not in syn_num_dic.keys():
                                        syn_num_dic['loc_err'] = syn[-1]
                                    else:
                                        syn_num_dic['loc_err'] += syn[-1]
                            else:
                                if syn[1] == 0:
                                    if 'loc_err' not in syn_num_dic.keys():
                                        syn_num_dic['loc_err'] = syn[-1]
                                    else:
                                        syn_num_dic['loc_err'] += syn[-1]
                if flag_length == 0:
                    data[distance] = syn_num_dic
                elif flag_length ==1:
                    data_1[distance] = syn_num_dic
                elif flag_length ==2:
                    data_2[distance] = syn_num_dic
        
        flag_lengths = [ 0, 1, 2 ]
        distances = [ 3, 5, 7, 9 ]

        dic = {}
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
        for flag_length in flag_lengths:
            
            shots = 50000
            total_shot = shots*4
            
            # 각 케이스별로 엑셀 파일의 sheet를 선택해 그 수정 개수를 가져와 오류율을 계산한다.
            for sheet in [ 'real p']:#, 'real p' ]:

                for distance in distances:
                    
                    p_err_list = []
                        
                    # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
                    file_name = f'./Main_result/distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                    df = pd.read_excel( file_name,sheet_name = [ sheet + ' result' ], index_col = 0 )
                    correct_nums = np.array( df[ sheet + ' result' ].values.tolist( ) )
                    
                    # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
                    p_err = [ np.sum(correct_nums[ i, : ]) for i in range( len( correct_nums ) ) ]
                    p_err_list.append( p_err )
                        
                    p_err_mean = [ np.sum(np.array(p_err_list)[:,i]) for i in range(1) ][-1]
                    
                    defect_num = 0
                    
                    if flag_length == 0:
                        syn_dic = data[distance]
                    elif flag_length ==1:
                        syn_dic = data_1[distance]
                    elif flag_length ==2:
                        syn_dic = data_2[distance]
                        
                    for syn_num in syn_dic.keys():
                        if syn_num != 0:
                            defect_num += syn_dic[syn_num]
                    
                    if 0 in syn_dic.keys():
                        syn0 = syn_dic[0]
                    else:
                        syn0 = 0
                    dic[(distance,flag_length)] = [syn0/total_shot, (p_err_mean - syn0)/total_shot, (defect_num-(p_err_mean - syn0))/total_shot]

        plot = fig.add_subplot(111)
        x = np.arange(12)
        tick = []
        values_list = []
        for distance in distances:
            for flag_length in flag_lengths:
                if flag_length == 0:
                    tick.append(f'$[[{distance},1,{distance}]]$' +r'$_{f={0}}$')
                elif flag_length == 1:
                    tick.append(f'$[[{distance},1,{distance}]]$' +r'$_{f={1}}$')
                else:
                    tick.append(f'$[[{distance},1,{distance}]]$' +r'$_{f={2}}$')
                    
                values_list.append( dic[(distance,flag_length)])
                
        y1 = np.array(values_list)[:,0]
        y2 = np.array(values_list)[:,1]
        y3 = np.array(values_list)[:,2]

        plot.bar(x, y1, color='darkblue',label='Non-defect')
        plot.bar(x, y2, color='darkgreen',bottom = y1,label='Corrected')
        plot.bar(x, y3, color='crimson',bottom = y1+y2,label = 'Non-corrected')
        plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True, ncol=5, fontsize = 15)
        plot.set_xticks(x)
        plot.set_xticklabels(tick, fontsize = 7)
        plot.set_xlabel('Repetition structure ' + r'$[[n,k,d]]_f$',fontsize =15)
        plot.set_ylabel('Sample probability',fontsize =15)
        return fig
     
    def draw_Sample_probability_sample(self):

        data = {}
        data_1 = {}
        data_2 = {}
        rounds = 10
        
        fig = Figure(figsize=(16, 5), dpi=150)
        for flag_length in [0,1,2]:
            for distance in [3,5,7,9]:
                
                syn_num_dic = {}
                
                for initial_state in ['0','1','+','-']:
                    
                    stab_path = f'./Results/'
                    file_name = f'distance-{distance}_rounds-{rounds}_flag_length-{flag_length}_initial_state-{initial_state}_index-0'
                    stab_data = pd.read_csv( stab_path + file_name + '.csv', index_col = 0, skiprows = [ 0 ] ).values.tolist( )
                    syn_data = data_analysis(distance, flag_length).stab2syn(stab_data, rounds)

                    for syn in syn_data:
                        if np.sum(syn[0]) not in syn_num_dic.keys():
                            syn_num_dic[np.sum(syn[0])] = syn[-1]
                        else:
                            syn_num_dic[np.sum(syn[0])] += syn[-1]
                        
                        if np.sum(syn[0]) == 0:
                            if initial_state in ['0','+']:
                                if syn[1] == 1:
                                    if 'loc_err' not in syn_num_dic.keys():
                                        syn_num_dic['loc_err'] = syn[-1]
                                    else:
                                        syn_num_dic['loc_err'] += syn[-1]
                            else:
                                if syn[1] == 0:
                                    if 'loc_err' not in syn_num_dic.keys():
                                        syn_num_dic['loc_err'] = syn[-1]
                                    else:
                                        syn_num_dic['loc_err'] += syn[-1]
                if flag_length == 0:
                    data[distance] = syn_num_dic
                elif flag_length ==1:
                    data_1[distance] = syn_num_dic
                elif flag_length ==2:
                    data_2[distance] = syn_num_dic
        
        flag_lengths = [ 0, 1, 2 ]
        distances = [ 3, 5, 7, 9 ]

        dic = {}
        # 모든 파라미터에 대응해 stabilizer measurement 라운드별 오류율 값들을 그래프로 나타낸다.
        for flag_length in flag_lengths:
            
            shots = 50000
            total_shot = shots*4
            
            # 각 케이스별로 엑셀 파일의 sheet를 선택해 그 수정 개수를 가져와 오류율을 계산한다.
            for sheet in [ 'real']:#, 'real p' ]:

                for distance in distances:
                    
                    p_err_list = []
                        
                    # 수정 개수가 저장된 엑셀표에 접근하도록 하는 file_name을 지정한다.
                    file_name = f'./Main_result/distance-{ distance }_flag_length-{ flag_length }_correct_num.xlsx'
                    df = pd.read_excel( file_name,sheet_name = [ sheet + ' result' ], index_col = 0 )
                    correct_nums = np.array( df[ sheet + ' result' ].values.tolist( ) )
                    
                    # 초기 양자 상태들에 대한 전체 평균을 구하여 논리 큐빗 오류율을 채택한다.
                    p_err = [ np.sum(correct_nums[ i, : ]) for i in range( len( correct_nums ) ) ]
                    p_err_list.append( p_err )
                        
                    p_err_mean = [ np.sum(np.array(p_err_list)[:,i]) for i in range(1) ][-1]
                    
                    defect_num = 0
                    
                    if flag_length == 0:
                        syn_dic = data[distance]
                    elif flag_length ==1:
                        syn_dic = data_1[distance]
                    elif flag_length ==2:
                        syn_dic = data_2[distance]
                        
                    for syn_num in syn_dic.keys():
                        if syn_num != 0:
                            defect_num += syn_dic[syn_num]
                    
                    if 0 in syn_dic.keys():
                        syn0 = syn_dic[0]
                    else:
                        syn0 = 0
                    dic[(distance,flag_length)] = [syn0/total_shot, (p_err_mean - syn0)/total_shot, (defect_num-(p_err_mean - syn0))/total_shot]

        plot = fig.add_subplot(111)
        x = np.arange(12)
        tick = []
        values_list = []
        for distance in distances:
            for flag_length in flag_lengths:
                if flag_length == 0:
                    tick.append(f'$[[{distance},1,{distance}]]$' +r'$_{f={0}}$')
                elif flag_length == 1:
                    tick.append(f'$[[{distance},1,{distance}]]$' +r'$_{f={1}}$')
                else:
                    tick.append(f'$[[{distance},1,{distance}]]$' +r'$_{f={2}}$')
                    
                values_list.append( dic[(distance,flag_length)])
                
        y1 = np.array(values_list)[:,0]
        y2 = np.array(values_list)[:,1]
        y3 = np.array(values_list)[:,2]

        plot.bar(x, y1, color='darkblue',label='Non-defect')
        plot.bar(x, y2, color='darkgreen',bottom = y1,label='Corrected')
        plot.bar(x, y3, color='crimson',bottom = y1+y2,label = 'Non-corrected')
        plot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True, ncol=5, fontsize = 15)
        plot.set_xticks(x)
        plot.set_xticklabels(tick, fontsize = 7)
        plot.set_xlabel('Repetition structure ' + r'$[[n,k,d]]_f$',fontsize =10)
        plot.set_ylabel('Sample probability',fontsize =15)
        return fig
 
