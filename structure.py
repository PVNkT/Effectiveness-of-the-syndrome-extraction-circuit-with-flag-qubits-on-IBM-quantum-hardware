import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import structure_utils

# Structure 클래스로 주어진 distance와 data 큐빗과 syndrome 큐빗 사이의 거리 flag length를 입력값으로 받는다.
# 그 결과로 Repetition code의 data 큐빗, syndrome 큐빗, flag 큐빗들의 상대적인 위치와 큐빗 번호 dictionary를 결과로 내놓는다.
class Structure_1d:
    
    # 본 python 코드에서 고려하는 Logical qubit은 Repetition code이다.
    # 클래스 내의 distance와 flag length를 입력값으로 사용한다.
    def __init__( self, distance, flag_length ):

        self.distance = distance
        self.flag_length = flag_length
        
    # 이때, 왼쪽에서 부터 큐빗 번호를 0번으로 지정하기 위해 qubit 리스트를 sorting 해주는 알고리즘이 필요하다.
    def sorting_qubit( self, qubit_list ):
    
        qubit_list.sort( key = lambda v : v.real )

        return qubit_list

    # 가장 왼쪽 큐빗을 기준으로 다른 큐빗들의 위치가 일렬로 배치된다.
    # 주어진 distance와 flag length를 가지고 계산한 data 큐빗들의 위치를 나타내기 위한 함수이다.
    def qubit_data( self ):
    
        distance = self.distance
        flag_length = self.flag_length
        
        # distance에 해당하는 data 큐빗 개수를 지정하며 이들의 상대적인 위치는 arange를 사용하여 계산한다.
        # data qubit이 양 끝에 존재하고 총 distance * ( 2*flag_length + 2 )-(2*flag_length + 1)개의 qubit이 있다.
        datas = np.arange( 0, distance * ( 2*flag_length + 2 )-(2*flag_length + 1), 2*flag_length + 2 )

        return list(datas)

    # 오류 정정 코드 내에 위치하는 syndrome 그룹으로 syn의 위치를 data 큐빗들의 위치로 얻는다.
    def qubit_syn( self, datas ):
    
        syns = [ ]
        
        # syn의 위치는 이웃한 두 data 큐빗의 가운데에 위치한다.
        for data1,data2 in zip(datas[:],datas[1:]):
            syns.append((data1+data2)/2)
    
        return syns

    # Logical 큐빗의 구조가 1차원 구조로 syndrome 큐빗과 data 큐빗 사이에 flag 큐빗들이 추가되어야 한다.
    def qubit_flag( self, syns ):
        
        flag_length = self.flag_length
        
        flags = [ ]
        
        # flag 큐빗들은 syn 큐빗을 중심으로 양쪽에 놓여 간접적으로 data 큐빗과 syn 큐빗을 이어준다.
        for syn in syns:
            for delta in np.arange(1,flag_length+1):
                flags.append(syn-delta)
                flags.append(syn+delta)

        return flags

    # 큐빗 종류들을 모아 하나의 structure를 나타내는 qubit들의 위치 및 역할을 구분하는 함수를 정의한다.
    def structure_dics( self ):
    
        flag_length = self.flag_length
        
        # Logical 큐빗 구조에 mapping한 data 큐빗, syndrome 큐빗, flag 큐빗들의 위치와 qubit 번호를 위한 Dictionary를 만든다.
        data = self.qubit_data( )
        syn = self.qubit_syn( data )
        
        # flag 큐빗의 길이가 0이면 flag 큐빗이 없는 경우이다.
        if 0 < flag_length:
            flag = self.qubit_flag( syn )
        else:
            flag = [ ]
        
        # 여러 큐빗 타입들을 모아 위치 순서대로 정리한다.
        qubits_list = self.sorting_qubit( data + syn + flag )
        
        # 큐빗 위치 및 번호를 연결하는 Dictionary를 구성한다.
        qubits_dic = { 'data_qubit' : [ ], 'syn_qubit' : [ ], 'flag_qubit' : [ ] }
        qubits_num2loc = { }
        qubits_loc2num = { }
        
        num = 0
        for node in qubits_list:
        
            if node in data:
                qubits_dic[ 'data_qubit' ].append( num )
                qubits_loc2num[ node ] = num
                qubits_num2loc[ num ] = node

            elif node in syn:
                qubits_dic[ 'syn_qubit' ].append( num )
                qubits_loc2num[ node ] = num
                qubits_num2loc[ num ] = node

            elif node in flag:
                qubits_dic[ 'flag_qubit' ].append( num )
                qubits_loc2num[ node ] = num
                qubits_num2loc[ num ] = node

            num += 1
        
        # structure에서 고려해야할 모든 data 큐빗, syndrome 큐빗, flag 큐빗들의 상대적인 위치 및 번호들을 구성할 수 있다.
        # qubits_num2loc : 큐빗 번호에서 위치
        # qubits_loc2num : 큐빗 위치에서 번호
        # qubit_dic : 큐빗 타입에 따른 큐빗 번호 분리
        return qubits_num2loc, qubits_loc2num, qubits_dic

    # structure 함수에서 고려한 큐빗들을 한데 모아 networkx를 통해 그래프를 그린다.
    def draw_qubits( self ):
        
        flag_length = self.flag_length
        
        qubits_num2loc, qubits_loc2num, qubits_dic = self.structure_dics( )

        # 큐빗들은 모두 점으로 나타내며, 노드의 색깔로 큐빗들의 역할을 유추할 수 있도록 한다.
        total_color_map = [ '#6495ED', '#00C957', '#FF6347' ]
        color_map = [ ]
        label = { }

        G = nx.Graph( )
        
        # 2차원 직각 좌표계에 큐빗들를 배치하여 상대적인 위치를 시각적으로 나타낸다.
        for num in range( len( list( qubits_num2loc.keys( ) ) ) ):
            
            # 큐빗의 존재를 그래프의 노드로 나타낸다.
            node = qubits_num2loc[ num ]

            real = node.real
            image = node.imag
            G.add_node( ( real, image ), pos = ( real, image ))
            
            # 큐빗의 타입에 따라 노드의 색을 달리한다.
            # data 큐빗 : 파란색
            # syndrome 큐빗 : 초록색
            # flag 큐빗 : 빨간색
            if num in qubits_dic[ 'data_qubit' ]:
                color_num = 0
            elif num in qubits_dic[ 'syn_qubit' ]:
                color_num = 1
            elif num in qubits_dic[ 'flag_qubit' ]:
                color_num = 2

            color_map.append( total_color_map[ color_num ] )
            label[ ( real, image ) ] = num


        # networkx 패키지로 논리 큐빗을 구성하는 큐빗들을 그림으로 나타낸다.
        plt.figure( figsize = ( 14 + 2 * flag_length, 6 ) )

        pos = nx.get_node_attributes( G, 'pos' )

        nx.draw_networkx_nodes( G, pos, label = label, node_color = color_map, node_size = 300 )
        nx.draw_networkx_labels( G, pos, labels = label )

        plt.show()

class locq_structure(structure_utils):

  # class locq_structure의 input 값인 distance_row, distance_col, arc를 class의 변수로 지정한다.
  def __init__(self, distance_row, distance_col, arc = 'Lattice'):
    super.__init__(distance_row, distance_col, arc)
    self.distance_row = distance_row
    self.distance_col = distance_col
    self.arc = arc

  # locq_structure에서 정의한 함수들을 모아 하나의 structure를 나타내는 qubit들의 위치 및 역할을 구분하는 함수를 정의한다.
  def structure( self ):

    total_q = []
    # Logical qubit 구조에 mapping한 data qubit, syndrome qubit, flag qubit들의 위치와 qubit 번호를 위한 Dictionary를 만든다.
    # 이때 data, syn_1, syn_2, flag 순으로 qubit 번호를 붙여준다.
    qubit_num_data, data_num = self.qubit_data( )
    qubit_num_syn_1, syn_1_num = getattr(self, 'qubit_syn_1')( data_num  )
    qubit_num_syn_2, syn_2_num = getattr(self, 'qubit_syn_2')( data_num  )
    qubit_num_flag = self.qubit_flag( qubit_num_syn_1.keys(), qubit_num_syn_2.keys(), syn_2_num)
    
    for q_type in [qubit_num_data,qubit_num_syn_1,qubit_num_syn_2,qubit_num_flag]:
      for qubit in q_type.keys():
        total_q.append(qubit)
        
    # structure에서 고려해야할 모든 data qubit, syndrome qubit, flag qubit들의 qubit의 상대적인 위치 및 번호들을 구성할 수 있다.
    return total_q

  # structure 함수에서 고려한 qubit들을 한데 모아 networkx를 통해 그림을 그려 그 결과를 확인한다.
  def draw_qubits( self, structure_list ):
    
    d = structure_list[0]
    flag_length = structure_list[1]
    real_qubit_index = structure_list[2]
    
    qubit_num_data, qubit_num_syn_1, qubit_num_syn_2, qubit_num_flag = self.structure()
    
    # qubit들은 모두 점으로 나타내며, 노드의 색깔로 qubit들의 역할을 유추할 수 있도록 한다.
    qubits = [ qubit_num_data, qubit_num_syn_1, qubit_num_syn_2, qubit_num_flag ]
    color_map = [ ]
    label = { }
    
    G = nx.Graph()  
    for num in range( len( qubits )):
      if num not in structure_list:
        for node in list( qubits[ num ].keys() ):
          
          real = node.real
          image = node.imag
          G.add_node( ( real, image ), pos = ( real, image ))
          color_map.append( 'grey')
          label[ ( real, image )] = qubits[ num ][ node ]

    plt.figure( figsize = ( 10, 8 ))

    pos = nx.get_node_attributes( G, 'pos' )

    nx.draw_networkx_nodes( G, pos, label = label, node_color = color_map, node_size = 300 )
    nx.draw_networkx_labels( G, pos, labels= label )
    
    plt.show()



















