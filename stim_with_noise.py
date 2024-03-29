
import stim
import numpy as np
import csv
import datetime
# 논리 큐빗의 구조 및 양자 회로 구성에 대한 정보를 얻기 위해 다음 파일들을 import 한다.
from structure import Structure_1d
from utils import encoder_utils

# 실제 양자 컴퓨터의 오류율 값들을 불러와 오류 모델의 파라미터로 사용한다.
class stim_noise_model:
    """
    실제 양자 컴퓨터의 calibration data를 기반으로 stim circuit을 구성하기 위해서 대응되는 gate 오류율을 설정한다.
    IBM eagle processor는 ECR, x, sx를 기본 gate로 사용하기 때문에 이를 H, CNOT, CZ gate에 대응되도록 오류율을 계산한다.
    2 qubit gate의 경우에는 ECR gate의 앞 뒤에 single qubit gate에 의한 오류율이 추가적으로 들어가게 된다.
    """
    # 전체 큐빗들의 번호와 하드웨어 및 선택한 큐빗들의 리스트를 입력값으로 사용한다.
    def __init__( self, total_qubit, datetime, real_qubit_index ):
        
        # 입력값들로 클래스 내의 변수들을 지정해준다.
        self.total_qubit_num = len( total_qubit )
        self.total_qubit = total_qubit
        self.real_qubit_index = real_qubit_index
        self.file_path = datetime
        
        # 고려하는 오류의 종류들의 dictionary를 구성한다.
        self.readout_dic = self.error_from_csv('readout')
        self.reset_dic = self.error_from_csv('reset')
        self.id_dic = self.error_from_csv('id')
        self.sx_dic = self.error_from_csv('sx')
        self.x_dic = self.error_from_csv('x')
        self.ecr_dic = self.error_from_csv('ecr')
    
    # 실험이 시작한 시점에 저장된 오류 calibration 값을 가져와 이를 통해서 오류 모델을 만든다.    
    def error_from_csv(self, file_name):
        # 파일이 저장된 경로
        file_path = self.file_path + file_name + '_properties.csv'
        data_dict = {}
        # 해당되는 gate에 대응되는 csv 파일을 연다.
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile) 
            column_headers = next(reader)
            # 각 qubit들에 대해서 dictionary의 key를 만든다.
            for header in column_headers:
                data_dict[header] = []
            # 각 qubit들에 대한 정보를 대응되는 key에 저장한다.
            for row in reader:
                for header, value in zip(column_headers, row):
                    data_dict[header].append(float(value))
        # 저장된 dictionary를 반환한다.
        return data_dict
     
    # 시뮬레이션의 큐빗 번호와 대응되는 실제 양자 컴퓨터의 큐빗 번호를 일치 시키기 위한 함수이다.
    def sim2real_index_swap( self, qubit_num_list ):
        
        # 시뮬레이션에서 고려하는 큐빗 번호는 0번 부터 시작한다.
        sim_qubit_num_list = range( self.total_qubit_num )
        
        # 실제 양자 컴퓨터의 큐빗 번호들의 순서들을 리스트로 나열한다.
        real_qubit_num_list = self.real_qubit_index
        
        # 시뮬레이션 내의 큐빗 번호와 실제 하드웨어 내의 큐빗 번호에 대한 1대1 대응 관계를 구성한다.
        sim2real = dict( zip( sim_qubit_num_list, real_qubit_num_list ) )
        
        # 시뮬레이션의 큐빗 리스트에 대응되는 하드웨어의 큐빗 리스트로 변환시킨다.
        return [ sim2real[ qubit_num ] for qubit_num in qubit_num_list ]
    
    # 큐빗을 가만히 놔두면, depolarization 오류에 취약해진다.
    # 시간에 대한 변화 오류를 추가한다.
    def idle_err( self, circuit, qubit_num_list ):
        # 실제 qubit의 오류율을 적용하기 위해서 실제 qubit의 번호를 얻는다.
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        # single qubit deporaization을 적용
        # 현재 gate가 가해지고 있지 않은 qubit들의 list를 받으면 그 qubit들에 대해서 idle 오류를 적용한다.
        for num in range( len( real_qubit_list ) ):
            # 1 qubit에 대한 depolarizing 오류 모델을 적용한다. 그 qubit에 대응되는 오류가 모두 depolarizing 오류라고 가정하고 오류 모델을 적용한다.
            #circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.id_dic[ (f'({real_qubit_list[ num ]},)') ] )
            circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.id_dic[ f'{real_qubit_list[ num ]}'] )

    # 전체 qubit들 중 gate가 가해진 qubit들을 입력 받으면 gate가 가해지지 않은 qubit을 확인하고 idle 오류를 적용한다.
    def apply_idle_err_to_idle_qubit(self, circuit, active_qubits):
        # 측정하지 않는 큐빗들에 대해서 idle 오류를 추가한다.
        err_list = [ qubit for qubit in self.total_qubit if qubit not in active_qubits ]
        self.idle_err( circuit, err_list )

    # 측정 오류를 추가한다.
    def readout_err( self, circuit, qubit_num_list ):    
        # 시뮬레이션 큐빗 번호를 하드웨어 큐빗 번호로 변환한 뒤에 측정 오류를 회로에 추가한다.
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        # 측정 오류는 고전적인 flip만 일어나기 때문에 X error를 추가
        # 양자 gate들은 depolaizing 오류가 일어난다고 가정하지만 measurement의 경우에는 0 상태가 1로 측정되거나 1상태가 0으로 측정되는 경우만 존재하기 때문에,
        # 고전적인 bitflip만을 고려하면 된다. 따라서 단순한 X error만을 고려하면 된다.
        for num in range( len( real_qubit_list ) ):
            #circuit.append_operation( "X_ERROR", qubit_num_list[ num ], self.readout_dic[ (f'({real_qubit_list[ num ]},)') ] )
            circuit.append_operation( "X_ERROR", qubit_num_list[ num ], self.readout_dic[ f'{real_qubit_list[ num ]}' ] )

        # 측정하지 않는 큐빗들에 대해서 idle 오류를 추가한다.
        #self.apply_idle_err_to_idle_qubit(circuit, qubit_num_list)
        
    # X 오류를 추가한다.
    def x_err( self, circuit, qubit_num_list ):
        
        # 시뮬레이션 큐빗 번호를 하드웨어 큐빗 번호로 변환한 뒤에 측정 오류를 회로에 추가한다.
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        # x gate를 적용할 때 depolazization 오류가 생기는 경우
        for num in range( len( real_qubit_list ) ):
            # x gate의 경우 IBM hardware의 기본 gate로 사용되기 때문에 오류율을 그대로 적용할 수 있다.
            # 일어나는 모든 종류의 오류를 depolizing 오류라고 가정한다.
            #circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.x_dic[ (f'({real_qubit_list[ num ]},)' )] )
            circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.x_dic[ f'{real_qubit_list[ num ]}'] )
        
        # 측정하지 않는 큐빗들에 대해서는 idle 오류를 추가한다.
        self.apply_idle_err_to_idle_qubit(circuit, qubit_num_list)
        
    # 하다마드 연산자에 대한 오류를 회로에 추가한다.
    def hadamard_err( self, circuit, qubit_num_list ):
        
        # IBM 양자 하드웨어에서 하다마드 연산자 오류는 square root x의 오류 값과 동일하다.
        # Hadamard gate의 경우 R_z SX R_z의 형태로 transpile되게 되는데
        # IBM hardware에서는 R_z gate는 소프트웨어 적인 처리이기 때문에 오류가 발생하지 않는다.
        # 따라서 Hadamard gate의 오류율은 SX gate의 오류율과 동일하다.
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        # 단일 qubit에 대한 depolarization 오류
        for num in range( len( real_qubit_list ) ):
            #circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[ f'({real_qubit_list[ num ]},)' ] )
            circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[f'{real_qubit_list[ num ]}'] )
        
        # 하다마드 연산자를 수행하지 않는 큐빗들에 idle 오류를 추가한다.
        self.apply_idle_err_to_idle_qubit(circuit, qubit_num_list)
    
    # 측정 된 큐빗들의 양자 상태를 다시 바닥 상태로 만든 뒤의 0 상태 준비 오류를 추가한다.
    def reset_err( self, circuit, qubit_num_list ):
        # 주어진 qubit들을 실제 양자 컴퓨터의 번호로 바꾼다.
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        # reset gate의 오류는 측정 오류와 x gate 오류로 나누어 질 수 있다. (오류율을 저장하는 단계에서 계산)
        # 기본적으로 repetition code에서는 측정뒤에 reset gate를 사용하게 된다.
        # 따라서 일반적으로 circuit을 구성하게 될 경우 measurement를 두번 적용하게 된다.
        # qiskit의 ResetAfterMeasureSimplification을 적용하게 되면 앞선 measurement 결과를 통해서 reset gate를 적용할 수 있다.
        # 이 경우에는 단순히 x gate에 대한 오류만 발생하게 된다. (이 부분은 save_property에서 고려하여 계산)
        # 따라서 reset gate의 오류는 x gate의 오류와 동일하다.
        # 그런데 measurement 이후에 양자 상태는 0 또는 1 상태로 붕괴하기 때문에 생기는 오류는 X 오류만 존재하게 된다. 
        for num in range( len( real_qubit_list ) ):
            #circuit.append_operation( "X_ERROR", qubit_num_list[ num ], self.reset_dic[ (f'({real_qubit_list[ num ]},)' )] )
            circuit.append_operation( "X_ERROR", qubit_num_list[ num ], self.x_dic[f'{real_qubit_list[ num ]}'] )
        # 만약 ResetAfterMeasureSimplification을 적용하지 않으면 measurement오류와 x 오류의 확률을 계산하여 오류가 가해져야 한다.
        # 이 부분을 고려하여 reset_dic가 계산되었다. 이 경우 measurement error가 고전적인 오류이기 때문에 X_error를 적용한다.
        """
        for num in range( len( real_qubit_list ) ):
            circuit.append_operation( "X_ERROR", qubit_num_list[ num ], self.reset_dic[ f'({real_qubit_list[ num ]},)'] )
            #circuit.append_operation( "X_ERROR", qubit_num_list[ num ], self.reset_dic[f'{real_qubit_list[ num ]}'] )
        """
        # 초기화하지 않는 큐빗들에 대해서는 idle 오류를 추가한다.
        #self.apply_idle_err_to_idle_qubit(circuit, qubit_num_list)
    
    # IBM 양자 컴퓨터에서 2 큐빗 연산자의 구성에 따라 오류 모델을 근사하여 CNOT 오류 모델을 구성한다.
    def CNOT_err_front( self, circuit, qubit_num_list ):
        # ECR gate를 basis gate로 사용하는 경우에 CNOT 오류는 여러 gate의 오류가 일어날 확률을 통해서 계산된다.
        # control과 target 큐빗에 square root x 오류를 추가한다.
        # 실제 양자 컴퓨터에서 CNOT gate는 ECR gate의 앞뒤에 각각의 qubit에 R_z SX R_z gate가 더해진 형태로 표현된다.
        # 그런데 R_z gate는 소프트웨어적인 처리이기 때문에 실제로 오류가 생기는 것은 SX gate뿐이다.
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        for num in range( len( real_qubit_list ) ):
            #circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[ f'({real_qubit_list[ num ]},)' ] )
            circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[ f'{real_qubit_list[ num ]}'] )
        
    # 두 큐빗 연산자에 대한 오류 모델로 2개의 큐빗으로 짝을 지어 Depolarization 오류를 추가한다.
    # 이는 하드웨어의 종류에 상관없이 적용된다.
    def CNOT_err_middle( self, circuit, qubit_num_list ):
        # ECR gate에 의해서 일어나는 2 qubit error
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        for num in np.arange( 0, len( real_qubit_list ), 2 ):
            # two qubit gate에 의해서 생기는 depolarizing error는 총 15 종류의 오류가 생길 수 있다. 이를 gate에 걸리는 2개의 qubit에 적용한다.
            circuit.append_operation( "DEPOLARIZE2", [ qubit_num_list[ num ], qubit_num_list[ num + 1 ] ],
                                                        self.ecr_dic[ ( f'({real_qubit_list[ num ]}, {real_qubit_list[ num + 1 ]})' ) ] )
        
    # CNOT 연산자를 구성하기 위해 2 큐빗 연산자 수행 후의 다른 연산자에 대한 오류를 추가한다.
    def CNOT_err_back( self, circuit, qubit_num_list ):
        #  ECR gate를 사용하는 경우, control과 target 큐빗에 square root x 오류를 추가한다.
        # 실제 양자 컴퓨터에서 CNOT gate는 ECR gate의 앞뒤에 각각의 qubit에 R_z SX R_z gate가 더해진 형태로 표현된다.
        # 그런데 R_z gate는 소프트웨어적인 처리이기 때문에 실제로 오류가 생기는 것은 SX gate뿐이다.
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        for num in range( len( real_qubit_list ) ):
            #circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[ (f'({real_qubit_list[ num ]},)') ] )
            circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[ f'{real_qubit_list[ num ]}' ] )
    
    # CNOT 연산자의 오류 모델과 비슷하게 CZ 연산자 또한 구성하는 큐빗 연산자에 따라 오류모델을 추가한다.
    def CZ_err_front( self, circuit, qubit_num_list ):
        # ECR gate를 통해서 CZ gate를 구성할 경우 ECR gate 앞에 생기는 gate에 대한 오류율을 추가한다.
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        # CZ gate를 transpile하는 경우에는 ECR gate의 앞의 control qubit은 X R_z gate로 구성되고 target qubit은 R_z SX R_z 형태로 gate가 구성된다. 
        # control qubit에 대한 오류
        for num in np.arange( 0, len( real_qubit_list ), 2 ):
            # x gate에 대한 depolarizing 오류를 추가한다.
            #circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.x_dic[ (f'({real_qubit_list[ num ]},)') ] )
            circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.x_dic[ f'{real_qubit_list[ num ]}' ] )
        # target qubit에 대한 오류
        for num in np.arange( 1, len( real_qubit_list ), 2 ):
            # sx gate에 대한 depolarizing 오류를 추가한다.
            #circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[ (f'({real_qubit_list[ num ]},)' )] ) 
            circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[ f'{real_qubit_list[ num ]}' ] )

    # 두 큐빗 연산자 오류인 경우에는 두 큐빗의 오류 모델을 고려한다.
    def CZ_err_middle( self, circuit, qubit_num_list ):
        # ECR gate로 인한 2 qubit 오류
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        for num in np.arange( 0, len( real_qubit_list ), 2 ):
            circuit.append_operation( "DEPOLARIZE2", [ qubit_num_list[ num ],qubit_num_list[ num + 1 ] ],
                                                        self.ecr_dic[f'({real_qubit_list[ num ]}, {real_qubit_list[ num + 1 ]})'] )
        
    # target 큐빗에 대한 추가 오류 모델을 구성한다.
    # CZ gate의 경우에는 ECR gate 뒤에 control qubit에 추가적인 gate가 가해지지 않는다.
    # target qubit에는 R_z SX R_z gate가 가해지기 때문에 SX gate에 대한 오류만을 추가한다.
    def CZ_err_back( self, circuit, qubit_num_list ):
        real_qubit_list = self.sim2real_index_swap( qubit_num_list )
        for num in np.arange( 0, len( real_qubit_list ), 2 ):
            #circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[ (f'({real_qubit_list[ num ]},)') ] )
            circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.id_dic[ f'{real_qubit_list[ num ]}' ] )
        for num in np.arange( 1, len( real_qubit_list ), 2 ):
            #circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[ (f'({real_qubit_list[ num ]},)') ] )
            circuit.append_operation( "DEPOLARIZE1", qubit_num_list[ num ], self.sx_dic[ f'{real_qubit_list[ num ]}' ] )

# IBM 하드웨어의 오류율을 고려하여 Repetition code를 수행하는 양자 회로의 오류 모델을 간접적으로 모방한다.
# 모방한 stim code로 시뮬레이션로 오류율을 구함과 동시에 실제 하드웨어의 오류 수정하기 위한 syndrome graph를 구성할 수 있다.
class Repetition_QEC_Stim_noise(stim_noise_model, encoder_utils):
    """
    저장된 error model을 통해서 repetition code에 대한 stim 회로를 만든다.
    distance: code distance (number of data qubit)
    round_num: 회로에서 round를 몇 번 반복할 것인지
    flag_length:data qubit과 syndrome qubit을 연결하기 위해서 flag qubit을 몇개 사용할 것인가.
    datetime: 어느 시점에서 저장한 calibration data를 사용할 것인가
    real_qubit_index: 실제 하드웨어에서 몇 번 qubit을 사용하였는가
    initial_state: 어떤 logical state에 대한 circuit을 만들 것인가
    """    
    def __init__( self, distance, round_num, flag_length, datetime, real_qubit_index, initial_state = '0' ):
        # 각 변수를 클래스 내에서 정의해준다.
        self.distance = distance
        self.rounds = round_num
        self.flag_length = flag_length
        self.real_qubit_index = real_qubit_index
        self.initial_state = initial_state
        # 파라미터를 사용하여 Repetition code의 구조를 계산하는 클래스로 큐빗 번호 & 위치의 dictionary를 구성한다.
        qubit_structure = Structure_1d( self.distance, self.flag_length )
        self.qubits_num2loc, self.qubits_loc2num, self.qubits_dic = qubit_structure.structure_dics( )
        # 각 qubit 종류의 qubit 번호를 저장한다.
        self.data = self.qubits_dic[ 'data_qubit' ]
        self.syn = self.qubits_dic[ 'syn_qubit' ]
        self.flag = self.qubits_dic[ 'flag_qubit' ]
        self.total_qubit = self.data + self.syn + self.flag
        # idle error가 발생하는 시기를 맞추기 위한 list
        self.active_qubits = []

        # 큐빗의 오류 모델을 추가하기 위해 noise 모델 클래스를 추가한다.
        super().__init__(self.total_qubit, datetime, self.real_qubit_index)
        
    # 회로 내에서 측정하는 연산자를 추가하기 위함이다.
    def measurement( self, circuit, groups ):
        
        # stim code에서 측정 결과를 추적하기 위한 상대적인 측정 시간이 필요하기에 current time을 정의한다.
        # 측정이 진행되면 측정 결과가 맨 마지막에 추가되기 때문에 - index를 사용해서 뒤에서 부터 측정 결과에 접근할 필요가 있다.
        # 따라서 첫 번째로 측정되는 qubit은 이번 시행에서 측정하는 list의 크기만큼 -한 값과 같다. 
        # 이렇게 모든 측정이 가장 마지막 측정으로 부터 얼마나 오래되었는지를 표현하는 dictionary를 만든다.
        current_time = -len( groups )
        measure_num = { }
        
        # 측정 하기 이전에 측정의 오류를 회로에 추가한다.
        # stim code는 측정을 진행한 후에 오류를 넣을 수 없기 때문에 측정 operation 전에 measurement error를 추가한다.
        self.readout_err( circuit, groups )
        
        # 측정을 큐빗들에 대해서 순서대로 수행하며, 각 측정 시간들을 기록한다.
        # 측정이 될 때 마다 1씩 측정 시간을 더하여 상대적인 위치를 저장한다.
        for qubit in groups:
            circuit.append_operation( "MR", [ qubit ] )
            measure_num[ qubit ] = current_time
            current_time += 1
        
        # 측정 뒤에 초기화를 하는데, 초기화 오류를 해당 큐빗들에 추가한다.
        # 초기화 과정에서 생긴 error는 다음 round에 검출되기 때문에 measurement 이후에 오류를 추가한다.
        self.reset_err( circuit, groups )

        return measure_num
        
    # flag 큐빗들을 사용하여 syndrome 큐빗과 data 큐빗 사이의 연결을 하기 위해 필요한 부분이다.
    # data 큐빗과 상호작용하기 이전의 과정이다.
    def stab_front( self ):
               
        circuit = stim.Circuit( )
        
        # flag_length가 0이 아닌 경우에 flag 큐빗이 추가되므로, flag 큐빗에 대한 상호작용 항을 회로에 고려한다.
        if self.flag_length != 0:
            
            # 먼저 syndrome 큐빗으로 부터 오류를 탐지하기 위해 하다마드 연산자를 가한다.
            # flag qubit을 사용하는 경우 초기 상태와 관련없이 모두 Hadamard gate를 가해야 한다.
            circuit.append_operation( "H", self.syn )
            self.hadamard_err( circuit, self.syn )
            
            # syndrome 큐빗과 가장 가까이에 인접한 두 flag 큐빗과 CNOT 연산자를 사용하여 연결한다.
            cnot_pairs = [ ]
            for syndrome_qubit in self.syn:
                # syndrome qubit의 위 아래의 qubit중 아래쪽 qubit과 syndrome qubit을 cnot으로 연결한다.
                qubits = self.patch_qubits( syndrome_qubit, [ 1, -1 ] )
                self.CNOT_append( cnot_pairs, [ syndrome_qubit, qubits[ 0 ] ] )
            # CNOT gate를 추가한다.
            # ECR gate에 대한 오류율과 다른 single qubit gate의 오류율을 CNOT gate의 오류율로 바꾸어서 CNOT gate의 오류를 적용한다.            
            self.CNOT_err_front( circuit, cnot_pairs )
            circuit.append_operation( "CNOT", cnot_pairs )
            self.CNOT_err_middle( circuit, cnot_pairs )
            self.CNOT_err_back( circuit, cnot_pairs )
            # CNOT이 적용되지 않는 qubit에 idle 오류를 추가한다.
            self.apply_idle_err_to_idle_qubit(circuit, cnot_pairs)
            # 새롭게 추가되는 CNOT pair
            cnot_pairs = [ ]
            for syndrome_qubit in self.syn:
                # syndrome qubit의 위 아래의 qubit중 위쪽 qubit과 syndrome qubit을 cnot으로 연결한다.
                qubits = self.patch_qubits( syndrome_qubit, [ 1, -1 ] )
                self.CNOT_append( cnot_pairs, [ syndrome_qubit, qubits[ 1 ] ] )
            
            # CNOT gate를 추가한다.
            # ECR gate에 대한 오류율과 다른 single qubit gate의 오류율을 CNOT gate의 오류율로 바꾸어서 CNOT gate의 오류를 적용한다.            
            self.CNOT_err_front( circuit, cnot_pairs )
            circuit.append_operation( "CNOT", cnot_pairs )
            self.CNOT_err_middle( circuit, cnot_pairs )
            self.CNOT_err_back( circuit, cnot_pairs )
            # 나중에 적용되는 CNOT은 다음 단계에서 같이 처리되기 때문에 idle오류를 나중에 처리하여야 한다.
            self.active_qubits.extend(cnot_pairs)
            # flag_length가 2 이상이면, flag 큐빗간의 상호작용을 구성해야한다. 
            if 1 < self.flag_length:
                
                # syndrome qubit으로 부터 CNOT으로 연결할 두 flag qubit이 얼마나 떨어져 있는지를 확인한다.
                for rel_pos in np.arange( 1, self.flag_length ):
                    # syndrome qubit으로 부터 rel_pos만큼 떨어진 이웃한 두 flag qubit 쌍
                    delta = [ rel_pos, 1 + rel_pos, -rel_pos, -1 - rel_pos ]
                    # syndrome 큐빗에 가까운 flag 큐빗과 data 큐빗에 가까이에 위치한 큐빗 사이를 구분하여
                    # 이들 사이를 주어진 순서에 맞게 CNOT 연산자로 상호작용을 구성한다.
                    cnot_pairs = [ ]
                    for syndrome_qubit in self.syn:
                        # syndrome qubit의 아랫쪽에 있는 flag qubit들을 CNOT으로 연결
                        qubits = self.patch_qubits( syndrome_qubit, delta )
                        self.CNOT_append( cnot_pairs, [ qubits[ 0 ], qubits[ 1 ] ] )

                    # CNOT gate를 추가한다.
                    # ECR gate에 대한 오류율과 다른 single qubit gate의 오류율을 CNOT gate의 오류율로 바꾸어서 CNOT gate의 오류를 적용한다.            
                    self.CNOT_err_front( circuit, cnot_pairs )
                    circuit.append_operation( "CNOT", cnot_pairs )
                    self.CNOT_err_middle( circuit, cnot_pairs )
                    self.CNOT_err_back( circuit, cnot_pairs )
                    # 이전 단계에서 추가된 cnot들과 현재의 cnot들을 통해서 gate가 가해지지 않은 qubit들에 idle 오류를 추가한다.
                    self.active_qubits.extend(cnot_pairs)
                    self.apply_idle_err_to_idle_qubit(circuit, self.active_qubits)
                    # idle 오류를 적용한 후 초기화
                    self.active_qubits=[]
                    # 새로운 CNOT pair
                    cnot_pairs = [ ]
                    for syndrome_qubit in self.syn:
                        qubits = self.patch_qubits( syndrome_qubit, delta )
                        # syndrome qubit의 위 쪽에 있는 flag qubit들을 CNOT으로 연결
                        self.CNOT_append( cnot_pairs, [ qubits[ 2 ], qubits[ 3 ] ] )

                    # CNOT gate를 추가한다.
                    # ECR gate에 대한 오류율과 다른 single qubit gate의 오류율을 CNOT gate의 오류율로 바꾸어서 CNOT gate의 오류를 적용한다.            
                    self.CNOT_err_front( circuit, cnot_pairs )
                    circuit.append_operation( "CNOT", cnot_pairs )
                    self.CNOT_err_middle( circuit, cnot_pairs )
                    self.CNOT_err_back( circuit, cnot_pairs )
                    # 이후에 나오는 cnot들과 병렬 처리를 하기 때문에 나중에 idle 오류를 추가
                    self.active_qubits.extend(cnot_pairs)
        else:
            # Z 오류를 탐지하기 위해서는, syndrome 큐빗으로 오류를 탐지하기 위해 하다마드 연산자를 가한다.
            if self.initial_state == '+' or self.initial_state == '-':
                circuit.append_operation( "H", self.syn )
                # hadamard gate오류
                self.hadamard_err( circuit, self.syn )
            
        return circuit
        
    # stab_back은 stab_front와 정확히 반대의 순서로 flag qubit과 syndrome qubit들 간의 상호작용을 구성한다.
    # 다만 CNOT들을 최대한 병렬로 처리하여 depth을 줄이기 위해서는 syndrome qubit을 기준으로 위쪽과 아래쪽에 CNOT gate를 가하는 순서는 동일하게 유지되어야 한다.
    # 이를 통해 data 큐빗의 오류가 flag 큐빗을 타고 syndrome 큐빗으로 전파가 되어 오류 탐지를 할 수 있다.
    def stab_back( self ):

        # 기본 회로 생성
        circuit = stim.Circuit( )
        # flag qubit이 존재하는 경우에만 flag qubit과 syndrome qubit을 연결하는 과정이 필요
        if self.flag_length != 0:
            
            # flag_length가 2 이상이면, flag 큐빗간의 상호작용을 구성해야한다. 
            if 1 < self.flag_length:
                
                # syndrome으로 부터 가장 멀리 떨어진 flag qubit부터 연결을 한다.
                for rel_pos in np.arange( self.flag_length - 1, 0, -1 ):
                    # 연결할 두 flag qubit 쌍
                    delta = [ rel_pos, 1 + rel_pos, -rel_pos, -1 - rel_pos ]
                    # syndrome 큐빗에 가까운 flag 큐빗과 data 큐빗에 가까이에 위치한 큐빗 사이를 구분하여
                    # 이들 사이를 주어진 순서에 맞게 CNOT 연산자로 상호작용을 구성한다.
                    cnot_pairs = [ ]
                    # 모든 syndrome qubit들에 대해서
                    for syndrome_qubit in self.syn:
                        # 아래쪽의 flag qubit부터 연결을 한다.
                        qubits = self.patch_qubits( syndrome_qubit, delta )
                        self.CNOT_append( cnot_pairs, [ qubits[ 0 ], qubits[ 1 ] ] )
                    
                    # CNOT gate를 추가한다.
                    # ECR gate에 대한 오류율과 다른 single qubit gate의 오류율을 CNOT gate의 오류율로 바꾸어서 CNOT gate의 오류를 적용한다.            
                    self.CNOT_err_front( circuit, cnot_pairs )
                    circuit.append_operation( "CNOT", cnot_pairs )
                    self.CNOT_err_middle( circuit, cnot_pairs )
                    self.CNOT_err_back( circuit, cnot_pairs )
                    # 병렬로 처리되는 cnot들과 현재 적용된 cnot들을 제외한 사용되지 않는 qubit들에 idle 오류를 적용한다.
                    self.active_qubits.extend(cnot_pairs)
                    self.apply_idle_err_to_idle_qubit(circuit, self.active_qubits)
                    # 처리후 초기화
                    self.active_qubits = []
                    # 위쪽 CNOT pair
                    cnot_pairs = [ ]
                    # 모든 syndrome qubit들에 대해서
                    for syndrome_qubit in self.syn:
                        # 위쪽의 flag qubit부터 연결을 한다.
                        qubits = self.patch_qubits( syndrome_qubit, delta )
                        self.CNOT_append( cnot_pairs, [ qubits[ 2 ], qubits[ 3 ] ] )

                    # CNOT gate를 추가한다.
                    # ECR gate에 대한 오류율과 다른 single qubit gate의 오류율을 CNOT gate의 오류율로 바꾸어서 CNOT gate의 오류를 적용한다.            
                    self.CNOT_err_front( circuit, cnot_pairs )
                    circuit.append_operation( "CNOT", cnot_pairs )
                    self.CNOT_err_middle( circuit, cnot_pairs )
                    self.CNOT_err_back( circuit, cnot_pairs )
                    # 다음에 적용되는 cnot들과 같이 idle 오류를 처리
                    self.active_qubits.extend(cnot_pairs)
            
            # syndrome 큐빗과 가장 가까이에 인접한 두 flag 큐빗과 CNOT 연산자를 사용하여 연결한다.
            cnot_pairs = [ ]
            # 모든 syndrome qubit에 대해서 
            for syndrome_qubit in self.syn:
                # syndrome qubit과 flag qubit의 얽힘을 풀기 위해 syndrome qubit과 가장 인접한 flag qubit을 찾는다.
                qubits = self.patch_qubits( syndrome_qubit, [ 1, -1 ] )
                # syndrome qubit의 아래 쪽에 있는 flag qubit과 syndrome qubit을 연결한다. 
                self.CNOT_append( cnot_pairs, [ syndrome_qubit, qubits[ 0 ] ] )
            
            # CNOT gate를 추가한다.
            # ECR gate에 대한 오류율과 다른 single qubit gate의 오류율을 CNOT gate의 오류율로 바꾸어서 CNOT gate의 오류를 적용한다.            
            self.CNOT_err_front( circuit, cnot_pairs )
            circuit.append_operation( "CNOT", cnot_pairs )
            self.CNOT_err_middle( circuit, cnot_pairs )
            self.CNOT_err_back( circuit, cnot_pairs )
            # 이전에 추가한 병렬 처리를 하는 qubit들과 현재의 qubit들을 합친다.
            self.active_qubits.extend(cnot_pairs)
            # idle 오류를 추가
            self.apply_idle_err_to_idle_qubit(circuit, self.active_qubits)
            # 초기화
            self.active_qubits = [] 

            cnot_pairs = [ ]
            # 모든 syndrome qubit에 대해서 
            for syndrome_qubit in self.syn:
                # syndrome qubit과 flag qubit의 얽힘을 풀기 위해 syndrome qubit과 가장 인접한 flag qubit을 찾는다.
                qubits = self.patch_qubits( syndrome_qubit, [ 1, -1 ] )
                # syndrome qubit의 위 쪽에 있는 flag qubit과 syndrome qubit을 연결한다. 
                self.CNOT_append( cnot_pairs, [ syndrome_qubit, qubits[ 1 ] ] ) 
                
            # CNOT gate를 추가한다.
            # ECR gate에 대한 오류율과 다른 single qubit gate의 오류율을 CNOT gate의 오류율로 바꾸어서 CNOT gate의 오류를 적용한다.            
            self.CNOT_err_front( circuit, cnot_pairs )
            circuit.append_operation( "CNOT", cnot_pairs )
            self.CNOT_err_middle( circuit, cnot_pairs )
            self.CNOT_err_back( circuit, cnot_pairs )
            # stabilizer의 마지막 부분이기 때문에 병렬처리하는 cnot이 없어서 idle 오류를 바로 추가한다.
            self.apply_idle_err_to_idle_qubit(circuit, cnot_pairs)

            # syndrome 큐빗으로 부터 오류를 탐지하기 위해 하다마드 연산자를 가한다.
            circuit.append_operation( "H", self.syn )
            # Hadamard 오류를 추가
            self.hadamard_err( circuit, self.syn )
        
        else:
            # Z 기저로 탐지하기 위해서는, syndrome 큐빗으로 오류를 탐지하기 위해 하다마드 연산자를 가한다.
            if self.initial_state == '+' or self.initial_state == '-':
                circuit.append_operation( "H", self.syn )
                # Hadamard 오류를 추가
                self.hadamard_err( circuit, self.syn )
        
        return circuit
    
    # X 오류를 탐지하기 위한 Z stabilizer 측정 양자 회로 구성이다.
    def Z_Stab_HH( self ):
        # 초기 회로를 만듬
        circuit = stim.Circuit()
        
        # syndrome 측정을 위해 stab_front를 통해 syndrome 큐빗과 flag 큐빗들 사이의 상호작용을 구성한다.
        circuit += self.stab_front( )
        
        # data qubit과 가장 가까운 flag qubit을 찾아 이들 사이의 상호작용을 구성하는데, CNOT과 H 연산자를 사용한다.
        # stab front와 stab back과 같은 순서로 syndrome qubit의 아래쪽을 먼저 연결하고 위쪽을 나중에 연결한다.
        delta = [ self.flag_length + 1, self.flag_length, -self.flag_length - 1, -self.flag_length ]
        # syndrome qubit의 아래쪽의 data qubit과의 연결
        cnot_pairs = [ ]
        # 모든 syndrome qubit에 대해서 
        for syndrome_qubit in self.syn:
            # data qubit과 가장 가까운 flag qubit (flag_length가 0인 경우 syndrome qubit)과 쌍을 만든다.
            qubits = self.patch_qubits( syndrome_qubit, delta )
            # syndrome qubit의 아랫쪽 data qubit과 syndrome qubit에 적용
            self.CNOT_append( cnot_pairs, [ qubits[ 0 ], qubits[ 1 ] ] )
        # flag length가 0인 경우에 Z stabilizer를 구성하기 위해서 CZ gate 대신에 
        # 앞 뒤의 Hadamard gate와 합쳐서 data qubit을 control, syndrome qubit을 target으로 하는 CNOT gate를 사용한다. 
        # flag length가 0이면 qubits[0]은 data qubit이고 qubits[1]은 syndrome qubit에 대응된다. 
        # stab front와 stab back에서 syndrome qubit에 시작과 끝에 Hadamard gate를 적용했기 때문에 
        # data qubit을 control로 하고 syndrome qubit을 target으로 하는 CNOT gate를 가하면 CZ gate의 역할을 하게 된다.    
        # 실제로 양자 컴퓨터에서 transpile하는 과정에서 
        # Hadamard와 CNOT을 구성하면서 생기는 R_z와 SX gate가 합쳐지면서 다른 형태의 gate를 만들 수도 있지만 이 부분은 무시한다.
        if self.flag_length == 0:
            # CNOT gate의 오류 모델을 ECR gate와 단일 qubit의 오류율로 구성한다.
            self.CNOT_err_front( circuit, cnot_pairs )
            circuit.append_operation( "CNOT", cnot_pairs )
            self.CNOT_err_middle( circuit, cnot_pairs )
            self.CNOT_err_back( circuit, cnot_pairs )
            # flag qubit이 없기 때문에 병렬처리를 하는 cnot이 없으므로 바로 idle 오류를 적용한다.
            self.apply_idle_err_to_idle_qubit(circuit, cnot_pairs)    
        # flag qubit이 존재하는 경우 가장 멀리 떨어진 flag qubit과 data qubit을 cz gate를 통해서 연결한다.
        # 이 경우는 stab front 부분의 마지막 CNOT gate와 동시에 작용되게 된다.
        else:
            # Z stabilizer를 구성하기 위해서 syndrome에서 가장 먼 flag qubit과 data qubit을 연결한다.
            self.CZ_err_front( circuit, cnot_pairs )
            circuit.append_operation( "CZ", cnot_pairs )
            self.CZ_err_middle( circuit, cnot_pairs )
            self.CZ_err_back( circuit, cnot_pairs )
            # 이전에 적용하였던 cnot과 함께 병렬 처리
            self.active_qubits.extend(cnot_pairs)
            # 병렬 처리를 하면서 사용되지 않은 qubit에 idle 오류를 추가
            self.apply_idle_err_to_idle_qubit(circuit, self.active_qubits)
            # 초기화
            self.active_qubits = []
        # syndrome qubit의 위쪽의 data qubit과의 연결
        cnot_pairs = [ ]
        # 모든 syndrome qubit들에 대해서 data qubit의 위치를 확인
        for syndrome_qubit in self.syn:
            # syndrome qubit의 위쪽에 있는 data qubit과 연결
            qubits = self.patch_qubits( syndrome_qubit, delta )
            self.CNOT_append( cnot_pairs, [ qubits[ 2 ], qubits[ 3 ] ] )
        # flag length가 0인 경우에 Z stabilizer를 구성하기 위해서 CZ gate 대신에
        # 앞 뒤의 Hadamard gate와 합쳐서 data qubit을 control, syndrome qubit을 target으로 하는 CNOT gate를 사용한다.     
        if self.flag_length == 0:
            self.CNOT_err_front( circuit, cnot_pairs )
            circuit.append_operation( "CNOT", cnot_pairs )
            self.CNOT_err_middle( circuit, cnot_pairs )
            self.CNOT_err_back( circuit, cnot_pairs )
            # flag qubit이 없기 때문에 병렬처리를 하는 cnot이 없으므로 바로 idle 오류를 적용한다.
            self.apply_idle_err_to_idle_qubit(circuit, cnot_pairs)    
        else:
            # Z stabilizer를 구성하기 위해서 syndrome에서 가장 먼 flag qubit과 data qubit을 연결한다.
            self.CZ_err_front( circuit, cnot_pairs )
            circuit.append_operation( "CZ", cnot_pairs )
            self.CZ_err_middle( circuit, cnot_pairs )
            self.CZ_err_back( circuit, cnot_pairs )
            # stab_back 부분의 CNOT과 함께 idle오류를 추가한다.
            self.active_qubits.extend(cnot_pairs)
        # stab_back을 구성하여 data qubit의 오류들이 syndrome qubit으로 전파가 되도록한다.
        circuit += self.stab_back( )
        
        return circuit

    
    # Z 오류를 탐지하기 위한 Z stabilizer 측정 양자 회로 구성이다.
    def X_Stab_HH( self ):
        # 초기 회로를 구성
        circuit = stim.Circuit()
        
        # syndrome 측정을 위해 stab_front를 통해 syndrome qubit과 flag qubit들 사이의 상호작용을 구성한다.
        circuit += self.stab_front( )
        # 가장 멀리 있는 flag qubit과 data qubit의 상대적인 위치
        delta = [ self.flag_length + 1, self.flag_length, -self.flag_length - 1, -self.flag_length ]
        
        # data qubit과 가장 가까운 flag qubit을 찾아 이들 사이의 상호작용을 구성하는데, CNOT과 H 연산자를 사용한다.
        cnot_pairs = [ ]
        # 모든 syndrome qubit에 대해서
        for syndrome_qubit in self.syn:
            # syndrome qubit의 아래쪽에 있는 data qubit과 flag qubit의 쌍을 만듬
            qubits = self.patch_qubits( syndrome_qubit, delta )
            # 이 경우에는 flag qubit이 control qubit이고 data qubit이 target qubit이 되어야 하므로 Z stab와 반대의 순서로 cnot pair를 구성한다. 
            self.CNOT_append( cnot_pairs, [ qubits[ 1 ], qubits[ 0 ] ] )
        # X stabilizer를 구성하기 위해서 flag qubit(혹은 syndrome qubit)을 data qubit과 CNOT gate로 연결한다. (flag(syndrome): control, data: target)
        # X stabilizer로 구성되는 경우에는 stab front와 stab back에서 추가적인 Hadamard gate가 가해지지 않기 때문에 flag qubit의 유무와 관계없이 동일하게 CNOT gate를 가하면 된다.    
        self.CNOT_err_front( circuit, cnot_pairs )
        circuit.append_operation( "CNOT", cnot_pairs )
        self.CNOT_err_middle( circuit, cnot_pairs )
        self.CNOT_err_back( circuit, cnot_pairs )
        # stab_front에서의 cnot들과 병렬처리를 하여 idle 오류를 계산하고 추가한다.
        self.active_qubits.extend(cnot_pairs)
        self.apply_idle_err_to_idle_qubit(circuit, self.active_qubits)
        # 초기화
        self.active_qubits = []

        cnot_pairs = [ ]
        # 모든 syndrome qubit에 대해서
        for syndrome_qubit in self.syn:
            # syndrome qubit의 위쪽의 data qubit과의 쌍을 만든다.
            qubits = self.patch_qubits( syndrome_qubit, delta )
            # Z stabilizer와 control과 target의 순서가 반대
            self.CNOT_append( cnot_pairs, [ qubits[ 3 ], qubits[ 2 ] ] )
            
        # X stabilizer를 구성하기 위해서 flag qubit(혹은 syndrome qubit)을 data qubit과 CNOT gate로 연결한다. (flag(syndrome): control, data: target)    
        self.CNOT_err_front( circuit, cnot_pairs )
        circuit.append_operation( "CNOT", cnot_pairs )
        self.CNOT_err_middle( circuit, cnot_pairs )
        self.CNOT_err_back( circuit, cnot_pairs )
        # stab_back 부분의 cnot과 동시에 idle 오류를 계산
        self.active_qubits.extend(cnot_pairs)
        # stab_back을 구성하여 data 큐빗의 오류들이 syndrome 큐빗으로 전파가 되도록한다.
        circuit += self.stab_back( )
        
        return circuit
    
    
    # class 내의 변수들을 사용하여 stabilizer 측정하기 위한 syndrome 큐빗과 data 큐빗 사이의 상호작용을 구성하도록 하는 circuit을 구성한다.
    def stabilizer( self ):

        # 준비된 논리 양자 상태에 따라 Z 또는 X stabilizer의 측정으로 오류를 줄일 수 있는 타입은 선정하여 수정한다.
        if self.initial_state == '0' or self.initial_state == '1':
            # 0, 1상태일 경우에는 z stabilizer만 사용
            stab_circuit = self.Z_Stab_HH( )
        else:
            # +, -상태일 경우에는 x stabilizer만 사용
            stab_circuit  = self.X_Stab_HH( )
        # syndrome과 flag qubit에 대한 측정을 진행    
        measure_num = self.measurement( stab_circuit, self.syn + self.flag )

        # stabilizer circuit에 측정 회로까지 포함하여 syndrome_extract_circuit을 구성한다.
        syndrome_extract_circuit = stab_circuit

        # 측정 순서에 대한 정보를 measurement_dic에 정의한다.
        measurement_dic = measure_num

        return syndrome_extract_circuit, measurement_dic
    
    # stabilizer 측정 회로 뒤에 측정된 syndrome 큐빗 결과를 통해 syndrome 정보를 만들기 위한 회로이다.
    def detector( self, measurement_dict ):
        #detector 정보를 담기 위한 회로        
        Detector_circuit = stim.Circuit( )
        # 측정된 결과를 불러오기 위한 수치들
        single_round_time = len( self.syn + self.flag )
        total_time = single_round_time
        measure = measurement_dict
        
        # syndrome 큐빗을 중심으로 모든 flag 큐빗들의 측정 결과까지 모두 고려하여 parity를 확인한다.
        delta = np.arange( -self.flag_length, self.flag_length + 1, 1 )
        # 모든 syndrome qubit에 대해서 
        for syndrome_qubit in self.syn:
            stab_time = [ ]
            #각 syndrome qubit에 대해서 같은 data qubit들 사이에 있는 flag qubit들을 추가
            qubits = self.patch_qubits( syndrome_qubit, delta )
            self.qubit_append( stab_time, [ qubits[ i ] for i in range( len( delta ) ) ])

            target_rec = [ ]
            # syndrome qubit과 flag qubit들에 대해서 
            for time in stab_time:
                # 그 qubit의 현재 round의 측정 결과를 비교할 parity에 추가한다.
                target_rec.append( stim.target_rec( measure[ time ] ) )
                # 그 qubit의 이전 round의 측정 결과를 비교할 parity에 추가한다.
                target_rec.append( stim.target_rec( measure[ time ] - total_time ) )
            # 두 round에 걸쳐서 syndrome qubit과 flag qubit에 대한 측정 결과에 대한 parity를 확인하는 detector를 추가한다.
            # 고려하는 모든 측정들을 target_rec에 추가하여 parity를 확인한다.
            Detector_circuit.append_operation( "DETECTOR", target_rec )

        return Detector_circuit
    
    # 논리 큐빗의 초기 상태가 0(1)인지 +(-)인지에 따라 data 큐빗들의 양자 상태를 준비한다.
    # 모든 data qubit을 원하는 상태로 초기화를 진행하면 repetition code에 대한 logical state를 구성할 수 있다.
    def initial_state_circuit( self ):
        # 초기 회로
        initial_circuit = stim.Circuit( )
        # 회로를 처음에 0상태로 초기화하며 생기는 오류 (사실 오류 모델은 측정후 x gate형태로 계산했는데 처음에 초기화하는 건 relaxation써서 다를지도?)
        self.reset_err( initial_circuit, self.data + self.syn + self.flag )

        # 논리 양자 상태가 +이면 하다마드 연산자를 모든 data 큐빗에 수행한다.
        if self.initial_state == '+':
            initial_circuit.append_operation( "H", self.data )
            self.hadamard_err( initial_circuit, self.data )
            
        # 논리 양자 상태가 1이면 X 연산자를 모든 data 큐빗에 수행한다.
        elif self.initial_state == '1':
            initial_circuit.append_operation( "X", self.data )
            self.x_err( initial_circuit, self.data )
            
        # 논리 양자 상태가 -이면 X 연산자와 하다마드 연산자를 모든 data 큐빗에 수행한다.
        elif self.initial_state == '-':
            initial_circuit.append_operation( "X", self.data )
            self.x_err( initial_circuit, self.data )
            
            initial_circuit.append_operation( "H", self.data )
            self.hadamard_err( initial_circuit, self.data )

        return initial_circuit
    
    # 초기 양자 상태 준비에 대한 오류 탐지를 위해 detector_first를 정의한다.
    # 첫번째 detector는 이전 round가 없기 때문에 다르게 정의할 필요가 있다.
    def detector_first( self, measurement_dict ):
        # 빈 회로를 만든다.        
        Detector_first_circuit = stim.Circuit( )
        # 측정한 순서에 대한 dictionary를 입력받는다.
        measure = measurement_dict
    # syndrome qubit에 대한 flag qubit들의 상대적 위치와 syndrome qubit (syndrome qubit (0)도 포함됨)
        delta = np.arange( -self.flag_length, self.flag_length + 1, 1 )
    
        # syn과 flag 큐빗들의 측정 상태들의 parity를 확인한다.
        # 초기 Detector는 이전 시간의 측정결과가 없으므로 이전 시간대의 측정을 추가하지 않는다.
        # 각 syndrome 큐빗에 대하여 인접한 flag 큐빗들을 모두 고려하여 Detector를 구성한다.
        for syndrome_qubit in self.syn:
            # 각 syndrome qubit으로 부터 떨어져 있는 flag qubit들의 list (sydrome qubit 자신도 포함됨)
            stab_time = [ ]
            # syndrome qubit과 flag qubit의 list
            qubits = self.patch_qubits( syndrome_qubit, delta )
            self.qubit_append( stab_time, [ qubits[ i ] for i in range( len( delta ) ) ])
            # syndrome qubit과 flag qubit들의 parity를 비교하는 detector를 추가한다. 
            # 두 data qubit의 사이에 존재하는 syndrome qubit과 flag qubit의 측정결과의 parity가 하나의 detector로 작용된다.
            target_rec = [ ]
            for time in stab_time:
                target_rec.append( stim.target_rec( measure[ time ] ) )
            
            Detector_first_circuit.append_operation( "DETECTOR", target_rec )

        return Detector_first_circuit
    
    # data 큐빗들의 양자 상태들을 측정하여 Logical 큐빗이 어느 상태인지 파악한다.
    def data_measurement( self, circuit ):

        # 측정한 순서대로 list에 추가되기 때문에 모두 측정한 뒤 첫번째 측정 결과는 -len(qubits)에 저장된다.
        current_time = -len( self.data )
        data_measure_num = { }
        measure_qubits = [ ]

        # data 큐빗들의 측정 순서를 모두 기록하여 이후에 논리 상태를 확인한다.
        for qubit in self.data:
            measure_qubits.append( qubit )
            # 각 qubit이 어느 시점에서 측정 되었는지를 dictionary에 기록
            data_measure_num[ qubit ] = current_time
            current_time += 1
        
        # 논리 양자 상태가 X 기저라면 Z 기저로 구성하기 위해 하다마드 연산자를 수행한다.
        # +, -상태는 X basis로 측정되어야 한다.
        if self.initial_state == '+' or self.initial_state == '-':
            circuit.append_operation( "H", measure_qubits )
            self.hadamard_err( circuit, measure_qubits )
        # 측정 오류 추가
        self.readout_err( circuit, measure_qubits )
        circuit.append_operation( "MR", measure_qubits )

        return data_measure_num
    
    # 측정된 data 큐빗들의 측정 값들을 통해 논리 상태를 parity로 확인한다.
    # 측정된 상태가 0상태인지 1상태인지 (혹은 +상태인지 -상태인지)를 확인하기 위해서 측정된 상태들의 parity를 확인한다.
    # 짝수일 경우 0(+), 홀수일 경우 1(-)
    def detector_observable( self, circuit, data_measure_num ):
        logic_state = [ ]

        # 논리 큐빗의 observable은 모든 data 큐빗들의 parity를 확인한다.
        for qubit in self.data:
            logic_state.append( stim.target_rec( data_measure_num[ qubit ] ) )
        # 회로에 observable을 추가
        circuit.append_operation( "OBSERVABLE_INCLUDE", logic_state, 0 )
    
    # QEC code의 마지막 단계에서
    # stabilizer 측정 뒤에, 각 stabilizer 패치마다 data 큐빗과 syndrome 큐빗과의 parity 확인 및 observable을 circuit에 추가한다.
    def detector_last( self, measurement_dict ):

        Detector_last_circuit = stim.Circuit( )
        # data qubit과 syndrome qubit을 모두 측정하기 때문에 syndrome qubit의 측정 순서를 찾기 위해서 data qubit의 측정 횟수가 필요하다.
        data_total_time = len( self.data )

        # data qubit들을 주어진 Stabilizer 그룹에 맞추어 측정한다.
        # 측정된 data qubit들의 순서를 저장
        data_num = self.data_measurement( Detector_last_circuit )
        # syndrome qubit의 이전 측정 결과를 입력받는다. 
        measure = measurement_dict
        # syndrome으로 부터 data qubit의 상대적인 위치
        delta_data = [ -self.flag_length - 1, self.flag_length + 1 ]
        # syndrome qubit으로 부터의 flag qubit과 syndrome qubit의 상대적인 위치
        delta_syn = np.arange( -self.flag_length, self.flag_length + 1, 1 )
        # data qubit의 측정 dictionary를 syndrome qubit의 dictionary에 합친다.
        # 마지막 round에서 측정한 모든 qubit에 대한 측정 순서를 담은 dictionary
        measure.update( data_num )

        # 마지막 detector에서는 syndrome 큐빗, flag 큐빗, data 큐빗까지 모두 확인을 한다.
    # 모든 syndrome들에 대해서
        for syndrome_qubit in self.syn:
            
            stab_time_data = [ ]
            stab_time_syn = [ ]
            # syndrome과 인접한 data qubit들의 list
            qubits_data = self.patch_qubits( syndrome_qubit, delta_data )
            self.qubit_append( stab_time_data, [ qubits_data[ i ] for i in range( len( delta_data ) ) ])
            # syndrome과 그와 인접한 flag qubit들
            qubits_syn = self.patch_qubits( syndrome_qubit, delta_syn )
            self.qubit_append( stab_time_syn, [ qubits_syn[ i ] for i in range( len( delta_syn ) ) ])

            target_rec = [ ]
            # syndrome, data, flag qubit들 전체의 parity를 확인한다.
            for time in stab_time_syn:
                # syndrome qubit의 경우에는 이전에 측정된 결과가 있기 때문에 그 qubit의 이전 측정 결과와의 parity를 비교한다.
                # 따라서 마지막 round의 syndrome과 flag qubit의 측정 결과와 이전 round의 syndrome과 flag qubit의 측정 결과를 모두 합친 parity를 하나의 detector로 사용한다.
                target_rec.append( stim.target_rec( measure[ time ] - data_total_time ) )
            # data qubit은 이전 round의 측정 결과가 없기 때문에 측정 결과가 바로 detector로 작용되게 된다.
            for time in stab_time_data:
                target_rec.append( stim.target_rec( measure[ time ] ) )

            Detector_last_circuit.append_operation( "DETECTOR", target_rec )

        # 마지막 detector를 통해 syndrome 정보를 추출하며, data 큐빗 측정 결과들로 부터 논리 큐빗의 양자 상태를 확인한다.
        self.detector_observable( Detector_last_circuit, data_num )

        return Detector_last_circuit
    
    # 각 타입별로 수행한 회로들을 구분하기 위한 구분 연산자이다.
    # qiskit의 barrier 역할, 
    def tick_circuit( self ):
        
        barrier_circuit = stim.Circuit( )
        barrier_circuit.append_operation( "TICK" )
        
        return barrier_circuit
    
    # 본 class의 함수들을 모아 실제 양자 컴퓨터의 오류 모델을 적용한 Repetition code를 구성한다.
    def generate_circuit( self ):
            
        rounds = self.rounds
        qubits_loc2num = self.qubits_loc2num
        
        barrier_circuit = self.tick_circuit( )
        
        # 초기 상태를 준비할 수 있도록 한다.
        initial_circuit = self.initial_state_circuit( )
        
        # 주어진 논리 큐빗의 stabilizer 측정 회로 및 측정 순서들을 구성한다.
        stab_circuit, measurement_dict = self.stabilizer( )
        
        # 측정된 순서들을 통해 확인해야하는 syndrome 정보들을 측정 결과들 사이의 parity로 확인한다.
        # 첫번째 측정에 대한 detector
        det_first_circuit = self.detector_first( measurement_dict )
        # 두번째부터 마지막 전까지의 detector
        det_circuit = self.detector( measurement_dict )
        # data qubit의 측정 결과를 포함하는 마지막 detector
        det_last_circuit = self.detector_last( measurement_dict ) 

        # 위의 양자 회로들을 가지고 순서대로 syndrome_first, syndrome, syndrome_last를 정의한다.
        # 첫번째 측정까지의 회로
        syndrome_first_circuit = stab_circuit + det_first_circuit + barrier_circuit
        # 중간에 반복되는 형태의 회로
        syndrome_circuit = ( rounds - 1 ) * ( stab_circuit + det_circuit + barrier_circuit )
        # 마지막 detector에 해당되는 회로
        syndrome_last_circuit = det_last_circuit
        
        # 전체 QEC code의 양자 회로에 큐빗들을 배치한 뒤에 위의 회로들을 하나하나 붙여 최종적인 QEC code quantum circuit을 구성한다.
        circuit = stim.Circuit( )
        # 회로에서 qubit들의 상대적인 위치와 번호를 circuit에 추가한다.
        for loc, num in qubits_loc2num.items( ):
            circuit.append_operation( "QUBIT_COORDS", [ num ], [ loc ] )

        # Stim 코드로 구성한 모든 회로들을 순차적으로 붙여 전체 양자 회로를 구성한다.
        circuit += initial_circuit + barrier_circuit + syndrome_first_circuit + syndrome_circuit + syndrome_last_circuit

        return circuit


if __name__ == '__main__':
    from cirq.contrib.svg import SVGCircuit
    #t = datetime.datetime(2023, 11, 21, 14, 55, 26, 546955)
    t = f'properties_d3_f1_ini0'
    circuit = Repetition_QEC_Stim_noise(3,1,1,t,[16,26,27,28,35,47,46,45,54], '0').generate_circuit()
    SVGCircuit(circuit)
    print(circuit.diagram())













