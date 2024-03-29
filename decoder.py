from typing import Callable, List, Optional
import math

import networkx as nx
import numpy as np
import pymatching
import stim

class decoder:
    """
    stim 회로로 부터 오류 모델을 불러와서 각 node가 연결될 확률에 대응되는 그래프를 만든다.
    그 그래프를 통해서 pymatching 알고리즘을 통해서 가장 가능성있는 오류의 조합을 찾는다.
    stim code의 오류모델을 사용하느냐 correlation matrix를 통해서 weight를 결정하느냐에 따라서 다른 class를 사용하게 된다.
    """
    def __init__(self, circuit, correlation_M_real = None) -> None:
        self.circuit = circuit
        self.model = circuit.detector_error_model( decompose_errors = True )
        self.correlation_M_real = correlation_M_real
        self.det_offset = 0
        # 정확한 크기를 알 수 없어서 100개를 설정
        self.coords_offset = np.zeros( 100, dtype = np.float64 )

    # 오류 모델에서 오류에 대한 정보를 얻어내서 그것을 바탕으로 graph를 만드는 함수
    def _helper(self,  m : stim.DetectorErrorModel, reps : int ):
        
        for _ in range(reps):
            # 오류 모델에 순차적으로 나열된 오류,detector, Logical 오류를 확인한다.
            for instruction in m:
                # round가 진행되면서 같은 과정이 여러번 반복되는 경우에는 repeated block의 형태로 표현된다.
                # 이 경우에는 block 내부에 있는 값들에 접근해야 하기 때문에 그 부분에 대해서 재귀적으로 내부에 들어가게 한다. 
                if isinstance( instruction, stim.DemRepeatBlock ):
                    self._helper( instruction.body_copy( ), instruction.repeat_count )
                
                # 반복되는 부분이 아닐 경우에는 입력받은 값이 detector error model인지를 확인한다.
                # detector error model인 경우에는 그 요소가 어떤 요소인지에 따라서 다른 실행을 한다.
                # Stim 코드의 가장 작은 단위로 분류를 통해 오류에 대한 정보를 찾는다.
                elif isinstance( instruction, stim.DemInstruction ):
                    # 게이트의 종류가 오류일 때, 오류에 의한 정보들을 가져온다.
                    if instruction.type == "error":
                        dets: List[ int ] = [ ]
                        frames: List[ int ] = [ ]
                        # target 값의 type을 정의한다.
                        # target은 오류가 영향을 미치는 detector나 observable을 표현해준다.
                        t: stim.DemTarget
                        # 오류가 일어날 확률을 저장한다.
                        p = instruction.args_copy( )[ 0 ]
                        
                        # 오류에 의한 모든 detector와 observable 그리고 확률에 대한 정보들을 순차적으로 고려한다.
                        for t in instruction.targets_copy( ):
                            # 오류가 detector에 영향을 주는 경우에 detector의 번호를 추가한다.
                            # 앞선 요소들중 detector가 있었을 경우 그 detector의 상대 index에 offset을 더해서 절대 index 값을 detector의 list에 추가한다. 
                            if t.is_relative_detector_id( ):
                                dets.append( t.val + self.det_offset )
                            # 오류가 observable에 영향을 주는 경우에 observable의 번호를 추가한다.
                            elif t.is_logical_observable_id( ):
                                frames.append( t.val )
                            # separator는 오류가 두종류 이상의 오류로 분리되는 것을 의미한다.
                            # 만약 두 개 이상의 detector에 영향을 주는 경우 이전까지 추가된 detector와 observable에 대한 정보를 저장하고
                            # separator이후의 detector와 observable을 저장할 새로운 list를 만든다.
                            elif t.is_separator( ):
                                # 각 구성 요소들을 개별적이며 uniform한 오류로 판단한다.
                                # 그래프에 이전까지 기록된 detector와 observable을 추가한다.
                                self.handle_error( p, dets, frames )
                                # detector와 observable의 list를 초기화
                                frames = [ ]
                                dets = [ ]
                                
                        # Handle last component.
                        # 마지막에 추가한 detector와 observable의 정보를 graph에 추가한다.
                        self.handle_error(p, dets, frames)
                    
                    # 요소가 shift_detector인 경우에는 detecotr의 offset을 준다.
                    elif instruction.type == "shift_detectors":
                        # 전체 실험에 대해서 index offset은 계속해서 증가하게 된다.
                        # index offset은 shift_detector의 target 부분(맨 뒤부분)에 표시된다.
                        self.det_offset += instruction.targets_copy( )[ 0 ]
                        # coordinate offset들을 추가 
                        # coordinate offset은 arg부분 (()안의 부분)에 표시된다.
                        # 좌표의 차원에 맞추어 offset을 더해준다.
                        a = np.array( instruction.args_copy( ) )
                        self.coords_offset[ : len( a ) ] += a
                        
                    # detector를 나타내는 Syndrome 노드를 구성한다.
                    elif instruction.type == "detector":
                        # detector의 위치를 표현하는 array를 구성한다.
                        a = np.array( instruction.args_copy( ) )
                        # 각 detector를 graph에 node로 추가한다.
                        # target에 표시된 상대 index값을 통해서 detector 번호에 맞추어 node를 추가한다.
                        for t in instruction.targets_copy( ):
                            self.handle_detector_coords( t.val + self.det_offset, a + self.coords_offset[ : len( a ) ] )
                    
                    # observable의 경우 추가적인 분석이 불필요
                    elif instruction.type == "logical_observable":
                        pass
                    else:
                        # 알 수 없는 형식
                        raise NotImplementedError( )
                else:
                    # 알 수 없는 형식
                    raise NotImplementedError( )
    
    # detector들에 대한 노드를 그래프에 추가하도록 한다.
    def handle_detector_coords( self, detector : int, coords : np.ndarray ):
        self.g.add_node( detector, coords = coords )
    
class stim_decoder(decoder):
    """
    stim circuit을 기반으로 각 오류 사건이 일어날 확률을 계산하여 그에 맞는 graph를 구성하고 MWPM을 시행한다.
    """
    def __init__(self, circuit, correlation_M_real = None) -> None:
        super().__init__(circuit, correlation_M_real)
        self.correlation_M_real = correlation_M_real
    
    # 각 오류들에 따른 detector pair들의 edge weight를 결정해주기 위한 함수이다.
    def handle_error( self, p : float, dets : List[ int ], frame_changes : List[ int ] ):
        
        # 오류의 확률 값, detecotor 그리고 frame에 대한 정보를 사용하여 syndrome 그래프의 선을 구성한다.
        # 확률이 0인 경우에는 생략한다.
        if p == 0:
            return
        
        # detector의 개수가 0인 경우에는 생략한다.
        if len( dets ) == 0:
            return
        
        # 오류에 의한 detector의 변함이 오직 1개에서 바뀔 때 이를 boundary와 결합한다.
        if len( dets ) == 1:
            dets = [ dets[ 0 ], self.boundary_node ]
            
        # detector의 개수가 3개 이상으로 넘어가면, 오류를 Syndrome 그래프로 대응 시킬 수 없다.
        if len( dets ) > 2:
            raise NotImplementedError(
                f"Error with more than 2 symptoms can't become an edge or boundary edge: {dets!r}." )
            
        # 이미 그래프 내에 존재하는 edge라면, 그 edge의 weight 값을 경우의 수에 맞게 확률값을 바꾸어 이를 업데이트한다.
        if self.g.has_edge( *dets ):
            # 선의 두 노드 값, 확률, frame을 확인하여 확률 p 값을 업데이트해준다.
            edge_data = self.g.get_edge_data( *dets )
            old_p = edge_data[ "error_probability" ]
            old_frame_changes = edge_data[ "qubit_id" ]
            # 기존의 오류와 새로 추가된 오류가 둘 중 하나만 일어날 확률을 계산
            # python의 집합 요소를 사용해서 모든 element가 같은지를 확인
            # 입력된 오류와 같은 observable에 영향을 미치는 경우 (혹은 detector에만 영향을 미치는 경우) 두 오류를 합친다.
            # 다른 경우에는 합치지 않고 새로운 edge를 추가?
            if set( old_frame_changes ) == set( frame_changes ):
                # 기존의 오류나 추가된 오류중 하나만 일어날 확률
                p = p * ( 1 - old_p ) + old_p * ( 1 - p )
                # 기존의 edge를 제거 
                self.g.remove_edge( *dets )
                
        # 업데이트된 오류 확률을 통해 새로운 edge로 추가한다.
        # 기존에 같은 오류가 있었다면 합친 값을 사용하고 없었다면 새로운 edge를 추가한다. 
        self.g.add_edge( *dets, weight = math.log( ( 1 - p ) / p ), qubit_id = frame_changes, error_probability = p )


    # stim 회로의 오류 모델로 부터 Tanner graph를 구성하기 위한 함수이다.
    def detector_error_model_to_nx_graph(self) -> nx.Graph:
        
        # networkx 그래프를 생성하고 이에 먼저 boundary 노드를 추가한다.
        self.g = nx.Graph()
        # boundary node의 번호는 가장 마지막 숫자
        self.boundary_node = self.model.num_detectors
        # -1, -1, -1의 위치에 모든 boundary node들을 위치시킨다. 
        self.g.add_node( self.boundary_node, is_boundary = True, coords = [ -1, -1, -1 ] )
        # 회로에서 오류에 대한 정보를 얻어서 graph를 그린다.
        self._helper(self.model, 1)

        return self.g

    # stim 회로를 가지고 구성한 networkx 그래프 내에서 pymatching 알고리즘 디코더를 구성하기 위한 추가적인 조치를 하기 위함이다.
    def detector_error_model_to_pymatching_graph(self ) -> pymatching.Matching:
        
        # networkx 그래프를 구성한 뒤에 pymatching 그래프로 바꾼다.
        g = self.detector_error_model_to_nx_graph()
        num_detectors = self.model.num_detectors
        num_observables = self.model.num_observables

        # 모든 그래프의 노드들이 연결 선을 가질 수있도록 하기 위해 추가적인 가상 노드를 넣어준다.
        # 기존에 node가 존재할 경우에는 추가가 안됨?
        for k in range( num_detectors ):
            g.add_node( k )
        # 가상의 node
        g.add_node( num_detectors + 1 )
        # boundary node를 포함하는 모든 node를 가상의 node와 연결한다. weight를 매우 큰 값을 주어 실제 계산에는 사용되지 않게 한다.
        for k in range( num_detectors + 1 ):
            g.add_edge( k, num_detectors + 1, weight = 10000000 )
        # qubit_id를 추가하여 edge를 다시 설정?
        g.add_edge( num_detectors, num_detectors + 1, weight = 10000000, qubit_id = list( range( num_observables ) ) )

        return pymatching.Matching( g ) 
    
class corr_decoder(decoder):
    """
    correlation matrix를 기반으로 각 오류가 일어날 확률을 얻고 이를 통해서 graph의 weight를 결정한다.
    """
    def __init__(self, circuit, correlation_M_real) -> None:
        super().__init__(circuit, correlation_M_real)
        self.correlation_M_real = correlation_M_real
    # 각 오류들에 따른 detector pair들의 edge weight를 결정해주기 위한 함수이다.
    def handle_error( self, p : float, dets: List[ int ], frame_changes : List[ int ] ):
        
        # 확률이 0 이거나 detector의 개수가 0개 이면 생략한다.
        if p == 0:
            return
        if len( dets ) == 0:
            return
        
        # 오류에 의한 detector의 변함이 오직 1개에서 바뀔 때 이를 boundary와 결합한다.
        # Boundary에 대한 확률값은 나중에 추가하기 때문에 선만 만들어 놓는다.
        if len( dets ) == 1:
            dets = [ dets[ 0 ], self.boundary_node ]
            p = -1
        
        # 두 개의 노드를 가지면, 그 확률 p 값은 coorelation_M_real을 통해 얻는다.
        elif len( dets ) == 2:
            p = self.correlation_M_real.p_matrix[ dets[ 0 ] ][ dets[ 1 ] ]
        
        # 두 개의 detector 이상에 영향을 주는 경우는 없다.
        if len( dets ) > 2:
            raise NotImplementedError(
                f"Error with more than 2 symptoms can't become an edge or boundary edge: {dets!r}." )
        
        # 오류의 확률을 가지고 syndrome 그래프의 weight를 구성한다.
        # 확률이 음수인 경우에는 해당 확률이 일어나지 않을 것이라 생각하자.
        try:
            weight = math.log( ( 1 - p ) / p )
        except:
            weight = 999999
        
        # 오류 확률을 통해 결정된 weight를 가지는 edge를 graph에 추가한다.
        self.g.add_edge( *dets, weight = weight, qubit_id = frame_changes, error_probability = p )

    
    # stim 회로의 오류 모델로 부터 Tanner graph를 구성하는데 실제 양자 컴퓨터의 샘플 확률분포로 만든다.
    def detector_error_model_to_nx_graph( self ) -> nx.Graph:
        
        # networkx 그래프를 생성하고 이에 먼저 boundary 노드를 추가한다.
        self.g = nx.Graph( )
        # 마지막 번호는 boundary node에 할당한다.
        self.boundary_node = self.model.num_detectors
        # boundary node를 추가
        self.g.add_node( self.boundary_node, is_boundary = True, coords=[ -1, -1, -1 ] ) 
        
        # 함수들을 사용하여 Syndrome 그래프를 구성한다.
        self._helper(self.model, 1)
        
        num_detectors = self.model.num_detectors
        
        # boundary 노드와 이어지는 모든 노드들(graph의 boundary에 속한 node들)의 인접한 확률들을 확인한다.
        # 확인한 확률들로 하여금 boundary와의 weight 값을 계산한다.
        for node in self.g.copy( ).neighbors( num_detectors ):

            # boundary와 이어지는 노드가 있을 경우 다음을 수행한다.
            if self.g.has_edge( node,num_detectors ):
                
                # 해당 노드와 인접한 boundary 아닌 노드들의 리스트를 얻는다.
                j_list = list( self.g.neighbors( node ) )
                # boundary가 아닌 경우를 제외
                j_list.remove( num_detectors )
                
                # 확률들의 조합들로 노드가 boundary와 이어질 확률을 계산한다.
                p = self.correlation_M_real.pib( node , j_list )

                # 확률의 값이 음수면, 그런 경우가 없다고 판단한다.
                try:
                    weight = math.log( ( 1 - p ) / p )
                except:
                    weight = 999999
                # 기존의 edge에 대한 정보?를 얻어서 저장한다.
                frame_changes = nx.get_edge_attributes( self.g, 'qubit_id' )[ ( num_detectors, node ) ]
                
                # 선을 제거하고 새롭게 업데이트하여 추가한다.
                self.g.remove_edge( node,num_detectors )
                self.g.add_edge( node, num_detectors, weight = weight, qubit_id = frame_changes, error_probability = p )
                
        return self.g

    # stim 회로를 가지고 구성한 networkx 그래프 내에서 pymatching 알고리즘 디코더를 구성하기 위한 추가적인 조치를 하기 위함이다.
    def detector_error_model_to_pymatching_graph( self ) -> pymatching.Matching:
        
        # syndrome 그래프를 구성한 뒤에, 
        g = self.detector_error_model_to_nx_graph()
        num_detectors = self.model.num_detectors
        num_observables = self.model.num_observables

        # 모든 detector에 대응되는 node들을 추가해준다.
        for k in range( num_detectors ):
            g.add_node( k )
        # detector에 대응되지 않는 가상의 node
        g.add_node( num_detectors + 1 )
        # 모든 node들이 최소한 하나의 detector에 연결되게 하기 위한 안전장치?
        # boundary node를 포함하는 모든 node가 가상의 node에 연결되게 한다.
        for k in range( num_detectors + 1 ):
            g.add_edge( k, num_detectors + 1, weight = 10000000 )
        # qubit_id를 추가하여 다시 추가?
        g.add_edge( num_detectors, num_detectors + 1, weight = 10000000, qubit_id = list( range( num_observables ) ) )

        return pymatching.Matching( g )
































