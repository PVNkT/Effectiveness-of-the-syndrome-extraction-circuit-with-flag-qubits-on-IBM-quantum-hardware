import numpy as np
import pandas as pd

from typing import Dict, List
import numpy as np
from itertools import product

class generator_utils:
    """
    IBM 양자 컴퓨터에서 실행시킬 양자 회로를 만들고 결과를 저장하는 기능들
    주로 dictionary 형태로 주어지는 실행 결과를 각 ruond와 qubit들에 대해서 csv 파일의 형태로 바꾸어 저장하는 역할을 한다. 
    """
    def __init__(self) -> None:
        pass

    # make raw data to measured data for each qubit
    def measured_bits( self, raw_results:dict ):
        results = [ ]
        
        # 측정 결과들에 따라 구분한다.
        for raw_result in raw_results.keys( ):
            
            # 측정 라운드별로 구분하여 나타낸다.
            # 측정된 상태에 대한 string은 하나로 합쳐져 있는데 각 round는 띄어쓰기로 구별된다.
            # 이를 나누어 list의 형태로 바꾼다.
            key_list = { }
            key_split = raw_result.split( )
            
            # 모든 라운드에 대하여 고려하는데, data 큐빗 측정과 flag, syn 큐빗 측정을 나눈다.
            # 맨 마지막 측정 결과는 data 큐빗의 측정 결과이다.
            key_list[ 'Data' ] = list( map( int, key_split[0] ) )
            # 각 측정된 round들에 대해서 
            for t in np.arange(1,  len( key_split ) ):
                # 각 측정 결과를 list의 형태로 바꾸어 저장한다.
                step = key_split[ t ]
                new_step = list( map( int, step ) )
                # qubit의 순서에 따라서 나누어 저장한다. (syndrome qubit과 flag qubit)
                key_list[ 'stab_' + str( t ) ] = new_step
            
            # 구분된 측정 결과들을 리스트로 묶어 결과에 추가한다.
            # 각 qubit들의 측정 결과와 그렇게 측정된게 몇 개 였는지를 tuple로 저장하고 이를 list에 추가 
            results.append( ( key_list, raw_results[ raw_result ] ) )
                
        return results
    
    # 측정된 데이터들을 엑셀 파일에 저장하기 위한 함수이다.
    def making_csv(self, distance, round_num, flag_length, initial_state, results):
        # raw 상태의 측정 결과 dictionary를 각 round에서 각 qubit들이 어떻게 측정되었는지에 대한 정보로 바꿈
        results_sim = self.measured_bits(results)

        # 양자회로 수행 뒤에 얻어진 측정 결과들을 리스트로 저장하기 위함이다.
        stabs = [ ]
        
        # 측정 결과들을 확인하면서, 라운드별 분류하여 순서대로 정리한다.
        # 각 round에 대해서
        for i in range( len( results_sim ) ):
            
            stab = [ ]
            # i번째 round의 각 qubit들의 측정 결과와 측정 횟수로 이루어진 tuple
            results_sim0 = results_sim[ i ]
            # 각 round에 대해서
            for time in range( 1, round_num+1):
                # 각 round에 해당되는 syndrome qubit과 flag qubit의 측정 결과를 list에 합친다.
                stab.extend( results_sim0[ 0 ][ f'stab_{time}'] )
                
            # 마지막 라운드에 데이터 큐빗 측정 결과를 포함해준다.
            stab.extend( results_sim0[ 0 ][ 'Data' ] )
            # 이러한 측정이 몇번 일어났는지를 저장한다.
            stab.append( results_sim0[ 1 ] )
            # 전체 측정 결과에 추가한다.
            stabs.append( stab )
        # 각 측정 결과에 대해서 각 qubit이 어떻게 측정되었는지를 나타내는 array 
        stabs = np.array( stabs )

        # 측정 결과들에 대한 데이터 분류를 위해 엑셀의 헤더를 만든다.
        stab_len = len( results_sim0[ 0 ][ f'stab_{time}' ] )
        data_len = len( results_sim0[ 0 ][ 'Data' ] )

        header_stab = [ ]
        header_stab_num = [ ]
        
        # 측정 라운드에 맞도록 헤더를 고려한다.
        for time in range( 1, round_num+1):
            # 측정된 syndrome qubit과 flag qubit의 수만큼 header를 추가한다.
            for num in range( stab_len ):
                header_stab.append( f'stab_{time}' )
                header_stab_num.append( num )
        # data qubit들에 대해서도 각 측정 값에 대한 header를 추가한다.
        for d in range( data_len ):
            header_stab.append( 'Data' )
            header_stab_num.append( d )
        
        # 몇 번 측정되었는지를 표기할 header를 추가한다.
        header_stab.append( 'num' )
        header_stab_num.append( 'simulation num' )
        
        # 측정 결과 리스트 및 측정 라운드 헤더를 모두 모아 dataframe을 만든다.
        col = pd.MultiIndex.from_arrays( [ header_stab, header_stab_num ] )
        
        data = pd.DataFrame( stabs, columns = col )

        # 위에서 계산한 모든 값들을 하나의 csv로 저장하여 그 결과를 추후에 사용할 수 있도록 한다.
        data.to_csv( self.file_path + f"distance-{ distance }_rounds-{ round_num }_flag_length-{ flag_length }_initial_state-{ initial_state }.csv")
        return data
    
    # 최적화된 회로를 수행하는 하드웨어에서 사용하는 큐빗 번호들을 순서대로 얻기 위한 함수이다.
    def qubit_indexing( self, total_qubit_num, trans_circ ):

        # 양자 회로내에 사용되는 실제 큐빗 번호와 코드에서 고려하는 큐빗 번호를 지정해주기 위한 리스트이다.
        real_qubit_index = list( np.zeros( total_qubit_num ) )

        # transpile된 양자 회로 내에서 activate한 큐빗들의 번호들을 불러와 real_qubit_index에 순서대로 추가한다.
        bit_locations = {
                            bit: { "register" : register, "index" : index }
                            for register in trans_circ._layout.initial_layout.get_registers( )
                            for index, bit in enumerate( register )
                        }
        for key, val in trans_circ._layout.initial_layout.get_virtual_bits( ).items( ):
            bit_register = bit_locations[ key ][ "register" ]
            if bit_register is None or bit_register.name != "ancilla":
                real_qubit_index[ bit_locations[ key ][ "index" ] ] = val

        return real_qubit_index


class data_analysis:
    """
    csv 파일로 저장된 실험 데이터를 불러와서 decoding을 할 때 필요한 syndrome data로 변화시킨다.
    종류 별 qubit이 어느 위치에 있는지 알기 위해서 distance와 flag_lenth의 정보가 필요하다.
    """
    def __init__(self, distance, flag_length) -> None:
        self.distance = distance
        self.flag_length = flag_length
        # data qubit 사이의 qubit의 갯수
        self.flag_syn_step = 2* self.flag_length + 1
        # flag qubit과 syndrome qubit의 총 갯수
        self.stab_step_len = self.flag_syn_step * ( self.distance - 1 )

    # syndrome 값과 flag length를 가지고 syndrome, flag를 모두 고려한 parity 값을 내놓는다.
    def flag_syn(self, syn):
        # 각 data qubit 사이에 있는 syndrome qubit과 flag qubit의 측정값의 parity를 계산해서 list로 만든다.
        syn_list = [np.sum( np.array(syn[ num : num + self.flag_syn_step ] )) % 2 for num in np.arange( 0, len( syn ), self.flag_syn_step )]
        # IBM computer는 결과를 반대 순서로 반환하기 때문에 순서를 뒤집어준다.
        new_syn = np.array(syn_list)[::-1]
        return new_syn

    # 데이터 측정 결과까지 고려한 syndrome을 구성한다.
    def flag_data_syn(self, data, syn ):
        # 마지막 round의 syndrome은 data qubit의 측정 오류까지 탐지할 수 있도록 두 개의 연속된 data qubit과 그 사이에 있는 syndrome, flag qubit들의 parity를 계산한다.  
        syn_list = [( np.sum( syn[ num * self.flag_syn_step : ( num + 1 ) * self.flag_syn_step ] ) + data[ num ] + data[ num + 1 ] ) % 2 for num in range( len( data ) - 1 )]
        # IBM은 classical memory를 반대로 읽기 때문에 실제 큐빗 순서를 바꾸어 stim code를 구성해야한다.
        new_syn = np.array(syn_list)[::-1]
        return new_syn
    
    # 측정된 결과 리스트로 부터 flag 큐빗을 고려한 syndrome 구성을 위한 함수이다.
    def stab2syn(self, stab_lists, round_num):
        
        syn_lists = [ ]
        
        # syn_lists에 각 샘플들을 계산한다.
        # 여러 번 시도한 측정 결과들에 대해서
        for stab_list in stab_lists:
            # IBM은 qubit 순서를 반대로 읽기 때문에 가장 마지막에 있는 결과가 0번째 round에 해당되게 된다.
            # 측정 라운드가 0인 경우에는 이전 측정 라운드가 없기에 그 자체로 syndrome을 구성한다.
            syn_arr = self.flag_syn( stab_list[ (round_num-1) * self.stab_step_len :  round_num  * self.stab_step_len ] )

            # 모든 라운드에 대하여 구분해 syndrome들을 계산한다.
            for time in range( round_num-2, -1, -1 ):
                
                # 두 측정 라운드 결과 사이의 parity를 확인한다.
                # parity 값을 syndrome 값으로 사용한다.
                # 현재 round의 syndrome과 flag qubit의 측정 결과
                syn_2 = stab_list[ time * self.stab_step_len : ( time + 1 ) * self.stab_step_len ]
                # 이전 round의 syndrome과 flag qubit의 측정 결과
                syn_1 = stab_list[ ( time + 1 ) * self.stab_step_len : ( time + 2 ) * self.stab_step_len ]
                # 같은 qubit에 대해서 현재와 이전의 측정 결과의 parity를 계산
                syn = ( np.array(syn_1) + np.array(syn_2) ) % 2 
                # syndrome과 flag qubit간의 parity를 비교
                syn_sample = self.flag_syn( syn )
                # 지금까지 계산된 결과 뒤에 값을 붙임
                syn_arr = np.concatenate( (syn_arr, syn_sample ))
            
            # 데이터 측정 결과와 syndrome, data 측정 결과로 마지막 syndrome을 구성한다.
            # data qubit을 측정한 결과
            data_list = stab_list[ round_num * self.stab_step_len : round_num * self.stab_step_len + self.distance ]
            #마지막으로 측정한 syndrome과 flag qubit의 결과
            syn_last_list = stab_list[ 0 : self.stab_step_len ]
            # data qubit의 측정 결과와 마지막 round에서 syndrome과 flag qubit의 측정 결과를 통해 얻은 syndrome
            syn_sample = self.flag_data_syn( data_list, syn_last_list )
            # 계산한 syndrome을 마지막에 추가
            syn_arr = np.concatenate( (syn_arr, syn_sample ))
            
            # 각 라운드별 얻어진 parity 결과들로 부터 오류의 정보를 제공하는 syndrome을 얻는다.
            # [각 round별로 syndrome을 계산한 결과, 오류를 수정하기 전에 data qubit을 측정하여 얻은 Observable, 이러한 측정이 몇번 이루어졌는지]
            # 위와 같은 list를 모든 측정 결과에 대해서 list로 만들고 하나의 array로 합쳐 반환한다.
            syn_lists.append( [ syn_arr, np.sum( data_list ) % 2, int( stab_list[ -1 ] ) ] )
            
        return np.array(syn_lists, dtype=object)

    def stab2syn_invert( self, stab_lists, round_num):
        
        syn_lists = [ ]
        
        stab_step_len = ( 2 * self.flag_length + 1 ) * ( self.distance - 1 )
        
        # syn_lists에 각 샘플들을 계산한다.
        for stab_list in stab_lists:

            syn_list = [ ]
            syn_invert = [ ]
            
            # 측정 라운드가 0인 경우에는 이전 측정 라운드가 없기에 그 자체로 syndrome을 구성한다.
            syn_sample = self.flag_syn( stab_list[ (round_num-1) * stab_step_len : round_num * stab_step_len ])[::-1]
            syn_list.extend( syn_sample )
            # 모든 라운드에 대하여 구분해 syndrome들을 계산한다.
            for time in range( round_num-2, -1, -1 ):
                
                # 두 측정 라운드 결과 사이의 parity를 확인한다.
                # parity 값을 syndrome 값으로 사용한다.
                syn_2 = stab_list[ time * stab_step_len : ( time + 1 ) * stab_step_len ]
                syn_1 = stab_list[ ( time + 1 ) * stab_step_len : ( time + 2 ) * stab_step_len ]
                syn = [ ( x + y ) % 2 for x, y in zip( syn_1, syn_2 ) ]
                syn_sample = self.flag_syn( syn)[::-1]
                syn_list.extend( syn_sample )
            
            # 데이터 측정 결과와 syndrome, data 측정 결과로 마지막 syndrome을 구성한다.
            data_list = stab_list[ round_num * stab_step_len : round_num * stab_step_len + self.distance ]
            syn_last_list = stab_list[ 0 : stab_step_len ]
            syn_sample = self.flag_data_syn( data_list, syn_last_list)[::-1]
            syn_list.extend( syn_sample )
            
            for syn_num in range(self.distance-1):
                syn_invert.extend( syn_list[syn_num::self.distance-1] )
                
            # 각 라운드별 얻어진 parity 결과들로 부터 오류의 정보를 제공하는 syndrome을 얻는다.
            syn_lists.append( [ syn_invert, np.sum( data_list ) % 2, int( stab_list[ -1 ] ) ] )
            
        return np.array(syn_lists, dtype=object)


# 양자 하드웨어로 부터 샘플로 얻어진 결과로 syndrome 그래프의 노드들 상관관계를 계산한다.
class correlation_matrix:
    """
    실제 양자 컴퓨터에서 얻은 qubit들의 측정값을 분석하면 이를 통해서 어떤 종류의 오류가 일어날 확률이 높은지를 계산해볼 수 있다.
    이렇게 얻어진 어떠한 오류가 일어날 확률을 통해서 우리는 MWPM에서 사용하는 graph의 weight를 결정할 수 있다.
    이를 위해서 우선 각 syndrome에서 1로 측정되는 사건이 얼마나 일어나는 지를 계산하고 이를 통해서 각 사건이 일어날 확률을 계산한다.
    syndrome_data: [각 round별로 syndrome을 계산한 결과, 오류를 수정하기 전에 data qubit을 측정하여 얻은 Observable, 이러한 측정이 몇번 이루어졌는지]가 여러 측정에 대해서 들어있는 array
    """    
    # 위의 함수들로 계산된 syndrome 데이터로 샘플들의 상관관계를 계산한다.
    def __init__( self, syndrome_data ):
        # [각 round별로 syndrome을 계산한 결과, 오류를 수정하기 전에 data qubit을 측정하여 얻은 Observable, 이러한 측정이 몇번 이루어졌는지]의 array
        self.syndrome_data = syndrome_data
        # shot 수
        self.syn_total_num = np.sum( self.syndrome_data[ : , 2 ] )
        # i번째 syndrome에서 오류가 탐지되었을 때 j번째 syndrome에서 오류가 탐지되는 확률을 나타내는 행렬
        self.x_matrix = self.xixj_matrix( )
        # i번째 syndrome과 j번째 syndrome이 탐지되게 하는 오류가 발생할 확률, 대각선 성분은 0으로 한다. 
        self.p_matrix = self.pij_matrix( )

    
    # 위의 함수를 사용하여 모든 syndrome 데이터에 대한 x matrix를 계산한다.
    def xixj_matrix( self ):                
        syndrome_data = self.syndrome_data
        # 한 round에 syndrome이 몇개 존재하는가 
        syn_num = len( syndrome_data[ 0 ][ 0 ] )
        # 비어있는 matirx, 크기는 syndrome의 수*syndrome의 수
        xixj_matrix = np.zeros( ( syn_num, syn_num ) )
        # 여러 측정 결과들에 대해서
        for num in range(len(syndrome_data)):
            # 측정 결과는 0 또는 1이기 때문에 np.outer함수를 사용하면 ij항은 i번째와 j번째 syndrome이 모두 1인 경우에만 1로 나온다.
            # 같은 측정이 여러번 반복된 경우 그 횟수만큼 값을 곱해준다. 
            xixj_matrix += np.outer(syndrome_data[num][0], syndrome_data[num][0])*syndrome_data[num][2]
        # 확률로 표현하기 위해서 전체 실험을 진행한 수만큼으로 나누어준다.
        return xixj_matrix / self.syn_total_num
    
    # 계산된 x matrix로 두 노드가 1로 얻어질 확률을 계산한다.
    def pij_matrix( self):
        # 계산된 syndrome 측정의 확률 분포
        x_matrix = self.x_matrix
        # 대각선 성분 값 (i번째 syndrome이 1로 측정될 확률)을 따로 계산한다.
        diagonal_element = np.diagonal(x_matrix)
        # 행렬 연산을 통해서 확률값을 계산하기 위해서 가로열 혹은 세로열이 대각선 성분들로 이루어진 행렬을 만든다.
        # 요소들의 연산에 대각선 성분이 사용되기 때문에 계산되는 요소에 맞게 행렬을 구성한다.
        xi = diagonal_element*np.ones(np.shape(x_matrix))
        xj = diagonal_element[:, np.newaxis]*np.ones(np.shape(x_matrix))
        xixj = x_matrix

        # 핵심은 i와 j의 기댓값과 p_i,p_j, p_ij의 관계식을 구성하는 것이다.
        # 근사를 사용한 식
        #pij = (xixj - xi*xj) / ((1 - 2*xi)*(1 - 2*xj))
        # 정확하게 계산한 식
        # *는 행렬의 각 요소별 곱셈이기 때문에 아래와 같이 행렬들의 연산으로 써도 같은 계산을 하게 된다.
        pij = 0.5 - 0.5 * np.sqrt( 1 - ( 4 * ( xixj - xi * xj ) / ( 1 - 2 * xi - 2 * xj + 4 * xixj ) ) )
        # 대각선 성분은 0이기 때문에 0으로 만들어 준다.
        mask = np.eye(pij.shape[0], dtype=bool)
        pij[mask] = 0
        pij = np.where(pij <0, 0, pij)
        #pij = np.where(pij<0, 1e-17, pij)
        # Create a mask of NaN values
        #nan_mask = np.isnan(pij)
        # Replace NaN values with the small float
        #pij[nan_mask] = 1e-17
        return pij
    
    # 만약 두 종류의 오류가 일어날 수 있다면 오류가 일어날 확률은 두 오류중 하나만 일어날 확률로 표현되어야 한다.
    # 만약 두 오류가 동시에 일어난다면 두 오류가 상쇄되어 오류가 없는 상태가 된다.
    def probability_sum(self, p, q ):
        # 1-[(1-p)*(1-q)+pq]
        # 둘중 하나에서만 오류가 일어날 확률
        # 이러한 방식으로 확률을 합치면 인접한 node에 의해서 오류가 발생할 확률을 계산할 수 있다.
        return p + q - 2 * p * q
    
    # boundary와의 weight는 인접한 다른 노드들의 확률 값을 모두 고려하여 계산한다.
    # xi로 계산된 x 번째 node에서 오류가 탐지될 확률에서 pij로 계산된 i번째 node와 연결된 다른 node와의 관계에서 생긴 오류를 빼는 것으로 
    # boundary node와 연결될 확률을 계산할 수 있다.
    # 해당 syndrome 노드와 인접한 다른 노드들의 리스트를 입력값으로 사용한다.
    def pib( self, i , j_list ):
        # i: decoding을 위한 graph상에 존재하는 node
        # j_list: node와 연결되어 있는 이웃한 node들의 list (boundary node는 제외)
        p_matrix = self.p_matrix
        x_matrix = self.x_matrix
        # node를 내림차순으로 정렬
        j_list.sort( reverse = True )
        # repetition code에서는 boundary에 있는 node와 인접한 node의 수는 최대 3개이다.
        j_list = j_list[ : 3 ]
        # boundary와 연결될 확률
        p_iB = 0
        # 각 syndrome에서 오류가 얼마나 일어났는가
        xi = x_matrix[ i, i ]

        # 인접한 노드들의 확률들을 고려하여 boundary 노드와 상관없는 확률 값 p를 얻는다.
        # 가상 노드를 제외하고 인접한 노드들의 p_ij를 이용하는데, p_iB가 음수가 되지 않도록 한다.
        #i번째 node가 j번째 node와 연관된 오류가 일어날 확률
        p_list = [ p_matrix[ i, x ] for x in j_list ]
        # i번째 node와 인접한 첫번째와 두번째 node에 대응되는 확률을 합친다.
        if len(p_list) >= 2:
            p = self.probability_sum( p_list[ 0 ], p_list[ 1 ] )
        else:
            p = p_list[0]
        # 만약 xi가 p보다 더 작다면 boundary node와 연결될 확률을 계산할 수 없다.
        # 따라서 인접한 node들중 연결될 확률이 가장 작은 값을 골라서 그것을 통해서 boundary와 연결될 확률을 계산한다.
        # 정확한 방법은 아니지만 임시적으로 구하기 위한 방법? 
        if xi < p :
            p = min( p_list )
        # 두 node에 의해서 오류가 일어날 확률을 합쳐도 xi보다 작다면 세번째 이후의 node에 의해서 오류가 일어날 확률을 계산해본다. 
        else:
            for j in p_list[ 2 : ]:
                p_new = self.probability_sum( p, j )
                # 세번째 이후의 node에 의해서 오류가 일어날 확률을 모두 합쳐도 그 노드에서 오류가 일어날 확률보다 작다면, 
                # 이 확률을 통해서 boundary node와 연결될 확률을 계산할 수 있다.
                # 만약 이 값이 그 node에서 오류가 일어날 확률 보다 높다면 처음 2개의 node만을 더한 확률 값을 통해서 boundary node와 연결될 확률을 계산한다.
                if p_new < xi :
                    p = p_new
        
        # p 값과 xi 값을 사용하여 weight를 계산한다.
        # xi=1-[(1-p_ib)(1-p)+p_ib*p] (probability sum)
        # -> p_ib = (p-xi)/(1-2p)
        # xi보다 p가 작으면 확률이 음수가 되기 때문에 이러한 계산을 할 수 없다. 
        p_iB = ( xi - p ) / ( 1 - 2 * p )
            
        return p_iB

class structure_utils:

    def __init__(self, distance_row, distance_col, arc = 'Lattice') -> None:
        self.distance_row = distance_row
        self.distance_col = distance_col

    # 본 python 코드에서 고려하는 Logical qubit은 Rotated Surface code이다.
    # 이를 위해 한 가운데의 qubit을 기준으로 다른 qubit들의 위치가 배치된다.
    # 이때, 왼쪽 상단에서 부터 큐빗 번호를 지정하기 위해 qubit 리스트를 sorting 해주는 알고리즘이 필요하다.
    # 다음과 같이 실수부로는 오름차순, 허수부는 내림차순으로 qubit 리스트를 재배치한다.
    def sorting_qubit( self, qubit_list ):
        
        qubit_list.sort( key = lambda v : v.real )
        qubit_list.sort( key = lambda v : v.imag, reverse=True )
        
        return qubit_list
    
    # 주어진 distance_row와 distance_col을 가지고 계산한 data qubit들의 위치를 나타내기 위함이다.
    def qubit_data( self ):
        
        distance_row = self.distance_row
        distance_col = self.distance_col
        
        # locq_structure의 output 중 하나인 qubit_num_data 변수를 딕셔너리 타입으로 정해준다.
        qubit_num_data : Dict[ complex, int ] = { }

        # rotated surface code에 맞게 모든 data qubit들은 사각형 형태로 배열된다.
        # 이를 위해 가운데 큐빗을 기준으로 상대적인 위치들을 찾기 위해 row와 col을 정의한다.
        row = ( 0.5 * ( distance_row - 1 ))
        col = ( 0.5 * ( distance_col - 1 ))
        
        # data qubit들의 위치들을 고려하여 리스트를 만들기 위해 data_x, data_y를 정의한다.
        data_x = np.arange( 0, 2 * 4 * row + 1, 4 )
        data_y = np.arange( 0, - 2 * 4 * col - 1, -4 )
        
        data_x_mid = np.arange( 2, 2 * 4 * row + 1, 4 )
        data_y_mid = np.arange( -2, - 2 * 4 * col - 1, -4 )
        
        # 위의 두 리스트로 부터 조합으로 distance_row와 distance_col 내에 위치할 수 있는 모든 data qubit들의 위치를 얻는다.
        data_position = list( product ( data_x, data_y ))
        data_position_mid = list( product ( data_x_mid, data_y_mid ))
        data_position_new = []
        
        # data qubit의 위치를 tuple의 형태가 아닌 complex 형태로 바꾸어 나타낸다.
        for x,y in data_position+data_position_mid:
            data_position_new.append( x + 1j * y )
        
        # data qubit의 번호를 왼쪽 상단에서부터 붙이기 위해 sorting 함수를 사용하여 큐빗 리스트를 재배치한다.
        data_position = self.sorting_qubit( data_position_new )

        # 모든 data qubit에 대해서, key는 qubit 위치를 나타내는 complex number, value는 qubit의 번호를 붙어 나타낸다.
        for num in range( len( data_position_new )):
            qubit_num_data[ data_position_new[ num ] ] = num
        
        # 다음 syndrome qubit의 번호를 붙이기 위해 num+1 를 결과에 내놓아 모든 큐빗 번호를 고려해준다.
        return qubit_num_data, num + 1

    def qubit_syn_1( self, num ):
        
        distance_row = self.distance_row
        distance_col = self.distance_col
        
        # output으로 나올 qubit_num_syn을 구성한다.
        qubit_num_syn : Dict[ complex, int ] = { }
        
        # syndrome qubit들의 상대적인 위치를 통해서 계산하기 때문에, 기준이 되는 qubit 리스트 구성이 필요하다.
        standard_qubit : List[ complex ] = [ ]
        syn_qubit : List = [ ]
        
        row = ( 0.5 * ( distance_row - 1 ))
        col = ( 0.5 * ( distance_col - 1 ))
        
        position_x = np.arange( 2, 2 * 4 * row + 1 , 4 )
        position_y = np.arange( 0, -2 * 4 * col - 1 , -4 )
    
        # 위의 두 리스트로 부터 조합으로 distance_row와 distance_col 내에 위치할 수 있는 모든 syn_1 qubit들의 위치를 얻는다.
        standard_qubit = list( product ( position_x, position_y ))
            
        # 기준이 되는 qubit을 중심으로 syndrome qubit들의 위치를 complex number로 찾아 syn_qubit 리스트에 포함시킨다.
        for x, y in standard_qubit:
            syn_qubit.append( x + 1j * y )
        
        # syndrome qubit의 리스트를 sorting하여 왼쪽 상단에서 부터 번호를 붙어준다.
        syn_qubit = self.sorting_qubit( syn_qubit )

        # 마지막으로 syn에 대한 qubit_num_syn을 결과로 내놓는다.
        for s in range( len( syn_qubit )):
            qubit_num_syn[ syn_qubit[ s ] ] = num
            num += 1
        
        # syndrome type를 고려하여 업데이트된 num을 뽑아 다음 qubit 역할에 대한 넘버링을 한다.
        return qubit_num_syn, num
    # 오류 정정 코드 내에 위치하는 syndrome 그룹에 따라 2개로 나누어 syn_1과 syn_2로 나눈다.
    # 각각에 따른 큐빗 역할 및 위치를 나타내는 딕셔너리를 구성하도록 함수를 만든다.
    def qubit_syn_2( self, num ):
        
        distance_row = self.distance_row
        distance_col = self.distance_col
        
        # output으로 나올 qubit_num_syn을 구성한다.
        qubit_num_syn : Dict[ complex, int ] = { }
        
        # syndrome qubit들의 상대적인 위치를 통해서 계산하기 때문에, 기준이 되는 qubit 리스트 구성이 필요하다.
        standard_qubit : List[ complex ] = [ ]
        syn_qubit : List = [ ]
        
        row = ( 0.5 * ( distance_row - 1 ))
        col = ( 0.5 * ( distance_col - 1 ))
    
        position_y = np.arange( -2, -2 * 4 * col - 1 , -4 )
        position_x = np.arange( 0, 2 * 4 * row + 1 , 4 )
        
    # 위의 두 리스트로 부터 조합으로 distance_row와 distance_col 내에 위치할 수 있는 모든 syn_2 qubit들의 위치를 얻는다.
        standard_qubit = list( product ( position_x, position_y ))
        
        # 기준이 되는 qubit을 중심으로 syndrome qubit들의 위치를 complex number로 찾아 syn_qubit 리스트에 포함시킨다.
        for x, y in standard_qubit:
            syn_qubit.append( x + 1j * y )
        
        # syndrome qubit의 리스트를 sorting하여 왼쪽 상단에서 부터 번호를 붙어준다.
        syn_qubit = self.sorting_qubit( syn_qubit )

        # 마지막으로 syn에 대한 qubit_num_syn을 결과로 내놓는다.
        for s in range( len( syn_qubit )):
            qubit_num_syn[ syn_qubit[ s ] ] = num
            num += 1
        
        # syndrome type를 고려하여 업데이트된 num을 뽑아 다음 qubit 역할에 대한 넘버링을 한다.
        return qubit_num_syn, num

    # Logical qubit의 구조가 격자 구조가 아닌 Heavy-hexagon인 경우에는 flag qubit들이 추가되어야 한다.
    def qubit_flag( self, qubit_syn_1, qubit_syn_2, num ):
        
        distance_row = self.distance_row
        distance_col = self.distance_col
        arc = self.arc
        
        flags : List[ complex ] = list()
        qubit_num_flag : Dict[ complex, int ] = { }
        
        row = int( 0.5 * ( distance_row - 1 ))
        col = int( 0.5 * ( distance_col - 1 ))
        
        # syndrome 타입에 따라 boundary에서 정의되는 flag qubit들의 구성이 다르다.
        # 따라서 boundary에서의 flag qubit들이 syndrome qubit 중심으로 어떠한 구성으로 되어있는지 정의한다.
        # 상대적인 위치 차이를 complex number들의 리스트로 나타낸 것이다.
        
        # syndrome I
        Top_boundary = [ -1, 1 ]
        Bottom_boundary = [ -1 + 2j, -1 +1j, -1, 1 ]
        
        # boundary가 아닌 경우에는 delta로 총 6개의 flag qubit들을 고려한다.
        # flag from syndrome
        delta = [ -1 + 2j, -1 +1j, -1, 1, 1-1j, 1 - 2j ]
        
        
        # 큐빗 배치 구조가 격자 구조가 아닌 경우에 대해서 다음과 같이 flag qubit들을 고려한다.
        if arc != 'Lattice':
        
            # boundary를 결정하는 complex number의 상대적인 기준 라인을 정의할 수 있다.
            Top = 0
            Bottom = -4 * (distance_col -1)
            
            # check top and bottom boundaries
            # syndrome qubit을 중심으로 모든 flag qubit들을 고려해서 boundary인 것들을 구분한다.
            for node in qubit_syn_1:
                image = node.imag
                if Top == image:
                    for rel in Top_boundary:
                        flags.append( node + rel )
                elif image == Bottom:
                    for rel in Bottom_boundary:
                        flags.append( node + rel )
            
            # 모든 boundary를 고려한 뒤에 boundary가 아닌 부분까지의 flag qubit들을 찾아 모든 flag qubit들을 고려한다.
            for node in qubit_syn_2:
                for rel in delta:
                    if node+rel not in flags:
                        flags.append( node + rel )
        
        # flag qubit 리스트를 sorting하여 flag qubit 리스트 배열을 재배치한다.
        flags = self.sorting_qubit( flags )
        
        # 찾아낸 모든 flag qubit들에 넘버링을 해준다.
        for f in range( len( flags )):
            qubit_num_flag[ flags[ f ]] = num
            num += 1

        # 마지막으로 qubit_num_flag을 결과로 내놓으며, 더 이상 사용하는 qubit이 없기에 num은 결과로 내놓지 않아도 된다.
        return qubit_num_flag

class encoder_utils:
    """
    여러 개의 qubit들에 대해서 같은 gate를 한번에 걸어주기 위해서 각 gate를 가해주는 함수를 저장해 놓는다.
    2 qubit gate들은 qubit list를 2개 단위로 끊어 각각 control과 target qubit으로 사용한다.
    qubit들간의 상대적 위치 관계를 기반으로 gate를 구성하거나 원하는 qubit을 찾기 위한 함수들을 구성한다. 
    """
    def __init__(self) -> None:
        pass
    
    def CNOT_operation(self, qubit_list):

        for i in np.arange( len( qubit_list[ : : 2 ] ) ) * 2:
            self.quantum_circuit.cx( qubit_list[ i ], qubit_list[ i + 1 ] )

    def CZ_operation(self, qubit_list):

        for i in np.arange( len( qubit_list[ : : 2 ] ) ) * 2:
            self.quantum_circuit.cz( qubit_list[ i ], qubit_list[ i + 1 ] )
    
    def H_operation(self, qubit_list):

        for i in np.arange( len( qubit_list ) ):
            self.quantum_circuit.h( qubit_list[ i ] )
    def X_operation(self, qubit_list):

        for i in np.arange( len( qubit_list ) ):
            self.quantum_circuit.x( qubit_list[ i ] )
    def Z_operation(self, qubit_list):

        for i in np.arange( len( qubit_list ) ):
            self.quantum_circuit.z( qubit_list[ i ] )
        
    # 현재 큐빗 번호와 상대적인 위치 차이를 input으로 받는다.
    # 현재 위치에서 상대적인 거리 delta만큼 떨어진 큐빗 번호를 찾는 함수이다.
    def find_qubit_num( self, current_qubit_num, delta ):
        
        qubits_num2loc, qubits_loc2num = self.qubits_num2loc, self.qubits_loc2num
        
        # 모든 큐빗의 위치와 번호에 대한 dictionary를 통해 위치의 상대적인 거리에 놓여있는 큐빗의 번호를 불러온다.
        # 없으면 무시하고 넘어간다.
        try:
            current_loc = qubits_num2loc[ current_qubit_num ]
            qubit_num = qubits_loc2num[ current_loc + delta ]

            return qubit_num

        # 해당 위치에 큐빗이 존재하지 않는다면, None을 내놓는다.
        except:
            return 

    # 현재 syndrome 큐빗의 중심으로 하나의 stabilizer 연산자의 고윳값을 측정을 위해 고려해야하는 큐빗을 찾는 함수이다.
    def patch_qubits( self, syndrome_qubit, delta ):

        # 주어진 syndrome 큐빗을 중심으로 고려해야하는 delta로 find_qubit_num을 사용하여 큐빗 리스트를 구성한다.
        qubits = [ ]
        for rel in delta:
            # 그 위치에 qubit이 존재하지 않을 경우 None 값을 포함할 수 있다.
            qubits.append( self.find_qubit_num( syndrome_qubit, rel ) )

        return qubits

    # 주어진 qubit들에 대해서 CNOT gate를 적용
    # qubit list에 None 값이 포함될 경우 그 gate를 적용하지 않는다.
    def CNOT_append(self, qubit_list, target_qubit_list):

        for num in range( 0, len( target_qubit_list ), 2 ):
            qubit_1 = target_qubit_list[ num ]
            qubit_2 = target_qubit_list[ num + 1 ]
            
            if qubit_1 != None and qubit_2 != None:
                qubit_list.append( qubit_1 )
                qubit_list.append( qubit_2 )

    # qubit_list에 target_qubit_list의 값들을 포함할 때, None인 값을 제거한다.
    def qubit_append( self, qubit_list, target_qubit_list):

        for qubit in target_qubit_list:
            if qubit != None:
                qubit_list.append( qubit )


if __name__ == '__main__':
    file_name = f'distance-9_rounds-10_flag_length-2_initial_state-+'
    file_path = f'./Results/ibm_kyoto_check/'
    # read data from raw data and turn it to syndrome string
    stab_data = pd.read_csv( file_path + file_name + '.csv', index_col = 0, skiprows = [ 0 ] ).values.tolist( )
    data = data_analysis(9,2).stab2syn( stab_data, 10 )
    # calculate the correlation matrix from data distribution
    correlation_M_real = correlation_matrix( data )
    p_matrix =correlation_M_real.p_matrix
    print(np.isnan(p_matrix).any())

    """
    #result_1 = {'101 101010':10, '011 000011': 4}
    result_2 = {'101 101010 101010':10, '011 000011 000111': 4}
    #data_1 = generator_utils().making_csv(distance=3, round_num=2, flag_length=1, initial_state='0',results=result_1)
    data_2 = generator_utils().making_csv(distance=3, round_num=2, flag_length=1, initial_state='0',results=result_2)
    data = [np.array(data_2.iloc[0]), np.array(data_2.iloc[1])]
    print(data)
    syn_arr = data_analysis(3,1).stab2syn(data, 2)
    print(syn_arr)
    """


