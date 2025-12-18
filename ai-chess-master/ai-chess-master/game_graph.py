import json
import hashlib
import chess
from collections import defaultdict


class GameGraph:
    """
    체스 게임의 학습 데이터를 그래프 형식으로 저장하는 클래스
    
    노드(Node): 보드 상태 (FEN 표기법)
    엣지(Edge): 수(move)와 평가 점수
    """
    
    def __init__(self):
        # 노드: FEN -> {evaluation, visit_count, outcomes}
        self.nodes = {}
        
        # 엣지: (from_fen, to_fen) -> {move, count, avg_score}
        self.edges = defaultdict(lambda: {
            'move': None,
            'count': 0,
            'scores': []
        })
        
        # 게임 결과 저장 (현재 세션)
        self.game_results = []
        
        # 전체 누적 결과 (파일 로드 시 복원됨)
        self.total_results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    
    def _fen_hash(self, fen):
        """FEN 문자열의 해시값 생성 (중복 제거용)"""
        # 무브 카운터와 50수 규칙 카운터 제거 (동일한 위치로 간주)
        parts = fen.split()
        core_fen = ' '.join(parts[:4])  # 보드 상태, 차례, 캐슬링, 앙파상만 유지
        return hashlib.md5(core_fen.encode()).hexdigest()[:16]
    
    def add_game(self, moves_history, scores_history, result):
        """
        게임 데이터를 그래프에 추가
        
        Args:
            moves_history: [(fen_before, move, fen_after), ...]
            scores_history: [score1, score2, ...]
            result: "1-0", "0-1", "1/2-1/2"
        """
        self.game_results.append(result)
        if result in self.total_results:
            self.total_results[result] += 1
        
        for i, (fen_before, move, fen_after) in enumerate(moves_history):
            fen_before_hash = self._fen_hash(fen_before)
            fen_after_hash = self._fen_hash(fen_after)
            
            # 노드 추가/업데이트
            if fen_before_hash not in self.nodes:
                self.nodes[fen_before_hash] = {
                    'fen': fen_before,
                    'visit_count': 0,
                    'evaluations': [],
                    'outcomes': {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
                }
            
            if fen_after_hash not in self.nodes:
                self.nodes[fen_after_hash] = {
                    'fen': fen_after,
                    'visit_count': 0,
                    'evaluations': [],
                    'outcomes': {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
                }
            
            # 방문 횟수 증가
            self.nodes[fen_before_hash]['visit_count'] += 1
            
            # 평가 점수 추가
            if i < len(scores_history):
                self.nodes[fen_before_hash]['evaluations'].append(scores_history[i])
            
            # 게임 결과 추가
            self.nodes[fen_before_hash]['outcomes'][result] += 1
            
            # 엣지 추가/업데이트
            edge_key = (fen_before_hash, fen_after_hash)
            
            # UCI 수를 SAN(Algebraic Notation)으로 변환
            # move는 UCI 문자열 (예: "e2e4")
            try:
                board = chess.Board(fen_before)
                move_obj = chess.Move.from_uci(move)
                san_move = board.san(move_obj)
                self.edges[edge_key]['move'] = san_move
            except ValueError:
                # 변환 실패 시 원본 유지
                self.edges[edge_key]['move'] = move
            
            self.edges[edge_key]['count'] += 1
            if i < len(scores_history):
                self.edges[edge_key]['scores'].append(scores_history[i])
    
    def remove_duplicates(self):
        """중복 제거 및 평균 점수 업데이트"""
        # 1. 노드 평균 업데이트
        for node_hash, node_data in self.nodes.items():
            new_evals = node_data['evaluations']
            if not new_evals:
                continue
                
            new_count = len(new_evals)
            new_sum = sum(new_evals)
            
            # 기존 데이터가 있는 경우 가중 평균 계산
            if 'avg_evaluation' in node_data:
                old_avg = node_data['avg_evaluation']
                total_count = node_data['visit_count']
                old_count = total_count - new_count
                
                if old_count > 0:
                    # (기존총점 + 신규총점) / 전체횟수
                    current_avg = (old_avg * old_count + new_sum) / total_count
                else:
                    # 혹시 모를 에러 방지 (visit_count가 new_count보다 작을 순 없지만)
                    current_avg = new_sum / new_count
            else:
                # 신규 데이터만 있는 경우
                current_avg = new_sum / new_count
            
            # 업데이트
            node_data['avg_evaluation'] = current_avg
            
            # 메모리 절약을 위해 평가 리스트 비우기 (평균에 반영되었으므로)
            # 단, 시각화나 디버깅을 위해 최근 10개는 남겨둘 수도 있지만,
            # 여기서는 파일 저장 직전이므로 초기화하거나 최소한으로 유지
            node_data['evaluations'] = [] 

        # 2. 엣지 평균 업데이트
        for edge_key, edge_data in self.edges.items():
            new_scores = edge_data['scores']
            if not new_scores:
                continue
                
            new_count = len(new_scores)
            new_sum = sum(new_scores)
            
            if 'avg_score' in edge_data:
                old_avg = edge_data['avg_score']
                total_count = edge_data['count']
                old_count = total_count - new_count
                
                if old_count > 0:
                    current_avg = (old_avg * old_count + new_sum) / total_count
                else:
                    current_avg = new_sum / new_count
            else:
                current_avg = new_sum / new_count
            
            edge_data['avg_score'] = current_avg
            edge_data['scores'] = []
    
    def save_to_file(self, filename='store.txt'):
        """그래프 데이터를 파일에 저장"""
        self.remove_duplicates()
        
        # 저장할 데이터 구조
        data = {
            'metadata': {
                'total_games': sum(self.total_results.values()),
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
                'results_summary': self.total_results
            },
            'nodes': {
                node_hash: {
                    'fen': node_data['fen'],
                    'visits': node_data['visit_count'],
                    'avg_eval': node_data.get('avg_evaluation', 0),
                    'outcomes': node_data['outcomes']
                }
                for node_hash, node_data in self.nodes.items()
            },
            'edges': {
                f"{from_hash}->{to_hash}": {
                    'move': edge_data['move'],
                    'count': edge_data['count'],
                    'avg_score': edge_data.get('avg_score', 0)
                }
                for (from_hash, to_hash), edge_data in self.edges.items()
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return data['metadata']
    
    def load_from_file(self, filename='store.txt'):
        """파일에서 그래프 데이터 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return None
                data = json.loads(content)
            
            # 메타데이터에서 전체 결과 복원
            if 'metadata' in data and 'results_summary' in data['metadata']:
                self.total_results = data['metadata']['results_summary']
            
            # 노드 복원
            for node_hash, node_data in data.get('nodes', {}).items():
                self.nodes[node_hash] = {
                    'fen': node_data['fen'],
                    'visit_count': node_data['visits'],
                    'evaluations': [],  # 평균만 유지
                    'outcomes': node_data['outcomes']
                }
                if 'avg_eval' in node_data:
                    self.nodes[node_hash]['avg_evaluation'] = node_data['avg_eval']
            
            # 엣지 복원
            for edge_str, edge_data in data.get('edges', {}).items():
                from_hash, to_hash = edge_str.split('->')
                edge_key = (from_hash, to_hash)
                self.edges[edge_key] = {
                    'move': edge_data['move'],
                    'count': edge_data['count'],
                    'scores': []
                }
                if 'avg_score' in edge_data:
                    self.edges[edge_key]['avg_score'] = edge_data['avg_score']
            
            return data.get('metadata', None)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return None
    
    def get_move_priority(self, current_fen, next_fen):
        """
        특정 수의 우선순위 점수를 계산
        
        Args:
            current_fen: 현재 보드 상태 (FEN)
            next_fen: 수를 둔 후 보드 상태 (FEN)
        
        Returns:
            우선순위 보너스 점수 (0~1000)
        """
        current_hash = self._fen_hash(current_fen)
        next_hash = self._fen_hash(next_fen)
        edge_key = (current_hash, next_hash)
        
        # 학습 데이터가 없으면 0 반환
        if edge_key not in self.edges or next_hash not in self.nodes:
            return 0
        
        edge_data = self.edges[edge_key]
        next_node = self.nodes[next_hash]
        
        priority = 0
        
        # 1. 방문 횟수 보너스 (0~300점)
        visit_count = edge_data.get('count', 0)
        if visit_count > 0:
            # 로그 스케일로 변환하여 과도한 가중치 방지
            import math
            priority += min(300, math.log(visit_count + 1) * 50)
        
        # 2. 평균 평가 점수 보너스 (0~400점)
        avg_score = edge_data.get('avg_score', 0)
        # 평가 점수를 -10000~10000 범위로 가정하고 정규화
        normalized_score = max(-1, min(1, avg_score / 5000))
        priority += (normalized_score + 1) * 200  # 0~400 범위
        
        # 3. 승률 보너스 (0~300점)
        outcomes = next_node.get('outcomes', {'1-0': 0, '0-1': 0, '1/2-1/2': 0})
        total_games = sum(outcomes.values())
        
        if total_games > 0:
            # 현재 차례에 따라 승리 판단
            # FEN의 두 번째 필드가 'w'면 백 차례, 'b'면 흑 차례
            current_turn = current_fen.split()[1]
            
            if current_turn == 'w':  # 백 차례
                win_rate = outcomes.get('1-0', 0) / total_games
                loss_rate = outcomes.get('0-1', 0) / total_games
            else:  # 흑 차례
                win_rate = outcomes.get('0-1', 0) / total_games
                loss_rate = outcomes.get('1-0', 0) / total_games
            
            draw_rate = outcomes.get('1/2-1/2', 0) / total_games
            priority += win_rate * 300 + draw_rate * 10
            
            # 4. 패배 페널티 (패배 회피 로직 강화)
            # 사용자의 요청: "2번만" -> 2번 이상 패배한 경우 강력하게 회피
            loss_count = outcomes.get('0-1', 0) if current_turn == 'w' else outcomes.get('1-0', 0)
            
            if loss_count >= 2:
                # 2번 이상 졌으면 아주 강력한 페널티 부여
                priority -= 5000 * loss_rate
            else:
                # 일반적인 패배 페널티
                priority -= loss_rate * 500
        
        return priority
    
    def get_best_moves(self, current_fen, move_list):
        """
        가능한 수들에 대한 우선순위 정보 반환
        
        Args:
            current_fen: 현재 보드 상태 (FEN)
            move_list: [(move, next_fen), ...] 리스트
        
        Returns:
            {move: priority_score, ...}
        """
        priorities = {}
        for move, next_fen in move_list:
            priorities[move] = self.get_move_priority(current_fen, next_fen)
        return priorities
    
    def get_statistics(self):
        """통계 정보 반환"""
        if not self.nodes:
            return "그래프가 비어있습니다."
        
        stats = f"""
=== 체스 AI 학습 데이터 통계 ===
총 게임 수: {sum(self.total_results.values())}
유니크 보드 상태 수: {len(self.nodes)}
총 수(moves) 수: {len(self.edges)}

게임 결과:
  백 승리: {self.total_results.get('1-0', 0)}
  흑 승리: {self.total_results.get('0-1', 0)}
  무승부: {self.total_results.get('1/2-1/2', 0)}

가장 많이 방문한 상태 Top 5:
"""
        # 방문 횟수로 정렬
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: x[1]['visit_count'],
            reverse=True
        )[:5]
        
        for i, (node_hash, node_data) in enumerate(sorted_nodes, 1):
            stats += f"  {i}. {node_hash}: {node_data['visit_count']}회 방문\n"
        
        return stats

    def visualize_statistics(self, filename='learning_stats.png'):
        """
        학습 데이터를 시각화하여 차트로 저장
        
        Args:
            filename: 저장할 이미지 파일명
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            
            # 한글 폰트 설정 (Windows)
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            print("matplotlib이 설치되지 않았습니다. 'pip install matplotlib' 실행")
            return
        
        if not self.nodes:
            print("시각화할 데이터가 없습니다.")
            return
        
        # Figure 생성 (2행 2열)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('체스 AI 학습 데이터 통계', fontsize=16, fontweight='bold')
        
        # 1. 방문 횟수 Top 10 막대 그래프
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: x[1]['visit_count'],
            reverse=True
        )[:10]
        
        if sorted_nodes:
            node_labels = [f"State {i+1}" for i in range(len(sorted_nodes))]
            visit_counts = [node_data['visit_count'] for _, node_data in sorted_nodes]
            
            ax1.bar(node_labels, visit_counts, color='steelblue', alpha=0.8)
            ax1.set_title('방문 횟수 Top 10 보드 상태', fontweight='bold')
            ax1.set_xlabel('보드 상태')
            ax1.set_ylabel('방문 횟수')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
        
        # 2. 게임 결과 원형 차트
        total_games = sum(self.total_results.values())
        if total_games > 0:
            results_count = {
                '백 승리': self.total_results.get('1-0', 0),
                '흑 승리': self.total_results.get('0-1', 0),
                '무승부': self.total_results.get('1/2-1/2', 0)
            }
            
            colors = ['#4CAF50', '#F44336', '#FFC107']
            explode = (0.05, 0.05, 0.05)
            
            ax2.pie(results_count.values(), labels=results_count.keys(), 
                   autopct='%1.1f%%', startangle=90, colors=colors, 
                   explode=explode, shadow=True)
            ax2.set_title('게임 결과 분포', fontweight='bold')
        
        # 3. 통계 요약 텍스트
        ax3.axis('off')
        stats_text = f"""
        === 학습 데이터 요약 ===
        
        총 게임 수: {sum(self.total_results.values())}
        유니크 보드 상태: {len(self.nodes)}
        총 수(moves): {len(self.edges)}
        
        게임 결과:
          • 백 승리: {self.total_results.get('1-0', 0)}
          • 흑 승리: {self.total_results.get('0-1', 0)}
          • 무승부: {self.total_results.get('1/2-1/2', 0)}
        
        평균 방문 횟수: {sum(n['visit_count'] for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0:.1f}
        """
        ax3.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 4. 노드/엣지 수 비교 막대 그래프
        categories = ['보드 상태\n(Nodes)', '수\n(Edges)']
        counts = [len(self.nodes), len(self.edges)]
        colors_bar = ['#2196F3', '#FF9800']
        
        ax4.bar(categories, counts, color=colors_bar, alpha=0.8, width=0.6)
        ax4.set_title('그래프 구조 통계', fontweight='bold')
        ax4.set_ylabel('개수')
        ax4.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for i, (cat, count) in enumerate(zip(categories, counts)):
            ax4.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n차트가 저장되었습니다: {filename}")

