import chess
import random

import multiprocessing

# Top-level worker function for multiprocessing
def worker_minimax(fen, move_uci, depth, is_maximizing_target):
    """
    Worker function to evaluate a single move in a separate process.
    Reconstructs board state and calls minimax.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    board.push(move)
    
    # After pushing the move, we call minimax for the 'next' player.
    # If the target player (who made the move) was maximizing, then inside minimax:
    # the turn has switched, so we need to maintain the correct perspective.
    # original minimax signature: minimax(board, depth, alpha, beta, maximizing_player)
    # The 'maximizing_player' arg in minimax usually refers to "is the current turn's player maximizing?".
    # But in the provided minimax implementation:
    # if board.turn == chess.WHITE: (Max) ... else: (Min) ...
    # The 'maximizing_player' argument effectively acts as a flag or is unused in the recursion base on turn check.
    # Let's check the existing minimax again.
    # It checks `if board.turn == chess.WHITE:` to decide logic.
    # The `maximizing_player` arg is passed recursively but logic seems to rely on `board.turn`.
    # So we just call it with dummy boolean or consistent one.
    
    # We want to return the evaluation from the perspective of the ROOT player?
    # No, minimax returns absolute score.
    # We just need to return that score.
    
    score = minimax(board, depth - 1, -float('inf'), float('inf'), not is_maximizing_target)
    return move_uci, score

def minimax(board, depth, alpha, beta, maximizing_player):
    """
    Minimax 알고리즘 with Alpha-Beta 가지치기
    """                 
    # 종료 조건: 게임이 끝났거나 깊이가 0
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    
    # 백 차례 (Maximizing Player)
    if board.turn == chess.WHITE:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta 가지치기
        return max_eval
    
    # 흑 차례 (Minimizing Player)
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha 가지치기
        return min_eval


def get_minimax_scores(board, depth=4):
    """
    모든 가능한 수에 대해 Minimax 점수를 계산하여 반환
    Returns:
        {move_uci: score, ...}
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {}

    # 백(White)은 점수를 최대화, 흑(Black)은 점수를 최소화
    is_maximizing = (board.turn == chess.WHITE)
    
    # 병렬 처리를 위한 인자 준비
    fen = board.fen()
    tasks = [(fen, move.uci(), depth, is_maximizing) for move in legal_moves]
    
    # 프로세스 풀 생성 및 실행
    with multiprocessing.Pool() as pool:
        results = pool.starmap(worker_minimax, tasks)
        
    return dict(results)

def ai_move(board, depth=4):
    """
    Minimax를 사용하여 최선의 수를 선택 (병렬 처리 적용)
    """
    move_scores = get_minimax_scores(board, depth)
    if not move_scores:
        return None
        
    is_maximizing = (board.turn == chess.WHITE)
    
    # 최적의 점수 찾기
    if is_maximizing:
        best_score = max(move_scores.values())
    else:
        best_score = min(move_scores.values())
    
    # 최적의 수들 수집
    best_moves = []
    for move_uci, score in move_scores.items():
        if score == best_score:
            best_moves.append(chess.Move.from_uci(move_uci))
            
    return random.choice(best_moves)


def ai_move_with_learning(board, graph=None, depth=4, learning_weight=0.15):
    """
    학습 데이터를 활용하여 최선의 수를 선택
    방식: Minimax 탐색 점수 + (학습 우선순위 * 가중치)
    """
    if graph is None or not graph.nodes:
        return ai_move(board, depth)
    
    # 1. 모든 수에 대해 Minimax 탐색 수행 (Depth 5 가정)
    #    학습된 데이터가 있어도 무조건 탐색하여 현재 상황을 판단
    minimax_scores = get_minimax_scores(board, depth)
    if not minimax_scores:
        return None

    current_fen = board.fen()
    final_scores = []
    
    is_maximizing = (board.turn == chess.WHITE)
    
    print(f"\n=== AI 생각 중 (Depth {depth} + Learning) ===")
    
    for move_uci, minimax_score in minimax_scores.items():
        move = chess.Move.from_uci(move_uci)
        
        # 2. 학습 데이터에서 우선순위 점수 가져오기
        board.push(move)
        next_fen = board.fen()
        # priority는 0~1000점 범위, 높을수록 좋음 (승률 기반)
        priority = graph.get_move_priority(current_fen, next_fen)
        board.pop()
        
        # 3. 점수 결합
        # Minimax 점수(폰=100)와 Priority(0~1000)를 합산
        # Priority 100점당 폰 1개(100) 정도의 가치를 주려면 weight=1.0
        # 여기서는 상황에 따라 튜닝 필요. 일단 0.5 정도로 설정 (Priority 200점 = 폰 1개)
        # 백은 점수가 클수록 좋고(Add), 흑은 작을수록 좋으므로(Subtract) 방향성 고려
        
        weighted_priority = priority * 0.5
        
        if is_maximizing:
            # 백: Minimax(양수 유리) + Priority(양수 유리)
            final_score = minimax_score + weighted_priority
        else:
            # 흑: Minimax(음수 유리) - Priority(양수 유리 -> 낮춰야 함)
            # Priority는 항상 '좋은 수'에 대해 양수이므로, 흑 입장에서는 점수를 낮춰야(더 음수로) 이득
            final_score = minimax_score - weighted_priority
            
        final_scores.append((move, final_score, minimax_score, priority))
    
    # 4. 최종 점수로 정렬 및 선택
    # 백: 점수 높은 순, 흑: 점수 낮은 순
    final_scores.sort(key=lambda x: x[1], reverse=is_maximizing)
    
    # 상위 3개 출력
    print(f"Top 3 Moves (Turn: {'White' if is_maximizing else 'Black'}):")
    for i, (m, f_score, m_score, p_score) in enumerate(final_scores[:3], 1):
        print(f"  {i}. {m.uci()}: Final={f_score:.1f} (Minimax={m_score}, Priority={p_score:.1f})")
        
    best_move = final_scores[0][0]
    return best_move



def play_game(graph=None, depth=4):
    """
    AI 대 AI 게임 실행 및 학습 데이터 수집
    
    Args:
        graph: 학습 데이터 그래프 (None이면 순수 Minimax)
        depth: Minimax 탐색 깊이
        
    Returns:
        result: 게임 결과 ("1-0", "0-1", "1/2-1/2")
        moves_history: [(fen_before, move, fen_after), ...]
        scores_history: [score1, score2, ...]
    """
    board = chess.Board()
    moves_history = []
    scores_history = []

    while not board.is_game_over():
        # 현재 보드 상태 저장
        fen_before = board.fen()
        
        # AI가 수를 선택하고 평가 점수도 기록
        if graph:
            # 학습 데이터가 있으면 활용
            move = ai_move_with_learning(board, graph, depth)
        else:
            # 없으면 순수 Minimax
            move = ai_move(board, depth)
        
        # 수를 두기 전 평가 점수 계산
        score = evaluate(board)
        scores_history.append(score)
        
        # 수를 둔다
        board.push(move)
        
        # 이동 후 보드 상태
        fen_after = board.fen()
        
        # 이동 정보 저장 (UCI 형식)
        moves_history.append((fen_before, move.uci(), fen_after))

    return board.result(), moves_history, scores_history

# Piece-Square Tables (백 기준, 흑은 뒤집어서 사용)
pawn_table = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

knight_table = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

bishop_table = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

rook_table = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

queen_table = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

king_middle_game_table = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

king_end_game_table = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

piece_square_tables = {
    chess.PAWN: pawn_table,
    chess.KNIGHT: knight_table,
    chess.BISHOP: bishop_table,
    chess.ROOK: rook_table,
    chess.QUEEN: queen_table,
    chess.KING: king_middle_game_table
}

def evaluate(board):
    """
    보드를 평가하는 함수
    현재 차례 플레이어의 관점에서 점수를 반환
    말의 가치 + 위치 보너스를 모두 고려
    """
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -20000  # 백이 체크메이트 당함 -> 흑 승리 -> 매우 낮은 점수
        else:
            return 20000   # 흑이 체크메이트 당함 -> 백 승리 -> 매우 높은 점수
    
    # 무승부 조건 확인 (스텔메이트, 기물 부족, 3회 반복, 50수 규칙)
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        # Contempt Factor: 무승부 회피를 위한 페널티
        # 반복을 만든(직전 수를 둔) 플레이어에게 불리한 점수 부여
        # 현재 턴이 흑(BLACK)이면 -> 백(WHITE)이 방금 둠 -> 백에게 불리하게 (-20)
        # 현재 턴이 백(WHITE)이면 -> 흑(BLACK)이 방금 둠 -> 흑에게 불리하게 (+20)
        
        CONTEMPT = 20
        if board.turn == chess.BLACK:
            return -CONTEMPT
        else:
            return CONTEMPT
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    score = 0
    
    # 엔드게임 여부 확인 (퀸이 없거나 기물 점수가 낮을 때)
    # 간단하게 양쪽 퀸이 없으면 엔드게임으로 간주
    w_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    b_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
    is_endgame = (w_queens == 0 and b_queens == 0)

    # 각 말의 가치와 위치 보너스 계산
    for piece_type in piece_values:
        # 테이블 선택
        table = piece_square_tables.get(piece_type)
        if piece_type == chess.KING and is_endgame:
            table = king_end_game_table
        
        if table is None:
            continue

        # 백의 말들 (+ 점수)
        for square in board.pieces(piece_type, chess.WHITE):
            score += piece_values[piece_type]
            score += table[square]
        
        # 흑의 말들 (- 점수)
        for square in board.pieces(piece_type, chess.BLACK):
            score -= piece_values[piece_type]
            flipped_square = 63 - square  # 보드 좌우상하 반전
            score -= table[flipped_square]
    
    return score

# 여러 판 대국 및 학습 데이터 수집
if __name__ == "__main__":
    from game_graph import GameGraph
    
    # 기존 학습 데이터 로드 (있다면)
    graph = GameGraph()
    existing_metadata = graph.load_from_file('store.txt')
    
    if existing_metadata:
        print("기존 학습 데이터 로드 완료:")
        print(f"  - 기존 게임 수: {existing_metadata['total_games']}")
        print(f"  - 기존 노드 수: {existing_metadata['total_nodes']}")
        print(f"  - 기존 엣지 수: {existing_metadata['total_edges']}")
        print()
    
    # 새로운 게임 실행
    print("AI 대국 시작 (100판)...")
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    
    for i in range(1):
        # 깊이 5로 설정하고, 기존 학습 데이터(graph)를 활용하여 게임 진행
        result, moves_history, scores_history = play_game(graph, depth=2)
        results[result] += 1
        
        # 학습 데이터를 그래프에 추가
        graph.add_game(moves_history, scores_history, result)
        1
        # 진행 상황 출력 (한 판씩 결과 표시)
        print(f"  Game {i + 1}/100: HO08 {result} KYY08")
    
    print("\n게임 결과:")
    print(results)
    
    # 그래프 데이터를 파일에 저장 (중복 자동 제거)
    print("\n학습 데이터를 store.txt에 저장 중...")
    metadata = graph.save_to_file('store.txt')
    
    print("\n저장 완료!")
    print(f"  - 총 게임 수: {metadata['total_games']}")
    print(f"  - 유니크 보드 상태 수: {metadata['total_nodes']}")
    print(f"  - 총 수(moves) 수: {metadata['total_edges']}")
    print(f"  - 백 승리: {metadata['results_summary']['1-0']}")
    print(f"  - 흑 승리: {metadata['results_summary']['0-1']}")
    print(f"  - 무승부: {metadata['results_summary']['1/2-1/2']}")
    
    # 통계 출력
    print(graph.get_statistics())
    
    # 학습 데이터 시각화
    print("\n학습 데이터를 차트로 시각화 중...")
    graph.visualize_statistics('learning_stats.png')