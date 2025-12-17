import chess
import pygame
import sys
from index import ai_move, ai_move_with_learning, evaluate
from game_graph import GameGraph

# 초기화
pygame.init()

# 상수
WIDTH = HEIGHT = 600
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15

# 색상
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# 이미지 로드
IMAGES = {}

def load_images():
    """체스 말 이미지를 PNG 파일에서 로드"""
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    
    for piece in pieces:
        # 파일명 생성: 대문자(백)는 'w', 소문자(흑)는 'b' 접두사
        color_prefix = 'w' if piece.isupper() else 'b'
        piece_name = piece.upper()  # 말 이름은 항상 대문자1
        filename = f"img/chesspieces/wikipedia/{color_prefix}{piece_name}.png"
        
        # 이미지 로드 및 크기 조정
        image = pygame.image.load(filename)
        IMAGES[piece] = pygame.transform.scale(image, (SQ_SIZE, SQ_SIZE))

class ChessGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chess AI')
        self.clock = pygame.time.Clock()
        self.board = chess.Board()
        self.selected_square = None
        self.player_clicks = []
        self.game_over = False
        self.ai_thinking = False
        
        # 게임 기록 초기화
        self.moves_history = []
        self.scores_history = []
        self.saved = False
        
        # 학습 데이터 로드
        self.graph = GameGraph()
        metadata = self.graph.load_from_file('store.txt')
        if metadata:
            print(f"학습 데이터 로드 완료: {metadata['total_games']}판, {metadata['total_nodes']}개 상태")
        else:
            print("학습 데이터 없음 - 기본 Minimax 사용")
        
    def draw_board(self):
        """체스보드 그리기"""
        colors = [WHITE, GRAY]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[((r+c) % 2)]
                pygame.draw.rect(self.screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
                
                # 좌표 표시 (왼쪽: 랭크, 아래쪽: 파일)
                font = pygame.font.SysFont("arial", 14, bold=True)
                
                # 랭크 표시 (1-8) - 첫 번째 열(c=0)에만 표시
                if c == 0:
                    text_color = colors[((r+c) % 2) ^ 1]  # 배경색과 반대색
                    label = font.render(str(8-r), True, text_color)
                    self.screen.blit(label, (c*SQ_SIZE + 2, r*SQ_SIZE + 2))
                
                # 파일 표시 (a-h) - 마지막 행(r=7)에만 표시
                if r == 7:
                    text_color = colors[((r+c) % 2) ^ 1]  # 배경색과 반대색
                    label = font.render(chr(ord('a') + c), True, text_color)
                    # 우측 하단 정렬
                    label_pos = (c*SQ_SIZE + SQ_SIZE - label.get_width() - 2, 
                               r*SQ_SIZE + SQ_SIZE - label.get_height() - 2)
                    self.screen.blit(label, label_pos)
    
    def draw_pieces(self):
        """체스 말 그리기"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # chess 라이브러리의 square를 화면 좌표로 변환
                col = chess.square_file(square)
                row = 7 - chess.square_rank(square)  # 체스보드는 위아래가 반대
                
                piece_symbol = piece.symbol()
                self.screen.blit(IMAGES[piece_symbol], 
                               pygame.Rect(col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))
    
    def highlight_square(self, square, color):
        """특정 칸을 하이라이트"""
        if square is not None:
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(color)
            self.screen.blit(s, (col*SQ_SIZE, row*SQ_SIZE))
    
    def draw_game_state(self):
        """게임 화면 그리기"""
        self.draw_board()
        
        # 선택된 칸 하이라이트
        if self.selected_square is not None:
            self.highlight_square(self.selected_square, YELLOW)
            
            # 가능한 이동 표시
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    self.highlight_square(move.to_square, GREEN)
        
        self.draw_pieces()
        
        # 게임 상태 표시
        font = pygame.font.Font(None, 36)
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            text = font.render(f"Checkmate! {winner} wins!", True, RED)
            self.screen.blit(text, (WIDTH//2 - 150, HEIGHT//2))
        elif self.board.is_stalemate():
            text = font.render("Stalemate!", True, BLUE)
            self.screen.blit(text, (WIDTH//2 - 80, HEIGHT//2))
        elif self.board.is_check():
            text = font.render("Check!", True, RED)
            self.screen.blit(text, (10, 10))
        
        # 턴 표시
        turn_text = "White" if self.board.turn == chess.WHITE else "Black"
        text = font.render(f"Turn: {turn_text}", True, BLACK)
        self.screen.blit(text, (WIDTH - 150, 10))
        
        if self.ai_thinking:
            text = font.render("AI thinking...", True, BLUE)
            self.screen.blit(text, (WIDTH//2 - 100, HEIGHT - 50))
    
    def get_square_under_mouse(self):
        """마우스 위치의 체스보드 square 반환"""
        mouse_pos = pygame.mouse.get_pos()
        col = mouse_pos[0] // SQ_SIZE
        row = mouse_pos[1] // SQ_SIZE
        # 화면 좌표를 chess square로 변환
        square = chess.square(col, 7 - row)
        return square
    
    def select_promotion(self, color):
        """승진할 말 선택 UI"""
        # 반투명 배경
        s = pygame.Surface((WIDTH, HEIGHT))
        s.set_alpha(128)
        s.fill((0, 0, 0))
        self.screen.blit(s, (0, 0))
        
        # 선택지 표시
        options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        option_images = []
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        
        button_size = SQ_SIZE * 1.5
        total_width = button_size * 4 + 20 * 3
        start_x = center_x - total_width // 2
        
        buttons = []
        for i, piece_type in enumerate(options):
            # 이미지 가져오기 (color_prefix가 'w' 또는 'b')
            piece_char = chess.Piece(piece_type, color).symbol()
            img = IMAGES[piece_char]
            img = pygame.transform.scale(img, (int(button_size * 0.8), int(button_size * 0.8)))
            
            # 버튼 영역
            rect = pygame.Rect(start_x + i * (button_size + 20), center_y - button_size // 2, button_size, button_size)
            pygame.draw.rect(self.screen, WHITE, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 2)
            
            # 이미지 그리기
            img_rect = img.get_rect(center=rect.center)
            self.screen.blit(img, img_rect)
            buttons.append((rect, piece_type))
        
        pygame.display.flip()
        
        # 선택 대기 루프
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for rect, piece_type in buttons:
                        if rect.collidepoint(pos):
                            return piece_type
    
    def handle_click(self, square):
        """마우스 클릭 처리"""
        if self.game_over:
            return
        
        # 첫 번째 클릭 (말 선택)
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            # 현재 턴의 말만 선택 가능
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                # 디버깅: 선택한 말의 가능한 이동 출력
                print(f"\n선택: {chess.square_name(square)} ({piece.symbol()})")
                legal_moves = [m for m in self.board.legal_moves if m.from_square == square]
                print(f"가능한 이동: {[m.uci() for m in legal_moves]}")
                if any(self.board.is_en_passant(m) for m in legal_moves):
                    print("--> 앙파상 가능!")
        else:
            # 두 번째 클릭 (이동)
            move = chess.Move(self.selected_square, square)
            
            # 프로모션 감지 (유효성 검사 전)
            promoted_piece_type = None
            if (self.board.piece_at(self.selected_square).piece_type == chess.PAWN and
                (chess.square_rank(square) == 0 or chess.square_rank(square) == 7)):
                
                # 가상의 퀸 프로모션으로 유효성 검사
                temp_move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                if temp_move in self.board.legal_moves:
                    # 유효하면 사용자에게 선택 요청
                    promoted_piece_type = self.select_promotion(self.board.turn)
                    move = chess.Move(self.selected_square, square, promotion=promoted_piece_type)
            
                    move = chess.Move(self.selected_square, square, promotion=promoted_piece_type)
            
            # 일반 이동 또는 선택된 프로모션 이동 검사
            if move in self.board.legal_moves or (promoted_piece_type and move in self.board.legal_moves):
                # 기록: 이동 전 상태 및 평가 점수
                fen_before = self.board.fen()
                score = evaluate(self.board)
                self.scores_history.append(score)
                
                self.board.push(move)
                
                # 기록: 이동 후 상태 및 이동
                fen_after = self.board.fen()
                self.moves_history.append((fen_before, move.uci(), fen_after))
                
                self.selected_square = None
                
                # AI 차례
                if not self.board.is_game_over():
                    self.ai_thinking = True
            else:
                # 이동이 유효하지 않은 경우
                # 클릭한 위치에 내 말이 있는지 확인 (다른 말 선택)
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    print(f"\n재선택: {chess.square_name(square)} ({piece.symbol()})")
                else:
                    print(f"이동 불가: {move.uci()} (Legal moves: {[m.uci() for m in self.board.legal_moves]})")
                    self.selected_square = None
    
    def run(self):
        """메인 게임 루프"""
        load_images()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.ai_thinking:
                    square = self.get_square_under_mouse()
                    self.handle_click(square)
            
            if self.ai_thinking and self.board.turn == chess.BLACK:
                # 학습 데이터를 활용한 AI 이동
                # 기록: 이동 전 상태
                fen_before = self.board.fen()
                
                ai_move_result = ai_move_with_learning(self.board, self.graph, depth=4)
                if ai_move_result:
                    # 기록: 평가 점수 (이동 전)
                    score = evaluate(self.board)
                    self.scores_history.append(score)
                    
                    self.board.push(ai_move_result)
                    
                    # 기록: 이동 후 상태 및 이동
                    fen_after = self.board.fen()
                    self.moves_history.append((fen_before, ai_move_result.uci(), fen_after))
                    
                self.ai_thinking = False
            
            if self.board.is_game_over():
                self.game_over = True
                if self.board.is_checkmate():
                    print("체크메이트 감지됨!")
                elif self.board.is_stalemate():
                    print("스텔메이트 감지됨!")
                
                # 게임 결과 저장 (한 번만)
                if not self.saved:
                    result = self.board.result()
                    print(f"\n게임 종료! 결과: {result}")
                    print("게임 데이터를 저장하는 중...")
                    
                    self.graph.add_game(self.moves_history, self.scores_history, result)
                    metadata = self.graph.save_to_file('store.txt')
                    
                    print(f"저장 완료! 총 게임 수: {metadata['total_games']}")
                    self.saved = True
            
            self.draw_game_state()
            pygame.display.flip()
            self.clock.tick(MAX_FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = ChessGame()
    game.run()
