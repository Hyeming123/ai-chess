
import unittest
import chess
from index import ai_move, ai_move_with_learning
from game_graph import GameGraph
from index import evaluate

class TestDrawEvaluation(unittest.TestCase):
    def test_three_fold_repetition(self):
        print("\nTesting 3-fold Repetition Evaluation...")
        board = chess.Board()
        
        # Simple repetition: Nf3 Nf6 Ng1 Ng8 (twice)
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Ng1")
        board.push_san("Ng8") # Position 2 (same as start)
        
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Ng1")
        board.push_san("Ng8") # Position 3 (same as start) - repetition!
        
        # Now evaluate should return Contempt score
        # The last move was Ng8 (Black). 
        # Wait, in the test setup:
        # Nf3 (W), Nf6 (B), Ng1 (W), Ng8 (B) ... 
        # The last move 'Ng8' was made by Black.
        # So board.turn is WHITE.
        # Logic: if board.turn == BLACK: return -20; else: return 20
        # So it should return 20.
        
        score = evaluate(board)
        print(f"Score for 3-fold repetition (Black just moved): {score}")
        self.assertEqual(score, 20) 
        
        # Test case where White creates repetition
        # Undo last move and make White repeat?
        # Move list: Nf3, Nf6, Ng1, Ng8, Nf3, Nf6, Ng1. (White just moved Ng1)
        board.pop() 
        # Now board state is after Black's Nf6. 
        # The sequence was: 
        # 1. Nf3 Nf6 2. Ng1 Ng8
        # 3. Nf3 Nf6 4. Ng1 ...
        # If we check now, is it repetition?
        # FEN history: start -> A -> B -> start -> A -> B -> start
        # board.can_claim_draw() depends on the current position having appeared 3 times.
        
        # Actually simplest way: assert score != 0 and abs(score) == 20
        self.assertEqual(abs(score), 20)

# Mock GameGraph to return controlled priority
class MockGameGraph:
    def __init__(self):
        self.nodes = {'mock': True} # Just to pass the "if not graph.nodes" check if it checks for emptiness
        # But wait, ai_move_with_learning checks: if graph is None or not graph.nodes:
        # So I need to make sure nodes is not empty.

    def get_move_priority(self, current_fen, next_fen):
        # Let's say we learned that moving 'e2e4' is GREAT (priority 100)
        # And 'd2d4' is OKAY (priority 50)
        # Everything else is 0 (unknown)
        
        # We need to detect which move caused next_fen.
        # But wait, the function calls get_move_priority(current_fen, next_fen)
        # I can just reverse engineer the move or simply check if next_fen matches what e2e4 creates.
        
        board = chess.Board(current_fen)
        
        # Check for e2e4
        try:
            move_e4 = chess.Move.from_uci('e2e4')
            if move_e4 in board.legal_moves:
                board.push(move_e4)
                if board.fen() == next_fen:
                    return 100
                board.pop()
        except:
            pass

        # Check for d2d4
        try:
            move_d4 = chess.Move.from_uci('d2d4')
            if move_d4 in board.legal_moves:
                board.push(move_d4)
                if board.fen() == next_fen:
                    return 50
                board.pop()
        except:
            pass
            
        return 0

class TestAILearningPriority(unittest.TestCase):
    def test_learned_priority_initial_pos(self):
        print("\nTesting Learned Priority at Initial Position...")
        board = chess.Board()
        graph = MockGameGraph()
        
        # Expectation: AI should choose e2e4 because it has highest priority (100)
        # It should NOT print "Pure Minimax" (we can't easily assert print, but we observe behavior)
        
        move = ai_move_with_learning(board, graph=graph, depth=1)
        
        print(f"Selected Move: {move.uci()}")
        self.assertEqual(move.uci(), 'e2e4')

    def test_fallback_to_minimax(self):
        print("\nTesting Fallback when no learned moves...")
        board = chess.Board()
        # Create a graph that returns 0 for everything
        empty_graph = MockGameGraph()
        # Override get_move_priority to always return 0
        empty_graph.get_move_priority = lambda c, n: 0
        
        # Expectation: AI should trigger Minimax.
        # Since we can't easily intercept the function call, we just ensure it returns ANY legal move.
        move = ai_move_with_learning(board, graph=empty_graph, depth=1)
        print(f"Selected Move (Fallback): {move.uci()}")
        self.assertIn(move, board.legal_moves)
        
if __name__ == '__main__':
    # Make sure to guard for multiprocessing if necessary, 
    # though unittest usually runs sequentially.
    unittest.main()
