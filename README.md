# Chess AI

파이썬으로 만든 체스 AI 게임입니다. Minimax 알고리즘과 Pygame을 사용하여 AI와 체스 대결을 즐길 수 있습니다.

## 기능

- 🤖 **Minimax AI**: Alpha-Beta 가지치기를 사용한 강력한 AI
- 📊 **위치 평가**: Piece-Square Tables로 전략적 위치 판단
- 🎮 **Pygame 인터페이스**: 직관적인 클릭 기반 플레이
- ✨ **시각적 피드백**: 이동 가능한 위치 하이라이트, 체크 알림
- 📜 **완전한 체스 규칙**: 모든 표준 체스 규칙 지원

## 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
python main.py
```

## 플레이 방법

1. **자기 차례의 말을 클릭**하여 선택
2. **초록색으로 표시된 칸을 클릭**하여 이동
3. AI가 자동으로 응수합니다

## 파일 구조

- `main.py` - Pygame 게임 인터페이스
- `index.py` - AI 로직 (Minimax, 평가 함수)
- `requirements.txt` - 필요한 라이브러리

## 기술 스택

- Python 3
- Pygame (게임 인터페이스)
- python-chess (체스 로직)

## AI 알고리즘

- **Minimax with Alpha-Beta Pruning**: 2수 앞을 내다보는 전략적 AI
- **평가 함수**: 말의 가치 + 위치 보너스
- **Piece-Square Tables**: 각 말의 이상적인 위치 정의

---

즐거운 체스 게임 되세요! ♟️
