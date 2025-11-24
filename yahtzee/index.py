import random

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.items:
            return self.items.pop()
        return None

    def peek(self):
        if self.items:
            return self.items[-1]
        return None

    def size(self):
        return len(self.items)

    def is_empty(self):
        return len(self.items) == 0

    def display(self):
        return self.items[:]  

dice = Stack()       
stored = Stack()    


def roll_dice(count):
    input("Press Enter to roll the dice...")
    for _ in range(count):
        dice.push(random.randint(1, 6))


def display_dice():
    print("\n=== Current Dice Stack ===")
    print(dice.display())

    print("=== Stored Dice Stack ===")
    print(stored.display())
    print()


def store():
    """dice 스택에서 pop → stored에 push"""
    if not dice.is_empty():
        stored.push(dice.pop())
        print("Stored one dice!")
    else:
        print("No dice to store.")


def out():
    """stored 스택에서 pop → dice에 push"""
    if not stored.is_empty():
        dice.push(stored.pop())
        print("Returned one dice!")
    else:
        print("No stored dice.")


def main():
    roll_dice(5)
    display_dice()

    reroll_count = 0

    while reroll_count < 3:
        print("1: store top dice")
        print("2: return stored -> dice")
        print("3: reroll remaining dice")
        print("0: finish")
        choice = int(input("Select: "))

        if choice == 1:
            store()

        elif choice == 2:
            out()

        elif choice == 3:
            remain = dice.size()
            dice.items.clear()
            roll_dice(remain)
            reroll_count += 1

        elif choice == 0:
            break

        display_dice()


main()