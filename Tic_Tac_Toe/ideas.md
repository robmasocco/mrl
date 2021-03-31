# TIC TAC TOE
## Markov Decision Process
### 1. States
- Each board is a state.
- A board is represented witha a 3x3 matrix.
- Each cell can be filled with None, X or O, numerically represented with 0, 1 and 2.
- Number of possible boards = 3^9 = 19683.
- Many boards aren't possible:
    - Number of Xs is strictly less than that of Os.
    - Number of Xs minus number of Os is strictly less than 0 or strictly greater than 1.
- Many boards are equal:
    - Rotations of 90 degree.
- Next player to play: if the number of Xs minus the number of Os is equal to 0 is the X player's turn, else if it's 1 is the O player's turn, else not valid board.
- Termianl states: 
    - Three Xs or Os in a horizontal, vertical or diagonal row (8 possibilities).
    - The board is full and no winner, it's a draw.
    - How many terminal states?
- The state associated with a board is the representation in base 3 of the number obtained by flattering the matrix.
- Define the states only for valid e unique boards.
- Create a lookup table, and a reverse lookup table!
- Each board has some info:
    1. The type of the board: -1 not valid, 1 terminal, 0 otherwise.
    2. If it's terminal the winner or the next player to play.
    3. The associated state. For each board, search if a its rotation matches with a previous one, otherwise define a new state.
- Use **afterstates**: a tranisiton from a state s to a new state s' depens on the moves of the Agent and of the Opponent.

### 2. Actions
- An action is where to palce a mark on the borad.
- Number of actions = 9
- Each state has a different set of actions.
- The Opponent takes action with uniform probability.

### 3. Transition probabilities
- Transition matrix P with dimensions *S x S x A*
- Use the *Law of Total Probability* to build P:
    - The Agent's move depends on the actions taken. -> ***P_X(s, s', a)***
    - The Opponent's move doesn't depens on the actions. -> ***P_O(s, s')***
    - ***P = P_X * P_O***

### 4. Rewards
- +1 for an action that leads to a winning board.
- -1 for an action that leads to a losing board.
- 0 for any other actions.
- Expected rewards matrix R with dimentions *S x A*.
- Use again the *Law of Total Probability* to build R:
    - How to compute it?

## Planning
- To be continued...
