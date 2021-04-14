"""
Auxiliary functions for Tic Tac Toe.
Authors: Alessandro Tenaglia, Roberto Masocco, Giacomo Solfizi
Date: March 31, 2021
"""

import numpy as np

def board_to_id(board):
    """Takes a board matrix and returns a base-10 board ID."""
    id = 0
    for i, elem in enumerate(board.flatten()):
        id += int(elem * (3 ** (8 - i)))
    return id

def board_to_ids(board):
    """Takes a board matrix and returns the IDs of all its rotations and symmetries with the relative transformations."""
    # Define an empty matrix for the IDs.
    ids = np.empty([0, 3], dtype=np.int32)
    # No flip.
    for i in range(4):
        id = board_to_id(np.rot90(board, i))
        # Append the ID only if not present.
        if not id in ids[:, 0]:
            ids = np.append(ids, [[id, 0, i]], axis=0)
    # Flip left-right.
    flipped_board = np.fliplr(board)
    for j in range(4):
        id = board_to_id(np.rot90(flipped_board, j))
        # Append the ID only if not present.
        if not id in ids[:, 0]:
            ids = np.append(ids, [[id, 1, j]], axis=0)
    # Return the sorted matrix.
    return ids[ids[:,0].argsort()]

def id_to_board(id):
    """Takes a base-10 board ID and returns a board matrix."""
    board_str = np.base_repr(id, base=3).zfill(9)
    return np.array(list(board_str), dtype=np.int8).reshape(3, 3)

def find_win(board, marker):
    """Takes a board matrix and checks if there are 3 equal markers in a row horizontal, vertical or diagonal."""
    # Checks the rows.
    for row in board:
        if np.all(row == marker):
            return True
    # Checks the columns.
    for col in board.T:
        if np.all(col == marker):
            return True
    # Checks the diagonal.
    diag = np.diagonal(board)
    if np.all(diag == marker):
        return True
    # Checks the anti-diagonal.
    flipped_board = np.fliplr(board)
    anti_diag = np.diagonal(flipped_board)
    if np.all(anti_diag == marker):
        return True
    # No winning combinations.
    return False

def board_info(board):
    """Takes a board matrix and returns its information: terminal, valid or invalid board and the winner or the next player."""
    xs = np.count_nonzero(board == 1)
    os = np.count_nonzero(board == 2)
    # Switch according to the difference of the markers.
    diff = xs - os
    if diff == 1:
        # Last player to move was X.
        if find_win(board, 2):
            return -1, -1
        if find_win(board, 1):
            return 1, 1
        else:
            # Board is full.
            if xs == 5:
                return 1, 0
            else:
                return 0, 2
    elif diff == 0:
        # Last player to move was O.
        if find_win(board, 1):
            return -1, -1
        if find_win(board, 2):
            return 1, 2
        else:
            return 0, 1
    else:
        return -1, -1

def map_action(action, flip, rot):
    """Takes an action and apples flip and rotations."""
    # Create a dummy matrix of zeros with a 1.
    flat_board = np.zeros(9, dtype=np.int32)
    flat_board[action] = 1
    board = flat_board.reshape(3, 3)
    # Flip the matrix.
    if flip == 1:
        flipped_board = np.fliplr(board)
    else:
        flipped_board = board
    # Rotate the matrix.
    rotated_B = np.rot90(flipped_board, rot)
    # Find the new postion of the 1.
    new_action = np.argmax(rotated_B)
    new_indices = np.unravel_index(new_action, (3, 3))
    return new_action, new_indices

def get_actions(id):
    """Takes a id and returns the possible actions to be taken."""
    flat_board = id_to_board(id).flatten()
    return np.where(flat_board == flat_board.min())[0]

def print_board(B):
    for i in range(3):
        for j in range(3):
            if B[i, j] == 1:
                print(" X ", end="")
            elif B[i, j] == 2:
                print(" O ", end="")
            else:
                print("   ", end="")
            if j != 2:
                print("|", end="")
            else:
                print()
        if i != 2:
            print("---+---+---")
    print("###########")
