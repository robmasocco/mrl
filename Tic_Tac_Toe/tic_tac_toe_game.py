#!/usr/bin/python3

"""
Tic-Tac-Toe CLI game.
Authors: Roberto Masocco, Alessandro Tenaglia, Giacomo Solfizi
Date: 16/4/2021
"""

import numpy as np
from tic_tac_toe_aux_funcs import *

def user_action(actions_O):
    """Gets an action from the user."""
    while True:
        a_O = int(input("Enter number from 0 to 8: "))
        if a_O in actions_O:
            break
    return a_O

def play(pi, player):
    """User-driven game loop."""
    global state_id_lkt, id_state_lkt
    s = 0
    id = state_id_lkt[s, 2]
    print_board(id_to_board(id))
    while True:
        # X's turn.
        info = id_state_lkt[id]
        a_X = map_action(pi[info[4]], info[2], info[3])[0]
        id += 3 ** (8 - a_X)
        print_board(id_to_board(id))
        if id_state_lkt[id, 0] == 1:
            if id_state_lkt[id, 1] == 0:
                print("DRAW")
            elif id_state_lkt[id, 1] == 1:
                print("WIN")
            else:
                print("LOSS, wtf?")
            break
        # O's turn.
        actions_O = get_actions(id)
        if player == 1:
            a_O = user_action(actions_O)
        else:
            a_O = np.random.choice(actions_O)
        id += 2 * (3 ** (8 - a_O))

def main():
    """Game menu."""
    global state_id_lkt, id_state_lkt
    state_id_lkt = np.load("ttt_s2id.dat", allow_pickle=True)
    id_state_lkt = np.load("ttt_id2s.dat", allow_pickle=True)
    print("######### TIC-TAC-TOE #########")
    while True:
        print("Select the policy to play against from:")
        print("\t1 - Policy Iteration.")
        print("\t2 - Value Iteration.")
        ans = int(input("Make your choice [1/2]: "))
        if ans not in [1, 2]:
            print("ERROR: Invalid choice.")
            continue
        if ans == 1:
            pi = np.load("ttt_p_pi.dat", allow_pickle=True)
        else:
            pi = np.load("ttt_p_vi.dat", allow_pickle=True)
        print("Do you want to play or watch a game against the uniform opponent?")
        print("\t1 - Play.")
        print("\t2 - Watch uniform opponent.")
        ans = int(input("Make your choice [1/2]: "))
        if ans not in [1, 2]:
            print("ERROR: Invalid choice.")
            continue
        if ans == 1:
            play(pi, 1)
        else:
            play(pi, 2)
        while True:
            ans = str(input("\nWanna play again? [y/n] "))
            if ans == "y":
                break
            elif ans == "n":
                print("Goodbye!")
                exit()
            else:
                print("ERROR: Invalid choice.")
                continue

if __name__ == '__main__':
    main()
