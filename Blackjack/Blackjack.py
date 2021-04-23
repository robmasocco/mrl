"""
    Blackjack game
"""

import numpy as np

class Blackjack:

    def __init__(self, init_s, pi, epsilon):
        # State of the game.
        self.s = init_s
        # Game set up.
        self.hand, self.dealer, self.ace = np.asarray(np.unravel_index(self.s, (10, 10, 2), order='F')) + [12, 1, 1]
        # Policy.
        self.pi = pi
        # Degree of exploration.
        self.epsilon = epsilon
        # Data of the episode.
        self.states = np.empty(0, dtype=np.int32)
        self.actions = np.empty(0, dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.float64)

    def hit(self):
        # Draw a card.
        card = np.min([np.random.randint(13) + 1, 10])
        # Check its value.
        if card == 1:
            # Ace.
            self.hand += 11
            if self.hand > 21:
                # Count the ace as 1.
                self.hand -= 10
                self.ace = 2
            else:
                # Count the ace as 11.
                self.ace = 1
        else:
            # Not ace.
            self.hand += card
            if self.hand > 21 and self.ace == 1:
                self.hand -= 10
                self.ace = 2
        # Next state.
        if self.hand > 21:
            # Burst.
            self.s = -1
            return -1.0
        else:
            self.s = np.ravel_multi_index((self.hand - 12, self.dealer - 1, self.ace - 1), (10, 10, 2), order='F')
            return 0.0
    
    def stick(self):
        # Check if the dealer has an ace.
        if self.dealer == 1:
            self.dealer += 10
            ace = 1
        else:
            ace = 2
        # Hit 
        while self.dealer < 17:
            # Draw a card.
            card = np.min([np.random.randint(13) + 1, 10])
            # Check its value.
            if card == 1:
                # Ace.
                self.dealer += 11
                if self.dealer > 21:
                    self.dealer -= 10
                    ace = 2
                else:
                    ace = 1
            else:
                # Not Ace.
                self.dealer += card
                if self.dealer > 21 and ace == 1:
                    self.dealer -= 10
                    ace = 2
        # New state
        if self.dealer > 21:
            # Burst.
            self.s = -1
            return 1.0
        else:
            if self.dealer > self.hand:
                # Defeat.
                self.s = -1
                return -1.0
            elif self.dealer == self.hand:
                # Draw
                self.s = -1
                return 0.0
            else:
                # Win.
                self.s = -1
                return 1.0

    def play(self):
        while self.s != -1:
            # Store the state.
            self.states = np.append(self.states, self.s)
            # Choose action.
            if np.random.choice([0, 1], p=[1.0 - self.epsilon, self.epsilon]) == 0:
                a = self.pi[self.s]
            else:
                a = np.random.choice([0, 1])
            # Store the action taken.
            self.actions = np.append(self.actions, a)
            # Play the action.
            if a == 0:
                # Hit
                r = self.hit()
            else:
                # Stick
                r = self.stick()
            # Store the reward obtained.
            self.rewards = np.append(self.rewards, r)
        
    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_rewards(self):
        return self.rewards
