"""
    Blackjack game
"""

import numpy as np

class Blackjack:

    def __init__(self, init_s, init_a, pi):
        # State of the game.
        self.s = init_s
        # First action to take.
        self.init_a = init_a
        # Game set up.
        self.hand, self.dealer, self.ace = np.asarray(np.unravel_index(self.s, (10, 10, 2))) + [12, 1, 0]
        # Game policy.
        self.pi = pi
        # Data of the episode.
        self.states = np.array(init_s, dtype=np.int32)
        self.actions = np.empty(0, dtype=np.float64)
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
                self.ace = 0
            else:
                # Count the ace as 11.
                self.ace = 1
        else:
            # Not ace.
            self.hand += card
            if self.hand > 21 and self.ace == 1:
                self.hand -= 10
                self.ace = 0
        # New state
        if self.hand > 21:
            # Burst.
            self.s = -1
            return -1.0
        else:
            self.s = np.ravel_multi_index((self.hand - 12, self.dealer - 1, self.ace), (10, 10, 2))
            return 0.0
    
    def stick(self):
        # Check if the dealer has an ace.
        if self.dealer == 1:
            self.dealer += 10
            ace = 1
        else:
            ace = 0
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
                    ace = 0
                else:
                    ace = 1
            else:
                # Not Ace.
                self.dealer += card
                if self.dealer > 21 and ace == 1:
                    self.dealer -= 10
                    ace = 0
        # New state
        if self.dealer > 21:
            # Burst.
            self.s = -1
            return 1
        else:
            if self.dealer > self.hand:
                # Defeat.
                self.s = -1
                return -1
            elif self.dealer == self.hand:
                # Draw
                self.s = -1
                return 0
            else:
                # Win.
                self.s = -1
                return 1


    def play(self):
        print("Hand: {}\tDealer: {}".format((self.hand, self.ace), self.dealer))
        while self.s != -1:
            # Choose action.
            if self.actions.size == 0:
                a = self.init_a
            else:
                a = self.pi[self.s]
            #
            if a == 0:
                # Hit
                r = self.hit()
                print("Hand: {}\tDealer: {}".format((self.hand, self.ace), self.dealer))
            else:
                # Stick
                r = self.stick()
                print("Hand: {}\tDealer: {}".format((self.hand, self.ace), self.dealer))
            # Store
            self.states = np.append(self.states, self.s)
            self.actions = np.append(self.actions, self.init_a)
            self.rewards = np.append(self.rewards, r)
        
    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_rewards(self):
        return self.rewards