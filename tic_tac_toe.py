"""
Tic Tac Toe

TODO
* Refactor code to make it more AI tournament-friendly
* ai_move function must only take in game state next time
    ^ re factor as class? (keep track of game_tree and trash talk flag)
"""
import collections
from functools import partial
import logging
import random

import numpy as np

try:
    import ujson as json
except ImportError:
    import json


GAME_TREE_FILE = "tic-tac-toe_game_tree.json"


class TicTacToe(object):
    """TicTacToe object representing a game state

    Notes:
    * game states are encoded as 9-character strings
        * valid characters are "1", "2", and "_"
    """

    def __init__(self, state, symbols=("O", "X")):
        """
        PARAMETERS
        state (str): 9 character string representing state
        symbols (tuple): 2-element tuple representing game symbols
            * tuple in first position goes first; vice versa
        """
        assert len(symbols) == 2, "`symbols` must have exactly 2 elements"

        self.state = state
        self.counter = collections.Counter(state)
        self.validate_state(state)

        self.symbols = symbols
        self.data = np.array(list(state))
        self.board = self.data.reshape((3, 3))
        self.game_status = self.check_game_status()
        self.next_moves = self.find_next_states()
        self.next_states = list(self.next_moves.values())

    def validate_state(self, state):
        """Check if a state is valid"""
        assert isinstance(state, str), "`state` must be of type str"
        assert len(state) == 9, "`state` must have exactly 9 elements"

        state = state.upper()
        assert set(state) <= {
            "1",
            "2",
            "_",
        }, "`state` elements must be in {'1', '2', '_'}"
        assert (
            0 <= self.counter["1"] - self.counter["2"] <= 1
        ), "Invalid state: Players must alternate."

    def check_game_status(self):
        """Check status of game (turn, tie, win)"""
        for player in ("1", "2"):
            row_win = np.apply_along_axis(
                lambda x: set(x) == {player}, 1, self.board
            ).any()
            col_win = np.apply_along_axis(
                lambda x: set(x) == {player}, 0, self.board
            ).any()
            d1_win = set(self.data[[0, 4, 8]]) == {player}
            d2_win = set(self.data[[2, 4, 6]]) == {player}
            if any([row_win, col_win, d1_win, d2_win]):
                return ("win", player)

        if self.counter["_"] == 0:
            return ("tie", None)
        else:
            return ("turn", "1" if self.counter["1"] == self.counter["2"] else "2")

    def find_next_states(self):
        """Determine possible next moves. Returns a dict {index: new_state}"""
        status, player = self.game_status
        moves = {}
        if status == "turn":
            for idx in np.where(self.data == "_")[0]:
                new_move = self.data.copy()
                new_move[idx] = player
                moves[idx] = "".join(new_move)

        return moves

    def printable_board(self, indent_char="\t", legend_hint=True, symbols=None):
        """Returns a string representing the game board for printing"""
        symbols = symbols or self.symbols
        assert len(symbols) == 2, "`symbols` must have exactly 2 elements"

        data_symbols = self.data.copy()
        for orig, new in zip(("1", "2"), symbols):
            data_symbols[data_symbols == orig] = new

        board_symbols = data_symbols.reshape((3, 3))

        if legend_hint:
            legend_board = np.where(
                self.data == "_", range(9), " ").reshape((3, 3))
            return "\n".join(
                [indent_char + "GAME   |  INDEX"]
                + [indent_char + "=====  |  ====="]
                + [
                    indent_char + " ".join(b_row) + "  |  " + " ".join(l_row)
                    for b_row, l_row in zip(board_symbols, legend_board)
                ]
            )
        else:
            return "\n".join([indent_char + " ".join(row) for row in board_symbols])


def gen_game_tree(state_init):
    """Generate full game tree from initial state"""
    current_path = [state_init]
    game_tree = {}
    while current_path:
        cur_state = current_path[-1]
        if cur_state not in game_tree:
            ttt = TicTacToe(cur_state)
            game_tree[cur_state] = {
                "unexplored": ttt.next_states,
                "explored": [],
                "status": ttt.game_status,
            }

        if game_tree[cur_state]["unexplored"]:
            current_path.append(game_tree[cur_state]["unexplored"].pop(0))
        else:
            explored = current_path.pop(-1)
            if explored != state_init:
                game_tree[current_path[-1]]["explored"].append(explored)

            status, player = game_tree[cur_state]["status"]
            if status == "tie":
                value = 0
                outcomes = {0: 1}
            elif status == "win":
                value = -1 if player == "1" else 1
                outcomes = {value: 1}
            else:
                value = (min if player == "1" else max)(
                    [
                        game_tree[state]["value"]
                        for state in game_tree[cur_state]["explored"]
                    ]
                )
                outcomes = {}
                for state in game_tree[cur_state]["explored"]:
                    for res, res_ct in game_tree[state]["outcomes"].items():
                        outcomes[res] = outcomes.get(res, 0) + res_ct

            game_tree[cur_state]["value"] = value
            game_tree[cur_state]["outcomes"] = outcomes

    return game_tree


def answer_exercise():
    """Function to answer exercise in Chapter 2 section III"""
    state = "1212__21_"
    ttt = TicTacToe(state)
    game_tree = gen_game_tree(state)
    print(
        f"The value of game state:\n{ttt.printable_board('    ', legend_hint=False)}\n"
        f"is {game_tree[state]['value']}"
    )
    return game_tree[state]["value"]


def learn(state="_________"):
    """Build game tree and export given initial state"""
    game_tree = gen_game_tree(state)
    with open(GAME_TREE_FILE, "w") as gt_file:
        json.dump(game_tree, gt_file, indent=4)


def human(gstate: TicTacToe, *args):
    """Function for a human player"""
    return input_with_validation("Please enter move.", list(gstate.next_moves.keys()))


def ai_w_level(gstate: TicTacToe, game_tree, level=3):
    """AI with levels

    Level Descriptions
    * 0 = stupid
    * 1 = easy
    * 2 = medium
    * 3 = hard
    * 4 = unfair
    """
    assert isinstance(level, int), "`level` must be `int`"
    assert 0 <= level <= 4, "level values must be from 0 to 4"

    seed = random.random()
    logging.debug(f"seed value: {seed:.3f}")

    if level == 0:
        ai_func = ai_derp
    elif level == 1:
        ai_func = ai_derp if seed <= 0.3 else ai_strategy1
    elif level == 2:
        ai_func = ai_derp if seed <= 0.2 else ai_strategy2
    elif level == 3:
        ai_func = ai_derp if seed <= 0.1 else ai_strategy3
    elif level == 4:
        ai_func = ai_strategy3

    return ai_func(gstate, game_tree)


def ai_derp(gstate: TicTacToe, *args):
    """AI that randomly picks the next move"""
    return random.choice(list(gstate.next_moves.keys()))


def ai_strategy1(gstate: TicTacToe, game_tree):
    """Strategy assuming opponent plays optimally"""
    status, player = gstate.game_status

    if status != "turn":
        logging.warning("Game status = %s. No move needed.", status)
        return None

    mod = -1 if player == "1" else 1
    next_move_vals = {
        idx: mod * game_tree[state]["value"] for idx, state in gstate.next_moves.items()
    }
    max_val = max(next_move_vals.values())
    moves = [idx for idx, val in next_move_vals.items() if val == max_val]
    logging.debug("moves: %s; value: %i", moves, max_val)
    move = random.choice(moves)

    return move


def ai_strategy2(gstate: TicTacToe, game_tree):
    """Strategy maximizing the number of paths to winning end states"""
    status, player = gstate.game_status

    if status != "turn":
        logging.warning("Game status = %s. No move needed.", status)
        return None

    win, lose = (-1, 1) if player == "1" else (1, -1)
    next_move_vals = {
        idx: win * game_tree[state]["value"] for idx, state in gstate.next_moves.items()
    }
    max_val = max(next_move_vals.values())
    moves = {
        idx: (
            game_tree[gstate.next_moves[idx]]["outcomes"].get(str(win), 0),
            game_tree[gstate.next_moves[idx]]["outcomes"].get(str(0), 0),
            game_tree[gstate.next_moves[idx]]["outcomes"].get(str(lose), 0)
        )
        for idx, val in next_move_vals.items() if val == max_val
    }

    win_ct = {idx: vals[0] for idx, vals in moves.items()}
    win_pct = {idx: vals[0] / sum(vals) for idx, vals in moves.items()}
    lose_pct = {idx: vals[2] / sum(vals) for idx, vals in moves.items()}
    wl_ratio = {idx: vals[0] / max(vals[2], 0.5)
                for idx, vals in moves.items()}

    # criteria, agg_func = lose_pct, min
    # criteria, agg_func = win_pct, max
    criteria, agg_func = wl_ratio, max

    if max_val == 1 and 1 in win_ct.values():
        move = [idx for idx, val in win_ct.items() if val == 1][0]
    else:
        move = random.choice(
            [idx for idx, val in criteria.items() if val ==
             agg_func(criteria.values())]
        )

    logging.debug("move: %i; value: %i, win paths %%: %.1f%%, lose paths %%: %.1f%%, moves: %s\n",
                  move, max_val, win_pct[move] * 100, lose_pct[move] * 100, moves)

    # trash talk
    if max_val == 1:
        print(
            "*beep* *boop* *beep*"
            " -=[ I calculate chances of winning to be 100% ]=- "
            "*beep* *boop* *beep*"
        )

    return move


def ai_strategy3(gstate: TicTacToe, game_tree):
    """AI strategy that maximizes the opponent's losing moves in the next turn"""
    status, player = gstate.game_status

    if status != "turn":
        logging.warning("Game status = %s. No move needed.", status)
        return None

    win, lose = (-1, 1) if player == "1" else (1, -1)

    move_vals = {
        move: win * game_tree[state]["value"] for move, state in gstate.next_moves.items()
    }
    max_val = max(move_vals.values())

    if max_val == 1:
        # ai_strategy2 can handle "won" states
        return ai_strategy2(gstate, game_tree)

    else:
        ok_moves = [move for move, val in move_vals.items() if val == max_val]

        move_vals = {
            move: collections.Counter(
                [
                    game_tree[state2]["value"] for state2 in game_tree[gstate.next_moves[move]]["explored"]
                ]
            )
            for move in ok_moves
        }

        move_eval = {
            move: val_ctr.get(win, 0) / val_ctr.get(lose, 0.5)
            for move, val_ctr in move_vals.items()
        }
        max_win_pct = max(move_eval.values())
        good_moves = [move for move, win_pct in move_eval.items() if win_pct ==
                      max_win_pct]

        move = random.choice(good_moves)

        all_ct = sum(move_vals[move].values())
        win_ct = move_vals[move].get(win, 0)
        logging.debug(
            "move: %i; value: %i, win %%: %.1f%%, moves: %s\n",
            move, win * game_tree[gstate.state]["value"],
            win_ct / max(all_ct, 0.1) * 100, move_vals
        )

        return move


def input_with_validation(text, choices):
    """Take input with validation"""
    choice_vals = set(map(str, choices))
    while True:
        val = input(f"{text} | choices={choices}: ")
        if val in choice_vals:
            return val
        else:
            print(f"{val} is not a valid value. Please choose from: {choices}")


def start_game(player1, player2, symbols=("O", "X")):
    """Starts a command line tic tac toe game between 2 players (bot or human)"""

    assert len(symbols) == 2, "`symbols` must have exactly 2 elements"

    gstate = TicTacToe("_________", symbols=symbols)
    with open(GAME_TREE_FILE, "r") as gt_file:
        game_tree = json.load(gt_file)

    while True:
        status, player = gstate.game_status
        if status == "turn":
            print(f"\n=== Player {player}'s turn:\n\n")
            print(gstate.printable_board(legend_hint=True))
            print("\n")
            if player == "1":
                p_move = player1(gstate, game_tree)
            else:
                p_move = player2(gstate, game_tree)

            print(f"\n>>> Player {player} has chosen: {p_move}")
            new_state = gstate.data.copy()
            new_state[int(p_move)] = player
            gstate = TicTacToe("".join(new_state), symbols=symbols)

        else:
            print('\n')
            print(gstate.printable_board(legend_hint=True))

            if status == "win":
                print(f"\n~~~~~ Player {player} wins! ~~~~~\n")
            else:
                print("\n~~~~~ It's a Tie! ~~~~~\n")

            return gstate.game_status


def menu(n_player=None, symbols=("O", "X"), ai_1=None, ai_2=None):
    """start CLI based game menu

    PARAMETERS
    n_player (int): number of players
        * 0 = ai_1 vs ai_2; order decided at random
        * 1 = human vs ai_1; order decided by player
        * 2 = human vs human
    symbols (tuple): 2 element tuple for board symbols (in order)
    ai_1, ai_2 (callable): ai function that takes the following as parameters:
        * gstate (TicTacToe): game state
        * game_tree (dict): game tree JSON
            * TODO: in the future, each AI class should store its own game tree
    """

    assert len(symbols) == 2, "`symbols` must have exactly 2 elements"

    ai_1 = ai_1 or ai_strategy3
    ai_2 = ai_2 or ai_derp

    print("\n***   Let's play TIC TAC TOE!   ***\n")
    n_player = n_player if n_player is not None else int(
        input_with_validation("Choose number of players.", ["0", "1", "2"])
    )

    if n_player < 2:
        if n_player == 1:
            player_symbol = input_with_validation(
                f"Choose symbol ({symbols[0]} goes first).", symbols)
            player_order = symbols.index(player_symbol)
            player1, player2 = [human, ai_1][::(1 - player_order * 2)]
        else:
            if random.random() > 0.5:
                print("\n\t* O: ai_2, X: ai_1...")
                player1, player2 = ai_2, ai_1
            else:
                print("\n\t* O: ai_1, X: ai_2...")
                player1, player2 = ai_1, ai_2
    else:
        player1, player2 = human, human

    start_game(player1, player2, symbols=symbols)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    LEVEL_DESCS = ["stupid", "easy", "normal", "hard", "unfair"]
    LEVEL = 2
    print(f"\n[ You're playing with an AI at level {LEVEL} ({LEVEL_DESCS[LEVEL]}) ]\n")
    menu(n_player=1, ai_1=partial(ai_w_level, level=LEVEL), symbols=("O","X"))
