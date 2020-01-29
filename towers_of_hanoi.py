"""Towers of Hanoi"""

from search_solver import SearchSolver


class TowersOfHanoi(SearchSolver):
    """Class to solve Towers of Hanoi"""

    def __init__(self, n_discs, n_poles):
        discs = list(range(n_discs))
        poles = list(range(n_poles))
        super().__init__(discs, poles)

    def valid_state(self, state):
        return True

    def valid_transition(self, state1, state2):
        """
        Rules
        * Only one disc can move per transition
        * The moving disc must be the smallest from the origin pole (i.e. top most disc)
        * The moving disc must be the smallest in the destination pole
        """
        assert set(state1.keys()) == set(
            state2.keys()
        ), "state1 and state2 must have the same agents"

        moves = [
            (agent, state1[agent], state2[agent])
            for agent in state1.keys()
            if state1[agent] != state2[agent]
        ]

        if len(moves) == 1:
            disc_m, pole_src, pole_dest = moves[0]
            discs_in_src = []

            discs_in_src = [disc for disc, pole in state1.items() if pole == pole_src]
            discs_in_dest = [
                disc for disc, pole in state2.items() if (pole == pole_dest)
            ]

            if discs_in_dest:
                return disc_m == min(discs_in_src) == min(discs_in_dest)
            else:
                return True

        else:
            return False


def main():
    """Solves puzzle, given number of discs and number of poles

    RETURN VALUE
    tuple: path length, and possible states per step
    """
    n_discs = 2
    n_poles = 3

    solver = TowersOfHanoi(n_discs, n_poles)

    state_labels = {
        "A": {0: 1, 1: 1},
        "B": {0: 2, 1: 2},
        "C": {0: 0, 1: 1},
        "D": {0: 2, 1: 1},
        "E": {0: 1, 1: 2},
        "F": {0: 0, 1: 2},
        "top": {0: 0, 1: 0},
        "left": {0: 1, 1: 0},
        "right": {0: 2, 1: 0},
    }

    solver.rename_states(state_labels)
    return solver.trans_dict


if __name__ == "__main__":
    print(main())
