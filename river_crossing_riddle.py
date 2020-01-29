"""River Crossing Riddle"""

from search_solver import SearchSolver


class RiverCrossingRiddle(SearchSolver):
    """Class to solve River Crossing riddle"""

    def __init__(self, boat_capacity):
        agents = ["robot", "fox", "chicken", "chicken-feed"]
        agent_states = [0, 1]
        self.capacity = boat_capacity
        super().__init__(agents, agent_states)

    def valid_state(self, state):
        """Checks if a `state` is valid

        Rules:
            The following cannot have the same state (i.e. together),
                unless 'robot' has the same state (i.e. accompanying them):
            * 'fox' and 'chicken'
            * 'chicken' and 'chicken-feed'

        PARAMETERS
        state (dict-like)

        RETURN VALUE
        bool
        """
        return ~(
            ((state["fox"] == state["chicken"]) & (state["robot"] != state["chicken"]))
            | (
                (state["chicken"] == state["chicken-feed"])
                & (state["robot"] != state["chicken"])
            )
        )

    def valid_transition(self, state1, state2):
        """Check if `state1` can transition to `state2`

        Rules:
        * The robot must always change state (i.e. row the boat)
        * Total state changes cannot be more than `capacity` (i.e. boat passengers)
        * Change state must be the same direction as robot
            (i.e. if robot goes 1 -> 0, all changes must be 1 -> 0; vice versa)

        PARAMETERS:
        state1 (dict-like)
        state2 (dict-like)
        capacity (int): max # of passengers in the boat, including the robot

        RETURN VALUE
        bool
        """
        assert set(state1.keys()) == set(
            state2.keys()
        ), "state1 and state2 must have the same agents"

        return (
            (state1["robot"] != state2["robot"])
            & (
                sum(
                    int(state1[agent] != state2[agent])
                    for agent, state in state1.items()
                    if state == state1["robot"]
                )
                <= self.capacity
            )
            & (
                sum(
                    int(state1[agent] != state2[agent])
                    for agent, state in state1.items()
                    if state != state1["robot"]
                )
                == 0
            )
        )


def main(boat_capacity):
    """Solves puzzle, given boat capacity

    PARAMETERS
    boat_capacity (int): number of passengers in the boat, including the robot

    RETURN VALUE
    tuple: path length, and possible states per step
    """
    solver = RiverCrossingRiddle(boat_capacity)
    state_i = solver.get_state_id({agent: 0 for agent in solver.agents})
    state_n = solver.get_state_id({agent: 1 for agent in solver.agents})
    paths = solver.find_shortest_paths(state_i, state_n)
    path_length = len(paths) - 1
    return (path_length, paths)


if __name__ == "__main__":
    main(2)
