"""Search Solver Abstract Base Class module"""

from abc import ABC, abstractclassmethod
from functools import reduce
import itertools as it
import operator

import pandas as pd


class SearchSolver(ABC):
    """Abstract Base Class for Solving "Search" Problems

    Search Problems are defined as problems that have "states" and "transitions".
    The solution would be to find a path(s) between two states

    LIMITATIONS:
    The following assumptions are made for transitions:
    * bidirectional
    * uniform cost
    """

    def __init__(self, agents, agent_states):
        self.agents = agents
        self.agent_states = agent_states

        all_df = self.gen_all_states()
        self.valid_df = all_df[all_df.apply(self.valid_state, axis=1)]
        self.gen_transition_dict()

    @abstractclassmethod
    def valid_state(cls, state):
        """Determine if state is valid; returns `True` or `False`"""
        pass

    @abstractclassmethod
    def valid_transition(cls, state1, state2):
        """Determine if transition is valid; returns `True` or `False`"""
        pass

    def gen_all_states(self):
        """Generates all states given the agents and possible agent states

        NOTES
        This function will generate a pd.DataFrame with dimensions:
            `(len(agent_states) ** len(agents) x len(agents))`
            with values being a subset of `agent_states`

        It is assumed all agents can take on all agent_states.
        Invalid states should be filetered out using another funciton.

        PARAMETERS
        agents (collection): collection of agents
        agent_states (collection): collection containing possible agent states

        RETURN VALUE
        pd.DataFrame: all possible states
        """
        all_states = []
        for _ in self.agents:
            if all_states:
                all_states = [
                    state + [agent_state]
                    for state in all_states
                    for agent_state in self.agent_states
                ]
            else:
                all_states = [[agent_state]
                              for agent_state in self.agent_states]

        return pd.DataFrame(all_states, columns=self.agents)

    def gen_transition_dict(self):
        """Transition dictionary generator

        Notes:
        * This default implementation is terribly inefficient at O(n^2)
        * Consider trade-off between brute force vs. searching reachable states
        """
        all_trans = {
            frozenset([s1_id, s2_id]): self.valid_transition(
                self.valid_df.loc[s1_id, :], self.valid_df.loc[s2_id, :]
            )
            for s1_id, s2_id in it.combinations(self.valid_df.index, 2)
        }

        valid_trans = set([pair for pair, check in all_trans.items() if check])
        trans_dict = {}
        for sid_1, sid_2 in valid_trans:
            trans_dict[sid_1] = trans_dict.get(sid_1, set()) | {sid_2}
            trans_dict[sid_2] = trans_dict.get(sid_2, set()) | {sid_1}

        self.trans_dict = trans_dict
        return self.trans_dict

    def find_shortest_paths(self, state_id1, state_id2, max_depth=1000):
        """Finds shortest path(s) from state1 to state2, given state id's

        PARAMETERS
        state_id1 (hashable obj): id of state1
        state_id2 (hashable obj): id of state2
        max_depth (int): optional; maximum search depth
        """
        transition_dict = self.trans_dict

        steps = [{state_id1}]
        for step_ct in range(max_depth):
            current_step = steps[-1]
            reachable_states = reduce(
                operator.or_, [transition_dict[state_id]
                               for state_id in current_step]
            )

            if state_id2 in reachable_states:
                break
            else:
                steps.append(reachable_states)
        else:
            print(f"Could not find a valid path after {step_ct} steps")
            return None

        print(f"found a valid path(s) within {step_ct + 1} steps")

        # find path
        path = [{state_id2}]
        for current_step in steps[::-1]:
            path.append(
                current_step
                & reduce(
                    operator.or_, [transition_dict[state_id]
                                   for state_id in path[-1]]
                )
            )

        return path[::-1]

    def get_state_id(self, state):
        """Returns state_id given agents as parameters and agent states as values"""

        state_ids = self.valid_df[
            reduce(
                operator.and_, [self.valid_df[col] ==
                                val for col, val in state.items()]
            )
        ].index

        assert len(state_ids) > 0, "parameters do not describe a state_id"
        assert len(state_ids) == 1, "parameters do not describe a unique state_id"
        return state_ids.values[0]

    def rename_states(self, rename_dict):
        """Assigns a name to a provided state

        PARAMETERS
        rename_dict (dict): key = new name, value = state
        """

        rename_df = pd.DataFrame(rename_dict).T
        rename_df.index.name = "__new_name__"
        agents = rename_df.columns.tolist()
        rename_df.reset_index(inplace=True)

        merged = self.valid_df.merge(rename_df, on=agents, how="inner")
        self.valid_df.rename(
            index=merged["__new_name__"].to_dict(), inplace=True)
        self.gen_transition_dict()
