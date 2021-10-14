# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # python3 pacman.py -l tinyMaze -p SearchAgent
    # print(type(problem))
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Problem actions for initial state", problem.getActions(problem.getStartState()))

    # for action in problem.getActions(problem.getStartState()):
    #     print("If you go ", action)
    #     print("\tNew state --> ", problem.getNextState(problem.getStartState(), action))

    # print(problem.expand(problem.getStartState()))

    # for i in problem.expand(problem.getStartState()):
    #     print(problem.getActions(i[0]))
    
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    def convert_dir(dir):
        if dir == "North":
            return n
        elif dir == "South":
            return s
        elif dir == "East":
            return e
        elif dir == "West":
            return w

    def rev_dir(dir):
        if dir == "North":
            return "South"
        elif dir == "South":
            return "North"
        elif dir == "East":
            return "West"
        elif dir == "West":
            return "East"

    
    frontier = util.Stack()
    frontier.push((problem.getStartState(),None,None,0))
    expanded = []
    all_info = []

    while frontier is not frontier.isEmpty():
        current_node = frontier.pop()
        print("CURRENT_NODE:", current_node)
        if problem.isGoalState(current_node[0]):
            # print("Solution -->", current_node)
            all_info.append(current_node) 
            expanded.append(current_node[0])
            
            print("\n------------")
            print("All_info -->", all_info)
            print("Expanded -->", expanded)
            print("\n------------")

            path = []
            idx = -1
            state = all_info[idx][0]

            while state != problem.getStartState():              
                path_action = all_info[idx][1]
                action = all_info[idx][2]
                path.append(path_action)
                state = problem.getNextState(state, action)
                idx = expanded.index(state)

            path.reverse()
            return path


        if current_node not in expanded:
            expanded.append(current_node[0])   
            all_info.append(current_node)         
            for child in problem.expand(current_node[0]):
                if child[0] not in expanded:
                    #print("\tExpands", child)
                    frontier.push((child[0],child[1],rev_dir(child[1]),child[2]))

    return None
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
