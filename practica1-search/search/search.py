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
    frontier = util.Stack()                       # LIFO
    frontier.push((problem.getStartState(),0,[])) # ((x,y), acc_cost, path_to_xy)
    expanded = []

    MAX_EXPANDED = 300
    n_expanded_nodes = 0

    while frontier is not frontier.isEmpty():   
        current_node = frontier.pop()  

        if problem.isGoalState(current_node[0]):
            return current_node[2]

        if current_node[0] not in expanded:
            expanded.append(current_node[0])   
            
            for child in problem.expand(current_node[0]):
                n_expanded_nodes += 1
                cost = current_node[1] + child[2]   # acc_cost
                current_path = []

                # Init current path
                for item in current_node[2]:
                    current_path.append(item)  
                current_path.append(child[1]) 

                """
                Modificación 1.

                En laberintos de mayor dimensión, se requieren más expansiones,
                por tanto si MAX_EXPANDED no es suficiente, nunca acaba el laberinto.

                A veces retrocede porque, el último movimiento se corresponde con la
                última expansión hecha, que depende del todas las acciones posibles.
                """
                if n_expanded_nodes >= MAX_EXPANDED:
                    return current_path

                # Add expanded child action to current path and all data to frontier          
                new_frontier = (child[0],cost,current_path)
                frontier.push(new_frontier)

    return -1


def breadthFirstSearch(problem):
    frontier = util.Queue()                         # FIFO
    frontier.push((problem.getStartState(),0,[]))   # ((x,y), acc_cost, path_to_xy)
    expanded = []

    MAX_EXPANDED = 300
    n_expanded_nodes = 0


    while frontier is not frontier.isEmpty():   
        current_node = frontier.pop()  
        
        if problem.isGoalState(current_node[0]):
            return current_node[2]

        if current_node[0] not in expanded:
            expanded.append(current_node[0])   
            
            for child in problem.expand(current_node[0]):
                n_expanded_nodes += 1
                cost = current_node[1] + child[2]   # acc_cost
                current_path = []

                # Init current path
                for item in current_node[2]:
                    current_path.append(item)  
                current_path.append(child[1]) 

                """
                Modificación 1.

                En laberintos de mayor dimensión, se requieren más expansiones,
                por tanto si MAX_EXPANDED no es suficiente, nunca acaba el laberinto.

                A veces retrocede porque, el último movimiento se corresponde con la
                última expansión hecha, que depende del todas las acciones posibles.
                """
                if n_expanded_nodes >= MAX_EXPANDED:
                    return current_path

                # Add expanded child action to current path and all data to frontier                          
                new_frontier = (child[0],cost,current_path)
                frontier.push(new_frontier)                

    return -1


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    frontier = util.PriorityQueue() # FIFO with priority (heuristic)

    # Initial prioriy = heuristic(init) --> ((x,y), acc_cost, path_to_xy)
    frontier.push((problem.getStartState(),0,[]), heuristic(problem.getStartState(), problem)) 
    expanded = []

    while frontier is not frontier.isEmpty():   
        current_node = frontier.pop()
        # print(current_node) # para DEBUG

        if problem.isGoalState(current_node[0]):
            return current_node[2]

        if current_node[0] not in expanded:
            expanded.append(current_node[0])   
            
            for child in problem.expand(current_node[0]):
                cost = current_node[1] + child[2]   # acc_cost
                current_path = []

                # Init current path
                for item in current_node[2]:
                    current_path.append(item)  

                # Add expanded child action to current path and all data to frontier
                current_path.append(child[1])           
                new_frontier = (child[0],cost,current_path)

                # priority = heuristic(current_xy) + acc_cost
                frontier.push(new_frontier, heuristic(child[0], problem) + cost)

    return -1


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
