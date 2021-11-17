# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game


pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.

    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')

    s1 = A | B
    s2 = ~ A % (~ B | C)
    s3 = logic.disjoin(~ A, ~ B, C)

    return logic.conjoin(s1, s2, s3)


def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.

    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')

    s1 = C % (B | D)
    s2 = A >> (~ B & ~ D)
    s3 = ~(B & ~ C) >> A
    s4 = ~ D >> C

    return logic.conjoin(s1, s2, s3, s4)

def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    WA1 = logic.PropSymbolExpr('WumpusAlive',1)
    WA0 = logic.PropSymbolExpr('WumpusAlive',0)
    WB0 = logic.PropSymbolExpr('WumpusBorn',0)
    WK0 = logic.PropSymbolExpr('WumpusKilled',0)

    s1 = WA1 % ((WA0 & ~ WK0) | (~ WA0 & WB0))
    s2 = ~ (WA0 & WB0)
    s3 = WB0

    return logic.conjoin(s1, s2, s3)

def modelToString(model):
    """Converts the model to a string for printing purposes. The keys of a model are
    sorted before converting the model to a string.

    model: Either a boolean False or a dictionary of Expr symbols (keys)
    and a corresponding assignment of True or False (values). This model is the output of
    a call to logic.pycoSAT.
    """
    if model == False:
        return "False"
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)

def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"
    return logic.pycoSAT(logic.to_cnf(sentence))

def atLeastOne(literals):
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    return logic.associate('|',literals)



def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in
    CNF (conjunctive normal form) that represents the logic that at most one of
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    sentences = []
    for i in range(0, len(literals) - 1):
        for j in range(i + 1, len(literals)):
            sentences.append((~ literals[i] | ~ literals[j]))

    return logic.associate('&', sentences)


def exactlyOne(literals):
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in
    CNF (conjunctive normal form)that represents the logic that exactly one of
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return (atLeastOne(literals) & atMostOne(literals))


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    # Action keys extraction
    plan = []
    priority_q = util.PriorityQueue()
    for expr_key in model:
        # Only true expr
        if model[expr_key]:
            id = logic.parseExpr(expr_key)[0]
            # Only true actions expr
            if id in actions:
                priority = int(logic.parseExpr(expr_key)[1])
                priority_q.push(id, priority)
                
    while not priority_q.isEmpty():
        plan.append(priority_q.pop())

    return plan

def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE H[ERE ***"
    current_P = logic.PropSymbolExpr(pacman_str, x, y, t)
    prev_Ps = []

    if not walls_grid[x][y - 1]:    # South
        prev_Ps.append(logic.PropSymbolExpr(pacman_str, x, y - 1, t - 1) & logic.PropSymbolExpr('North', t - 1))
    if not walls_grid[x + 1][y]:    # East
        prev_Ps.append(logic.PropSymbolExpr(pacman_str, x + 1, y, t - 1) & logic.PropSymbolExpr('West', t - 1))
    if not walls_grid[x][y + 1]:    # North
        prev_Ps.append(logic.PropSymbolExpr(pacman_str, x, y + 1, t - 1) & logic.PropSymbolExpr('South', t - 1))
    if not walls_grid[x - 1][y]:    # West
        prev_Ps.append(logic.PropSymbolExpr(pacman_str, x - 1, y, t - 1) & logic.PropSymbolExpr('East', t - 1))

    state_axiom = current_P % logic.disjoin(prev_Ps)
    return state_axiom

def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    x0, y0 = problem.getStartState()
    xg, yg = problem.getGoalState()

    "*** YOUR CODE HERE ***"
    T_LIMIT = 50
    t = 0
    possible_actions = ['North', 'East', 'South', 'West']

    # Initial pose
    poses = []
    for x in range(1, width + 1):
        for y in range(1, height  + 1):
            if (x0, y0) == (x, y):
                poses.append(logic.PropSymbolExpr(pacman_str, x, y, 0))
            else:
                poses.append(~logic.PropSymbolExpr(pacman_str, x, y, 0))
    initial_pose = logic.associate('&', poses)

    one_action_list = [] # Storage one-action sentences to conjoin later
    for t in range(1, T_LIMIT + 1):
        # Goal pose
        poses = []
        for x in range(1, width + 1):
            for y in range(1, height  + 1):
                if (xg, yg) == (x, y):
                    poses.append(logic.PropSymbolExpr(pacman_str, x, y, t))
                else:
                    poses.append(~logic.PropSymbolExpr(pacman_str, x, y, t))
        goal_pose = logic.associate('&', poses)

        # General
        if t > 0:
            state_successors_list = []
            actions_list = []

            # Successors for each t between 1 to t for every position
            for tt in range (1, t + 1):
                for x in range(1, width + 1):
                    for y in range(1, height  + 1):
                        if (x,y) not in walls_list:
                            state_successors_list.append(pacmanSuccessorStateAxioms(x, y, tt, walls))

            print(state_successors_list)
 
            # One action for t-1            
            for action in possible_actions:
                actions_list.append(logic.PropSymbolExpr(action, t - 1))
            one_action_list.append(exactlyOne(actions_list))

            # Sentences
            ## One action for every t
            one_action = logic.associate('&', one_action_list)   
            ## All posible successors for t
            state_successors = logic.associate('&',state_successors_list)

            # Model
            mdl = findModel(logic.conjoin(initial_pose, goal_pose, one_action, state_successors))
        else:
            # If t = 0
            ## Pacman don't move, because it's initial pose is it's goal pose
            mdl = findModel(logic.conjoin(initial_pose, goal_pose))
        
        # Sequence extraction
        if mdl:
            action_seq = extractActionSequence(mdl, possible_actions)
            # print(action_seq) # Debug print
            return action_seq
    return None


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    (x0, y0), food_locations = problem.getStartState()
    food_list = food_locations.asList()
    walls_list = walls.asList()


    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
#fglp = foodGhostLogicPlan

# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

