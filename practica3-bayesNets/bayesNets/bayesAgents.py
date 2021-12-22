# bayesAgents.py
# --------------
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


import bayesNet as bn
import game
from game import Actions, Agent, Directions
import inference
import layout
import factorOperations
import itertools
import operator as op
import random
import functools
import util

from hunters import GHOST_COLLISION_REWARD, WON_GAME_REWARD
from layout import PROB_BOTH_TOP, PROB_BOTH_BOTTOM, PROB_LEFT_TOP, PROB_ONLY_LEFT_TOP, \
    PROB_ONLY_LEFT_BOTTOM, PROB_FOOD_RED, PROB_GHOST_RED

X_POS_VAR = "xPos"
FOOD_LEFT_VAL = "foodLeft"
GHOST_LEFT_VAL = "ghostLeft"
X_POS_VALS = [FOOD_LEFT_VAL, GHOST_LEFT_VAL]

Y_POS_VAR = "yPos"
BOTH_TOP_VAL = "bothTop"
BOTH_BOTTOM_VAL = "bothBottom"
LEFT_TOP_VAL = "leftTop"
LEFT_BOTTOM_VAL = "leftBottom"
Y_POS_VALS = [BOTH_TOP_VAL, BOTH_BOTTOM_VAL, LEFT_TOP_VAL, LEFT_BOTTOM_VAL]

FOOD_HOUSE_VAR = "foodHouse"
GHOST_HOUSE_VAR = "ghostHouse"
HOUSE_VARS = [FOOD_HOUSE_VAR, GHOST_HOUSE_VAR]

TOP_LEFT_VAL = "topLeft"
TOP_RIGHT_VAL = "topRight"
BOTTOM_LEFT_VAL = "bottomLeft"
BOTTOM_RIGHT_VAL = "bottomRight"
HOUSE_VALS = [TOP_LEFT_VAL, TOP_RIGHT_VAL, BOTTOM_LEFT_VAL, BOTTOM_RIGHT_VAL]

OBS_VAR_TEMPLATE = "obs(%d,%d)"

BLUE_OBS_VAL = "blue"
RED_OBS_VAL = "red"
NO_OBS_VAL = "none"
OBS_VALS = [BLUE_OBS_VAL, RED_OBS_VAL, NO_OBS_VAL]

ENTER_LEFT = 0
ENTER_RIGHT = 1
EXPLORE = 2

def constructBayesNet(gameState):
    """
    Question 1: Bayes net structure

    Construct an empty Bayes net according to the structure given in the project
    description.

    There are 5 kinds of variables in this Bayes net:
    - a single "x position" variable (controlling the x pos of the houses)
    - a single "y position" variable (controlling the y pos of the houses)
    - a single "food house" variable (containing the house centers)
    - a single "ghost house" variable (containing the house centers)
    - a large number of "observation" variables for each cell Pacman can measure

    You *must* name all position and house variables using the constants
    (X_POS_VAR, FOOD_HOUSE_VAR, etc.) at the top of this file. 

    The full set of observation variables can be obtained as follows:

        for housePos in gameState.getPossibleHouses():
            for obsPos in gameState.getHouseWalls(housePos)
                obsVar = OBS_VAR_TEMPLATE % obsPos

    In this method, you should:
    - populate `obsVars` using the procedure above
    - populate `edges` with every edge in the Bayes Net (a tuple `(from, to)`)
    - set each `variableDomainsDict[var] = values`, where `values` is the set
      of possible assignments to `var`. These should again be set using the
      constants defined at the top of this file.
    """
    
    # Cómo crear BayesNet
    ##  1) Obtener todas las variables (obsVars, X, Y, Ghost House, Food House)
    ##  2) Definir los edges (X --> Ghost House, X --> Food House, ...etc)
    ##  3) Definir el dominio o los valores posibles para cada variable (X = x1,x2) o ('D' = ['wet','dry'])
    ##  4) Agrupar todas las variables y crear la red bayesiana.

    # Para todas las posiciones de una casa, se obtienen las observaciones de cada pared
    obsVars = []
    for housePos in gameState.getPossibleHouses():
        for obsPos in gameState.getHouseWalls(housePos):
            obsVar = OBS_VAR_TEMPLATE % obsPos      # obs(1,2) = obs(?,?) % (1,2)
            obsVars.append(obsVar)

    # Definición de los bordes
    edges = []
    for house in HOUSE_VARS:
        edges.append((X_POS_VAR,house))             # X --> Ghost/Food House
        edges.append((Y_POS_VAR,house))             # Y --> Ghost/Food House
        for obs in obsVars:
            edges.append((house,obs))               # Ghost/Food House --> Cada observación

    # Definición del dominio
    variableDomainsDict = {}
    
    variableDomainsDict[X_POS_VAR] = X_POS_VALS
    variableDomainsDict[Y_POS_VAR] = Y_POS_VALS
    variableDomainsDict[FOOD_HOUSE_VAR] = HOUSE_VALS
    variableDomainsDict[GHOST_HOUSE_VAR] = HOUSE_VALS
    
    for key in obsVars:
        variableDomainsDict[key] = OBS_VALS

    # Agrupación de todas las variables de la BayesNet
    variables = [X_POS_VAR, Y_POS_VAR] + HOUSE_VARS + obsVars

    # Creación de la BayesNet
    net = bn.constructEmptyBayesNet(variables, edges, variableDomainsDict)
    return net, obsVars

def fillCPTs(bayesNet, gameState):
    fillXCPT(bayesNet, gameState)
    fillYCPT(bayesNet, gameState)
    fillHouseCPT(bayesNet, gameState)
    fillObsCPT(bayesNet, gameState)

def fillXCPT(bayesNet, gameState):
    from layout import PROB_FOOD_LEFT 
    xFactor = bn.Factor([X_POS_VAR], [], bayesNet.variableDomainsDict())
    xFactor.setProbability({X_POS_VAR: FOOD_LEFT_VAL}, PROB_FOOD_LEFT)
    xFactor.setProbability({X_POS_VAR: GHOST_LEFT_VAL}, 1 - PROB_FOOD_LEFT)
    bayesNet.setCPT(X_POS_VAR, xFactor)

def fillYCPT(bayesNet, gameState):
    """
    Question 2: Bayes net probabilities

    Fill the CPT that gives the prior probability over the y position variable.
    See the definition of `fillXCPT` above for an example of how to do this.
    You can use the PROB_* constants imported from layout rather than writing
    probabilities down by hand.
    """

    # Creación de las CPTs, en este caso para la variable Y
    ##  1) Factor, contiene variables incondicionadas, condicionadas y dominio
    ##  2) Asignación de probabilidades, para cada assignment, o {VARIABLE : valor concreto de la variable}
    ##  3) Asignar CPT a la red bayesiana
     
    yFactor = bn.Factor([Y_POS_VAR], [], bayesNet.variableDomainsDict())
    yFactor.setProbability({Y_POS_VAR: BOTH_TOP_VAL}, PROB_BOTH_TOP)
    yFactor.setProbability({Y_POS_VAR: BOTH_BOTTOM_VAL}, PROB_BOTH_BOTTOM)
    yFactor.setProbability({Y_POS_VAR: LEFT_TOP_VAL}, PROB_ONLY_LEFT_TOP)
    yFactor.setProbability({Y_POS_VAR: LEFT_BOTTOM_VAL}, PROB_ONLY_LEFT_BOTTOM)
    bayesNet.setCPT(Y_POS_VAR, yFactor)

def fillHouseCPT(bayesNet, gameState):
    foodHouseFactor = bn.Factor([FOOD_HOUSE_VAR], [X_POS_VAR, Y_POS_VAR], bayesNet.variableDomainsDict())
    for assignment in foodHouseFactor.getAllPossibleAssignmentDicts():
        left = assignment[X_POS_VAR] == FOOD_LEFT_VAL
        top = assignment[Y_POS_VAR] == BOTH_TOP_VAL or \
                (left and assignment[Y_POS_VAR] == LEFT_TOP_VAL)

        if top and left and assignment[FOOD_HOUSE_VAR] == TOP_LEFT_VAL or \
                top and not left and assignment[FOOD_HOUSE_VAR] == TOP_RIGHT_VAL or \
                not top and left and assignment[FOOD_HOUSE_VAR] == BOTTOM_LEFT_VAL or \
                not top and not left and assignment[FOOD_HOUSE_VAR] == BOTTOM_RIGHT_VAL:
            prob = 1
        else:
            prob = 0

        foodHouseFactor.setProbability(assignment, prob)
    bayesNet.setCPT(FOOD_HOUSE_VAR, foodHouseFactor)

    ghostHouseFactor = bn.Factor([GHOST_HOUSE_VAR], [X_POS_VAR, Y_POS_VAR], bayesNet.variableDomainsDict())
    for assignment in ghostHouseFactor.getAllPossibleAssignmentDicts():
        left = assignment[X_POS_VAR] == GHOST_LEFT_VAL
        top = assignment[Y_POS_VAR] == BOTH_TOP_VAL or \
                (left and assignment[Y_POS_VAR] == LEFT_TOP_VAL)

        if top and left and assignment[GHOST_HOUSE_VAR] == TOP_LEFT_VAL or \
                top and not left and assignment[GHOST_HOUSE_VAR] == TOP_RIGHT_VAL or \
                not top and left and assignment[GHOST_HOUSE_VAR] == BOTTOM_LEFT_VAL or \
                not top and not left and assignment[GHOST_HOUSE_VAR] == BOTTOM_RIGHT_VAL:
            prob = 1
        else:
            prob = 0

        ghostHouseFactor.setProbability(assignment, prob)
    bayesNet.setCPT(GHOST_HOUSE_VAR, ghostHouseFactor)

def fillObsCPT(bayesNet, gameState):
    """
    This funcion fills the CPT that gives the probability of an observation in each square,
    given the locations of the food and ghost houses.

    This function creates a new factor for *each* of 4*7 = 28 observation
    variables. Don't forget to call bayesNet.setCPT for each factor you create.

    The XXXPos variables at the beginning of this method contain the (x, y)
    coordinates of each possible house location.

    IMPORTANT:
    Because of the particular choice of probabilities higher up in the Bayes
    net, it will never be the case that the ghost house and the food house are
    in the same place. However, the CPT for observations must still include a
    vaild probability distribution for this case. To conform with the
    autograder, this function uses the *food house distribution* over colors when both the food
    house and ghost house are assigned to the same cell.
    """

    bottomLeftPos, topLeftPos, bottomRightPos, topRightPos = gameState.getPossibleHouses()

    #convert coordinates to values (strings)
    coordToString = {
        bottomLeftPos: BOTTOM_LEFT_VAL,
        topLeftPos: TOP_LEFT_VAL,
        bottomRightPos: BOTTOM_RIGHT_VAL,
        topRightPos: TOP_RIGHT_VAL
    }

    for housePos in gameState.getPossibleHouses():
        for obsPos in gameState.getHouseWalls(housePos):

            obsVar = OBS_VAR_TEMPLATE % obsPos
            newObsFactor = bn.Factor([obsVar], [GHOST_HOUSE_VAR, FOOD_HOUSE_VAR], bayesNet.variableDomainsDict())
            assignments = newObsFactor.getAllPossibleAssignmentDicts()

            for assignment in assignments:
                houseVal = coordToString[housePos]
                ghostHouseVal = assignment[GHOST_HOUSE_VAR]
                foodHouseVal = assignment[FOOD_HOUSE_VAR]

                if houseVal != ghostHouseVal and houseVal != foodHouseVal:
                    newObsFactor.setProbability({
                        obsVar: RED_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, 0)
                    newObsFactor.setProbability({
                        obsVar: BLUE_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, 0)
                    newObsFactor.setProbability({
                        obsVar: NO_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, 1)
                else:
                    if houseVal == ghostHouseVal and houseVal == foodHouseVal:
                        prob_red = PROB_FOOD_RED
                    elif houseVal == ghostHouseVal:
                        prob_red = PROB_GHOST_RED
                    elif houseVal == foodHouseVal:
                        prob_red = PROB_FOOD_RED

                    prob_blue = 1 - prob_red

                    newObsFactor.setProbability({
                        obsVar: RED_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, prob_red)
                    newObsFactor.setProbability({
                        obsVar: BLUE_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, prob_blue)
                    newObsFactor.setProbability({
                        obsVar: NO_OBS_VAL,
                        GHOST_HOUSE_VAR: ghostHouseVal,
                        FOOD_HOUSE_VAR: foodHouseVal}, 0)

            bayesNet.setCPT(obsVar, newObsFactor)

def getMostLikelyFoodHousePosition(evidence, bayesNet, eliminationOrder):
    """
    Question 7: Marginal inference for pacman

    Find the most probable position for the food house.
    First, call the variable elimination method you just implemented to obtain
    p(FoodHouse | everything else). Then, inspect the resulting probability
    distribution to find the most probable location of the food house. Return
    this.

    (This should be a very short method.)
    """
    
    # Queremos consultar variable food house (posicion más probable)
    query_variables = [FOOD_HOUSE_VAR]

    # Aplicamos inferencia
    food_house_factor = inference.inferenceByVariableElimination(bayesNet, query_variables, evidence, eliminationOrder)

    # Iteramos sobre la distribución y nos quedamos el assignment de mayor probabilidad
    max_prob = 0
    for assignment in food_house_factor.getAllPossibleAssignmentDicts():
        current_prob = food_house_factor.getProbability(assignment)
        if current_prob > max_prob:
            max_prob = current_prob
            result_assignment = assignment
    
    return result_assignment


class BayesAgent(game.Agent):

    def registerInitialState(self, gameState):
        self.bayesNet, self.obsVars = constructBayesNet(gameState)
        fillCPTs(self.bayesNet, gameState)

        self.distances = cacheDistances(gameState)
        self.visited = set()
        self.steps = 0

    def getAction(self, gameState):
        self.visited.add(gameState.getPacmanPosition())
        self.steps += 1

        if self.steps < 40:
            return self.getRandomAction(gameState)
        else:
            return self.goToBest(gameState)

    def getRandomAction(self, gameState):
        legal = list(gameState.getLegalActions())
        legal.remove(Directions.STOP)
        random.shuffle(legal)
        successors = [gameState.generatePacmanSuccessor(a).getPacmanPosition() for a in legal]
        ls = [(a, s) for a, s in zip(legal, successors) if s not in gameState.getPossibleHouses()]
        ls.sort(key=lambda p: p[1] in self.visited)
        return ls[0][0]

    def getEvidence(self, gameState):
        evidence = {}
        for ePos, eColor in gameState.getEvidence().items():
            obsVar = OBS_VAR_TEMPLATE % ePos
            obsVal = {
                "B": BLUE_OBS_VAL,
                "R": RED_OBS_VAL,
                " ": NO_OBS_VAL
            }[eColor]
            evidence[obsVar] = obsVal
        return evidence

    def goToBest(self, gameState):
        evidence = self.getEvidence(gameState)
        unknownVars = [o for o in self.obsVars if o not in evidence]
        eliminationOrder = unknownVars + [X_POS_VAR, Y_POS_VAR, GHOST_HOUSE_VAR]
        bestFoodAssignment = getMostLikelyFoodHousePosition(evidence, 
                self.bayesNet, eliminationOrder)

        tx, ty = dict(
            zip([BOTTOM_LEFT_VAL, TOP_LEFT_VAL, BOTTOM_RIGHT_VAL, TOP_RIGHT_VAL],
                gameState.getPossibleHouses()))[bestFoodAssignment[FOOD_HOUSE_VAR]]
        bestAction = None
        bestDist = float("inf")
        for action in gameState.getLegalActions():
            succ = gameState.generatePacmanSuccessor(action)
            nextPos = succ.getPacmanPosition()
            dist = self.distances[nextPos, (tx, ty)]
            if dist < bestDist:
                bestDist = dist
                bestAction = action
        return bestAction

class VPIAgent(BayesAgent):

    def __init__(self):
        BayesAgent.__init__(self)
        self.behavior = None
        NORTH = Directions.NORTH
        SOUTH = Directions.SOUTH
        EAST = Directions.EAST
        WEST = Directions.WEST
        self.exploreActionsRemaining = \
                list(reversed([NORTH, NORTH, NORTH, NORTH, EAST, EAST, EAST,
                    EAST, SOUTH, SOUTH, SOUTH, SOUTH, WEST, WEST, WEST, WEST]))

    def reveal(self, gameState):
        bottomLeftPos, topLeftPos, bottomRightPos, topRightPos = \
                gameState.getPossibleHouses()
        for housePos in [bottomLeftPos, topLeftPos, bottomRightPos]:
            for ox, oy in gameState.getHouseWalls(housePos):
                gameState.data.observedPositions[ox][oy] = True

    def computeEnterValues(self, evidence, eliminationOrder):
        """
        Question 8a: Value of perfect information

        Given the evidence, compute the value of entering the left and right
        houses immediately. You can do this by obtaining the joint distribution
        over the food and ghost house positions using your inference procedure.
        The reward associated with entering each house is given in the *_REWARD
        variables at the top of the file.

        *Do not* take into account the "time elapsed" cost of traveling to each
        of the houses---this is calculated elsewhere in the code.
        """
        
        # Valor de la recompensa/penalización
        leftExpectedValue = 0
        rightExpectedValue = 0

        # Inferencia
        query = [FOOD_HOUSE_VAR, GHOST_HOUSE_VAR]
        house_factor = inference.inferenceByVariableElimination(self.bayesNet, query, evidence, eliminationOrder)

        # Probabilidad de las casas
        right_p = 0     # Probabilidad de que la casa de la comida este en TOP_RIGHT_VAL
        left_p = 0      # Probabilidad de que la casa de la comida este en TOP_LEFT_VAL
        for assignment in house_factor.getAllPossibleAssignmentDicts():
            if (assignment[FOOD_HOUSE_VAR] == TOP_LEFT_VAL) and (assignment[GHOST_HOUSE_VAR] == TOP_RIGHT_VAL):
                left_p += house_factor.getProbability(assignment)
            elif (assignment[FOOD_HOUSE_VAR] == TOP_RIGHT_VAL) and (assignment[GHOST_HOUSE_VAR] == TOP_LEFT_VAL):
                right_p += house_factor.getProbability(assignment)

        # Asignación de recompensa/penalización en función de las probabilidades obtenidas
        rightExpectedValue = right_p*WON_GAME_REWARD + left_p*GHOST_COLLISION_REWARD
        leftExpectedValue = left_p*WON_GAME_REWARD + right_p*GHOST_COLLISION_REWARD

        return leftExpectedValue, rightExpectedValue

    def getExplorationProbsAndOutcomes(self, evidence):
        unknownVars = [o for o in self.obsVars if o not in evidence]
        assert len(unknownVars) == 7
        assert len(set(evidence.keys()) & set(unknownVars)) == 0
        firstUnk = unknownVars[0]
        restUnk = unknownVars[1:]

        unknownVars = [o for o in self.obsVars if o not in evidence]
        eliminationOrder = unknownVars + [X_POS_VAR, Y_POS_VAR]
        houseMarginals = inference.inferenceByVariableElimination(self.bayesNet,
                [FOOD_HOUSE_VAR, GHOST_HOUSE_VAR], evidence, eliminationOrder)

        probs = [0 for i in range(8)]
        outcomes = []
        for nRed in range(8):
            outcomeVals = [RED_OBS_VAL] * nRed + [BLUE_OBS_VAL] * (7 - nRed)
            outcomeEvidence = dict(zip(unknownVars, outcomeVals))
            outcomeEvidence.update(evidence)
            outcomes.append(outcomeEvidence)

        for foodHouseVal, ghostHouseVal in [(TOP_LEFT_VAL, TOP_RIGHT_VAL),
                (TOP_RIGHT_VAL, TOP_LEFT_VAL)]:

            condEvidence = dict(evidence)
            condEvidence.update({FOOD_HOUSE_VAR: foodHouseVal, 
                GHOST_HOUSE_VAR: ghostHouseVal})
            assignmentProb = houseMarginals.getProbability(condEvidence)

            oneObsMarginal = inference.inferenceByVariableElimination(self.bayesNet,
                    [firstUnk], condEvidence, restUnk + [X_POS_VAR, Y_POS_VAR])

            assignment = oneObsMarginal.getAllPossibleAssignmentDicts()[0]
            assignment[firstUnk] = RED_OBS_VAL
            redProb = oneObsMarginal.getProbability(assignment)

            for nRed in range(8):
                outcomeProb = combinations(7, nRed) * \
                        redProb ** nRed * (1 - redProb) ** (7 - nRed)
                outcomeProb *= assignmentProb
                probs[nRed] += outcomeProb

        return list(zip(probs, outcomes))

    def computeExploreValue(self, evidence, enterEliminationOrder):
        """
        Question 8b: Value of perfect information

        Compute the expected value of first exploring the remaining unseen
        house, and then entering the house with highest expected value.

        The method `getExplorationProbsAndOutcomes` returns pairs of the form
        (prob, explorationEvidence), where `evidence` is a new evidence
        dictionary with all of the missing observations filled in, and `prob` is
        the probability of that set of observations occurring.

        You can use getExplorationProbsAndOutcomes to
        determine the expected value of acting with this extra evidence.
        """

        # Valor de exploración
        expectedValue = 0

        # Aplicación de la fórmula
        for p, expl in self.getExplorationProbsAndOutcomes(evidence):
            expectedValue += p * max(self.computeEnterValues(expl, enterEliminationOrder))

        return expectedValue

    def getAction(self, gameState):

        if self.behavior == None:
            self.reveal(gameState)
            evidence = self.getEvidence(gameState)
            unknownVars = [o for o in self.obsVars if o not in evidence]
            enterEliminationOrder = unknownVars + [X_POS_VAR, Y_POS_VAR]
            exploreEliminationOrder = [X_POS_VAR, Y_POS_VAR]

            print(evidence)
            print(enterEliminationOrder)
            print(exploreEliminationOrder)
            enterLeftValue, enterRightValue = \
                    self.computeEnterValues(evidence, enterEliminationOrder)
            exploreValue = self.computeExploreValue(evidence,
                    exploreEliminationOrder)

            # TODO double-check
            enterLeftValue -= 4
            enterRightValue -= 4
            exploreValue -= 20

            bestValue = max(enterLeftValue, enterRightValue, exploreValue)
            if bestValue == enterLeftValue:
                self.behavior = ENTER_LEFT
            elif bestValue == enterRightValue:
                self.behavior = ENTER_RIGHT
            else:
                self.behavior = EXPLORE

            # pause 1 turn to reveal the visible parts of the map
            return Directions.STOP

        if self.behavior == ENTER_LEFT:
            return self.enterAction(gameState, left=True)
        elif self.behavior == ENTER_RIGHT:
            return self.enterAction(gameState, left=False)
        else:
            return self.exploreAction(gameState)

    def enterAction(self, gameState, left=True):
        bottomLeftPos, topLeftPos, bottomRightPos, topRightPos = \
                gameState.getPossibleHouses()

        dest = topLeftPos if left else topRightPos

        actions = gameState.getLegalActions()
        neighbors = [gameState.generatePacmanSuccessor(a) for a in actions]
        neighborStates = [s.getPacmanPosition() for s in neighbors]
        best = min(zip(actions, neighborStates), 
                key=lambda x: self.distances[x[1], dest])
        return best[0]

    def exploreAction(self, gameState):
        if self.exploreActionsRemaining:
            return self.exploreActionsRemaining.pop()

        evidence = self.getEvidence(gameState)
        enterLeftValue, enterRightValue = self.computeEnterValues(evidence,
                [X_POS_VAR, Y_POS_VAR])

        if enterLeftValue > enterRightValue:
            self.behavior = ENTER_LEFT
            return self.enterAction(gameState, left=True)
        else:
            self.behavior = ENTER_RIGHT
            return self.enterAction(gameState, left=False)

def cacheDistances(state):
    width, height = state.data.layout.width, state.data.layout.height
    states = [(x, y) for x in range(width) for y in range(height)]
    walls = state.getWalls().asList() + state.data.layout.redWalls.asList() + state.data.layout.blueWalls.asList()
    states = [s for s in states if s not in walls]
    distances = {}
    for i in states:
        for j in states:
            if i == j:
                distances[i, j] = 0
            elif util.manhattanDistance(i, j) == 1:
                distances[i, j] = 1
            else:
                distances[i, j] = 999999
    for k in states:
        for i in states:
            for j in states:
                if distances[i,j] > distances[i,k] + distances[k,j]:
                    distances[i,j] = distances[i,k] + distances[k,j]

    return distances

# http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def combinations(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n-r, -1))
    denom = functools.reduce(op.mul, range(1, r+1))
    return numer / denom

'''
Modifications
'''
DIFFICULTY_VAR = "difficulty"
EASY = "easy" #d0
HARD = "hard" #d1
DIFFICULTY_VALS = [EASY, HARD]

INTELLIGENCE_VAR = "intelligence"
NOT_SMART = "not_smart" #i0
SMART = "smart" #i1
INTELLIGENCE_VALS = [NOT_SMART, SMART]

GRADE_VAR = "grade"
EXCELLENT = "excellent" #g1
GOOD = "good" #g2
AVERAGE = "average" #g3
GRADE_VALS = [EXCELLENT, GOOD, AVERAGE]

SAT_VAR = "sat"
LOW_SCORE = "low_score" #s0
HIGH_SCORE = "high_score" #s1
SAT_VALS = [LOW_SCORE, HIGH_SCORE]

LETTER_VAR = "letter"
WEAK_RECOMM_LETTER = "weak_recomm_letter" #l0
STRONG_RECOMM_LETTER = "strong_recomm_letter" #l1
LETTER_VALS = [WEAK_RECOMM_LETTER, STRONG_RECOMM_LETTER]

def constructStudentBayesNet():
    #pass
    "*** MODIFICATION ***"
    # Definición de nodos
    vars = [DIFFICULTY_VAR, INTELLIGENCE_VAR, GRADE_VAR, SAT_VAR, LETTER_VAR]
    
    # Definición de los bordes
    edges = [(DIFFICULTY_VAR, GRADE_VAR), (INTELLIGENCE_VAR, GRADE_VAR), (INTELLIGENCE_VAR, SAT_VAR), (GRADE_VAR, LETTER_VAR)]

    # Definición del dominio
    variableDomainsDict = {}
    
    variableDomainsDict[DIFFICULTY_VAR] = DIFFICULTY_VALS
    variableDomainsDict[INTELLIGENCE_VAR] = INTELLIGENCE_VALS
    variableDomainsDict[GRADE_VAR] = GRADE_VALS
    variableDomainsDict[SAT_VAR] = SAT_VALS
    variableDomainsDict[LETTER_VAR] = LETTER_VALS

    # Creación de la BayesNet
    net = bn.constructEmptyBayesNet(vars, edges, variableDomainsDict)

    # CPTs
    ## Difficulty
    difficultyFactor = bn.Factor([DIFFICULTY_VAR], [], net.variableDomainsDict())
    difficultyFactor.setProbability({DIFFICULTY_VAR: EASY}, 0.6)
    difficultyFactor.setProbability({DIFFICULTY_VAR: HARD}, 0.4)
    net.setCPT(DIFFICULTY_VAR, difficultyFactor)

    ## Intelligence
    intelligenceFactor = bn.Factor([INTELLIGENCE_VAR], [], net.variableDomainsDict())
    intelligenceFactor.setProbability({INTELLIGENCE_VAR: NOT_SMART}, 0.7)
    intelligenceFactor.setProbability({INTELLIGENCE_VAR: SMART}, 0.3)
    net.setCPT(INTELLIGENCE_VAR, intelligenceFactor)

    ## SAT
    satFactor = bn.Factor([SAT_VAR], [INTELLIGENCE_VAR], net.variableDomainsDict())
    satFactor.setProbability({SAT_VAR: LOW_SCORE, INTELLIGENCE_VAR: NOT_SMART}, 0.95)
    satFactor.setProbability({SAT_VAR: LOW_SCORE, INTELLIGENCE_VAR: SMART}, 0.2)
    satFactor.setProbability({SAT_VAR: HIGH_SCORE, INTELLIGENCE_VAR: NOT_SMART}, 0.05)
    satFactor.setProbability({SAT_VAR: HIGH_SCORE, INTELLIGENCE_VAR: SMART}, 0.8)
    net.setCPT(SAT_VAR, satFactor)

    ## Grade
    gradeFactor = bn.Factor([GRADE_VAR], [INTELLIGENCE_VAR, DIFFICULTY_VAR], net.variableDomainsDict())
    gradeFactor.setProbability({GRADE_VAR: EXCELLENT, INTELLIGENCE_VAR: NOT_SMART,  DIFFICULTY_VAR: EASY}, 0.3)
    gradeFactor.setProbability({GRADE_VAR: EXCELLENT, INTELLIGENCE_VAR: NOT_SMART,  DIFFICULTY_VAR: HARD}, 0.05)
    gradeFactor.setProbability({GRADE_VAR: EXCELLENT, INTELLIGENCE_VAR: SMART,  DIFFICULTY_VAR: EASY}, 0.9)
    gradeFactor.setProbability({GRADE_VAR: EXCELLENT, INTELLIGENCE_VAR: SMART,  DIFFICULTY_VAR: HARD}, 0.5)

    gradeFactor.setProbability({GRADE_VAR: GOOD, INTELLIGENCE_VAR: NOT_SMART,  DIFFICULTY_VAR: EASY}, 0.4)
    gradeFactor.setProbability({GRADE_VAR: GOOD, INTELLIGENCE_VAR: NOT_SMART,  DIFFICULTY_VAR: HARD}, 0.25)
    gradeFactor.setProbability({GRADE_VAR: GOOD, INTELLIGENCE_VAR: SMART,  DIFFICULTY_VAR: EASY}, 0.08)
    gradeFactor.setProbability({GRADE_VAR: GOOD, INTELLIGENCE_VAR: SMART,  DIFFICULTY_VAR: HARD}, 0.3)

    gradeFactor.setProbability({GRADE_VAR: AVERAGE, INTELLIGENCE_VAR: NOT_SMART,  DIFFICULTY_VAR: EASY}, 0.3)
    gradeFactor.setProbability({GRADE_VAR: AVERAGE, INTELLIGENCE_VAR: NOT_SMART,  DIFFICULTY_VAR: HARD}, 0.7)
    gradeFactor.setProbability({GRADE_VAR: AVERAGE, INTELLIGENCE_VAR: SMART,  DIFFICULTY_VAR: EASY}, 0.02)
    gradeFactor.setProbability({GRADE_VAR: AVERAGE, INTELLIGENCE_VAR: SMART,  DIFFICULTY_VAR: HARD}, 0.2)
    net.setCPT(GRADE_VAR, gradeFactor)

    ## Letter
    letterFactor = bn.Factor([LETTER_VAR], [GRADE_VAR], net.variableDomainsDict())
    letterFactor.setProbability({LETTER_VAR: WEAK_RECOMM_LETTER, GRADE_VAR: EXCELLENT}, 0.1)
    letterFactor.setProbability({LETTER_VAR: WEAK_RECOMM_LETTER, GRADE_VAR: GOOD}, 0.4)
    letterFactor.setProbability({LETTER_VAR: WEAK_RECOMM_LETTER, GRADE_VAR: AVERAGE}, 0.99)

    letterFactor.setProbability({LETTER_VAR: STRONG_RECOMM_LETTER, GRADE_VAR: EXCELLENT}, 0.9)
    letterFactor.setProbability({LETTER_VAR: STRONG_RECOMM_LETTER, GRADE_VAR: GOOD}, 0.6)
    letterFactor.setProbability({LETTER_VAR: STRONG_RECOMM_LETTER, GRADE_VAR: AVERAGE}, 0.01)
    net.setCPT(LETTER_VAR, letterFactor)

    return net

def inferenceOnStudentBayesNet(bayes_net):
    "*** MODIFICATION ***" 
    query = [LETTER_VAR]
    evidence = {INTELLIGENCE_VAR: NOT_SMART, DIFFICULTY_VAR: EASY}

    return inference.inferenceByEnumeration(bayes_net, query, evidence)

bayes_net = constructStudentBayesNet()
print(inferenceOnStudentBayesNet(bayes_net))