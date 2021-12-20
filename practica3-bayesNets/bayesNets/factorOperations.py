# factorOperations.py
# -------------------
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


from bayesNet import Factor
import operator as op
import util
import functools

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors, joinVariable):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


def joinFactors(factors):
    """
    Question 3: Your join implementation 

    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))

    factors_list = list(factors)                        # Lista de factores para iterar y operar mejor

    # Componentes para crear un factor
    uncond_set = set()
    cond_set = set()
    domain = factors_list[0].variableDomainsDict()      # El dominio es el mismo

    # Obtención de los sets definidos
    for factor in factors_list:
        uncond_set = factor.unconditionedVariables().union(uncond_set)
        cond_set = factor.conditionedVariables().union(cond_set)

    # Si la variable se encuentra en ambos sets --> se elimina del set condicionado.
    for item in uncond_set.intersection(cond_set):
        cond_set.remove(item)

    # Factor a retornar
    result_factor = Factor(uncond_set, cond_set, domain)

    # Asignación de probabilidades
    assignment_list = result_factor.getAllPossibleAssignmentDicts()     # Cada entrada de la CPT del factor final
    for assignment in assignment_list:
        current_p = 1
        for factor in factors_list:            
            current_p *= factor.getProbability(assignment)              # P(D,W) = P(D|W) * P(W) --> factores simples admiten assignments complejos
        result_factor.setProbability(assignment, current_p)

    return result_factor    



def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor, eliminationVariable):
        """
        Question 4: Your eliminate implementation 

        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        # Componentes para crear un factor
        uncond_set = factor.unconditionedVariables()
        cond_set = factor.conditionedVariables()
        domain = factor.variableDomainsDict()

        # Operaciones sobre los sets, en este caso la eliminación de una variable incondicionada
        if eliminationVariable in uncond_set:
            uncond_set.remove(eliminationVariable)

        # Creación del factor a retornar
        result_factor = Factor(uncond_set, cond_set, domain)
        eliminated_factor = Factor(eliminationVariable, set(), domain)  # Para poder generar assignments válidos para la tabla del factor sin eliminar

        # Asignación de probabilidades (se da la vuelta)
        for result_assignment in result_factor.getAllPossibleAssignmentDicts():
            current_p = 0            
            for elim_assignment in eliminated_factor.getAllPossibleAssignmentDicts():
                current_assignment = {**result_assignment, **elim_assignment}   # Entrada a la CPT sin eliminar
                current_p += factor.getProbability(current_assignment)                       
            result_factor.setProbability(result_assignment, current_p)

        return result_factor 

    return eliminate

eliminate = eliminateWithCallTracking()


def normalize(factor):
    """
    Question 5: Your normalize implementation 

    Input factor is a single factor.

    The set of conditioned variables for the normalized factor consists 
    of the input factor's conditioned variables as well as any of the 
    input factor's unconditioned variables with exactly one entry in their 
    domain.  Since there is only one entry in that variable's domain, we 
    can either assume it was assigned as evidence to have only one variable 
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables 
    with single value domains, but that is alright since we have to assign 
    variables that only have one value in their domain to that single value.

    Return a new factor where the sum of the all the probabilities in the table is 1.
    This should be a new factor, not a modification of this factor in place.

    If the sum of probabilities in the input factor is 0,
    you should return None.

    This is intended to be used at the end of a probabilistic inference query.
    Because of this, all variables that have more than one element in their 
    domain are assumed to be unconditioned.
    There are more general implementations of normalize, but we will only 
    implement this version.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    variableDomainsDict = factor.variableDomainsDict()
    for conditionedVariable in factor.conditionedVariables():
        if len(variableDomainsDict[conditionedVariable]) > 1:
            print("Factor failed normalize typecheck: ", factor)
            raise ValueError("The factor to be normalized must have only one " + \
                            "assignment of the \n" + "conditional variables, " + \
                            "so that total probability will sum to 1\n" + 
                            str(factor))

    # Variable que acumula la suma de probabilidades sin normalizar
    total_prob = 0

    # Componentes para crear un factor
    uncond_set = factor.unconditionedVariables()
    cond_set = factor.conditionedVariables() 

    for assignment in factor.getAllPossibleAssignmentDicts():
        total_prob += factor.getProbability(assignment) 

    # Cálculo del factor alpha de normalización
    try:
        alpha = 1/total_prob
    except ZeroDivisionError:
        return None    

    # CASO ESPECÍFICO: Si el dominio de una variable incondicionada tiene EXACTAMENTE UNA entrada.
    for var in factor.unconditionedVariables():
        if len(variableDomainsDict[var]) == 1:
            uncond_set.remove(var)
            cond_set.add(var)

    # Creación del factor a retornar
    result_factor = Factor(uncond_set, cond_set, variableDomainsDict)

    # Asignación de probabilidades
    for assignment in factor.getAllPossibleAssignmentDicts():
        new_prob = alpha*factor.getProbability(assignment)      # normalized_P = alpha*unormalized_P
        result_factor.setProbability(assignment, new_prob) 

    return result_factor
