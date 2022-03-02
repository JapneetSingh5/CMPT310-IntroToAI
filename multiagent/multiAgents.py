# multiAgents.py
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print(newGhostStates)
        
        # if PacMan has won on the successor state being considered, return the highest possible score
        if successorGameState.isWin():
            return 100000000;
        # if PacMan has lost on the successor state being considered, return the lowest possible score
        if successorGameState.isLose():
            return -100000000;

        compoundScore = 0;

        nearestFood = 10000000000;
        newFood = newFood.asList()
        for food in newFood:
            nearestFood = min(nearestFood, manhattanDistance(food, newPos));
        nearestFoodPoints = 1/(nearestFood + 1);
        compoundScore = compoundScore + 2*nearestFoodPoints;

        try:
            capsulePosition = currentGameState.getCapsules()[0]
            compoundScore = compoundScore + 5/(manhattanDistance(capsulePosition, newPos)+1)
        except:
            compoundScore = compoundScore + 0

        return successorGameState.getScore() + compoundScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentNum):
        Returns a list of legal actions for an agent
        agentNum=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentNum, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # building a depth-first-search-like minimax search tree

        # instead of having a min-value function and a max-value function 
        # like in the pseudo code for the lecture, we build a general purpose function here
        # and assign min-max values conditionally by checking the agent number

        def expandMiniMaxTree(startDepth,currentGameState,agentNum):
            
            # if we reach the assigned self.depth, return the score
            if startDepth == self.depth:
                return None,self.evaluationFunction(currentGameState)
            
            if agentNum == currentGameState.getNumAgents():
                startDepth+=1
                agentNum=0
                if startDepth == self.depth:
                    return None,self.evaluationFunction(currentGameState)
            
            currentScore = None
            action = None
            
            allowedActions = currentGameState.getLegalActions(agentNum)        
            for currAction in allowedActions:
                [nextAgentAction,nextAgentScore] = expandMiniMaxTree(startDepth,currentGameState.generateSuccessor(agentNum,currAction),agentNum+1)
                # if agentNum == 0, i.e. it is the pacMan, use max-value condition
                if agentNum==0:
                    if currentScore == None or currentScore < nextAgentScore:
                        currentScore = nextAgentScore
                        action = currAction
                # if agentNum > 0, i.e. it is the pacMan, use max-value condition
                else:
                    if currentScore == None or currentScore > nextAgentScore:
                        currentScore = nextAgentScore
                        action = currAction
            
            if currentScore == None:
                currentScore = self.evaluationFunction(currentGameState)
            
            return [action, currentScore]

        return expandMiniMaxTree(0,gameState,0)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaPruneTree(startDepth,currentGameState,agentNum, alpha, beta):
            
            # if we reach the assigned self.depth, return the score
            if startDepth == self.depth:
                return None,self.evaluationFunction(currentGameState)
            
            if agentNum == currentGameState.getNumAgents():
                startDepth+=1
                agentNum=0
                if startDepth == self.depth:
                    return None,self.evaluationFunction(currentGameState)
            
            currentScore = None
            action = None
            
            allowedActions = currentGameState.getLegalActions(agentNum)        
            for currAction in allowedActions:
                [nextAgentAction,nextAgentScore] = alphaBetaPruneTree(startDepth,currentGameState.generateSuccessor(agentNum,currAction),agentNum+1,resolveAlphaBeta(agentNum, alpha,beta, currentScore)[0], resolveAlphaBeta(agentNum, alpha,beta, currentScore)[1])
                # if agentNum == 0, i.e. it is the pacMan, use max-value condition
                if agentNum==0:
                    if currentScore == None or currentScore < nextAgentScore:
                        currentScore = nextAgentScore
                        action = currAction
                    if currentScore != None and beta != None and currentScore > beta:
                        break

                # if agentNum > 0, i.e. it is the pacMan, use max-value condition
                else:
                    if currentScore == None or currentScore > nextAgentScore:
                        currentScore = nextAgentScore
                        action = currAction
                    if currentScore != None and alpha != None and currentScore < alpha:
                        break
            
            if currentScore == None:
                currentScore = self.evaluationFunction(currentGameState)
            
            return [action, currentScore]

        
        def resolveAlphaBeta(agentNum, alpha, beta, currentScore):
            if(agentNum==0):
                if alpha == None:
                    return [currentScore, beta]
                elif currentScore == None:
                    return [alpha, beta]
                elif alpha != None and currentScore != None:
                    return [max(alpha, currentScore), beta]
                else:
                    return [None, beta]
            else:
                if currentScore == None:
                    return [alpha, beta]
                elif beta == None:
                    return [alpha, currentScore]
                elif beta != None and currentScore != None:
                    return [alpha, min(beta, currentScore)]
                else:
                    return [alpha, None]

        return alphaBetaPruneTree(0, gameState, 0, None, None)[0]        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectiMax(startDepth,currentGameState,agentNum):
            
            # if we reach the assigned self.depth, return the score
            if startDepth == self.depth:
                return None,self.evaluationFunction(currentGameState)
            
            if agentNum == currentGameState.getNumAgents():
                startDepth+=1
                agentNum=0
                if startDepth == self.depth:
                    return None,self.evaluationFunction(currentGameState)
            
            currentScore = None
            action = None
            
            allowedActions = currentGameState.getLegalActions(agentNum)  
            allowedActions.sort()      
            for currAction in allowedActions:
                [nextAgentAction,nextAgentScore] = expectiMax(startDepth,currentGameState.generateSuccessor(agentNum,currAction),agentNum+1)
                # if agentNum == 0, i.e. it is the pacMan, use max-value condition
                if agentNum==0:
                    if currentScore == None or currentScore < nextAgentScore:
                        currentScore = nextAgentScore
                        action = currAction
                # if agentNum > 0, i.e. it is the pacMan, use max-value condition
                else:
                    if currentScore == None:
                        currentScore = nextAgentScore
                        action = currAction
                    else:
                        currentScore = currentScore + nextAgentScore
                        # add scores for all non-pacman agents
                        # action = currAction
            # take expected value of score for non-pacman agents
            if agentNum > 0 and currentScore != None:
                totalMoves = len(allowedActions)
                currentScore = currentScore/totalMoves
            if currentScore == None:
                currentScore = self.evaluationFunction(currentGameState)
            
            return [action, currentScore]

        return expectiMax(0,gameState,0)[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    1. food proximity given higher weight - calculate manhattan distance to nearest food pellet,
    invert it and multiply it by 2 and add it to the basic getScore functions
    2. if nearest ghost is scared for over 1.75x the distance to it, add inverse of its distance to the score
    (1.75 chosen by going over values from 1.0 to 2.0, 1.75 resulted in highest score in autograder)
    3. Closer the capsule, higher the score
    added 5 times the inverse of manhattan distance 
    """
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    # print(newGhostStates)
    
    # if PacMan has won on the successor state being considered, return the highest possible score
    if currentGameState.isWin():
        return 100000000;
    # if PacMan has lost on the successor state being considered, return the lowest possible score
    if currentGameState.isLose():
        return -100000000;

    compoundScore = 0;

    nearestFood = 10000000000;
    foodList = foodList.asList()
    for food in foodList:
        nearestFood = min(nearestFood, manhattanDistance(food,pos));
    nearestFoodPoints = 1/(nearestFood + 1);
    compoundScore = compoundScore + 2*nearestFoodPoints;

    try:
        capsulePosition = currentGameState.getCapsules()[0]
        compoundScore = compoundScore + 5/(manhattanDistance(capsulePosition, pos)+1)
    except:
        compoundScore = compoundScore + 0

    ghostPos = ghostStates[0].getPosition()
    nearestScaredTime = currentGameState.getGhostStates()[0].scaredTimer
    if nearestScaredTime > 0 and nearestScaredTime > 1.75 * manhattanDistance(pos, ghostPos) :
        compoundScore =  compoundScore + 5/(manhattanDistance(pos, ghostPos)+1);

    return currentGameState.getScore() + compoundScore

# Abbreviation
better = betterEvaluationFunction
