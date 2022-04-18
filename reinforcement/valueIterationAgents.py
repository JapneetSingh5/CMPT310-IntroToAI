# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(0, self.iterations):
            updatedValues = util.Counter();
            for stateIteration in self.mdp.getStates():
                # get the optimal action from current state
                optimalAct = self.computeActionFromValues(stateIteration);
                if optimalAct:
                    updatedQvalue = self.computeQValueFromValues(stateIteration, optimalAct);
                    updatedValues[stateIteration] = updatedQvalue
            for stateIteration2 in self.mdp.getStates():
                    self.values[stateIteration2]=updatedValues[stateIteration2];



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        QValue = 0;
        stateAndProb = self.mdp.getTransitionStatesAndProbs(state, action);
        for tempState, tempProb in stateAndProb:
            QValue = QValue + tempProb*( self.values[tempState]*self.discount + self.mdp.getReward(state, action, tempState) ) 
        return QValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actionList = self.mdp.getPossibleActions(state);
        if not actionList:
            return None;
        optimalAct = None;
        optimalActReward = -10000000;
        for action in actionList:
            currentActionReward = 0;
            prob = self.mdp.getTransitionStatesAndProbs(state, action);
            for tempState, tempProb in prob:
                currentActionReward = currentActionReward + tempProb * (self.getValue(tempState)*self.discount + self.mdp.getReward(state, action, tempState) ) 
            optimalActReward = max(optimalActReward, currentActionReward);
            if currentActionReward == optimalActReward:
                optimalAct = action
        return optimalAct 
        

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        currentState = -1;
        stateList = self.mdp.getStates();
        n = len(stateList);
        for _ in range(0, self.iterations):
            currentState = (currentState + 1)%n;
            optimalAct = self.computeActionFromValues(stateList[currentState]);
            if optimalAct:
                self.values[stateList[currentState]]=self.computeQValueFromValues(stateList[currentState], optimalAct);
                    

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        stateList = self.mdp.getStates();
        predecessors = dict();
        pq = util.PriorityQueue();
        # get predecessors 
        for state in stateList:
            actionList = self.mdp.getPossibleActions(state);
            for action in actionList:
                stateAndProb = self.mdp.getTransitionStatesAndProbs(state, action);
                for nextState, _ in stateAndProb:
                    if nextState in predecessors:
                        predecessors[nextState].add(state);
                    else:
                        predecessors[nextState] = set([state])
        # construct priority queue
        for state in stateList:
            actionList = self.mdp.getPossibleActions(state);
            if len(actionList)>0:
                diff = abs(self.computeQValueFromValues(state, self.computeActionFromValues(state)) - self.values[state])
                pq.update(state, -diff)  
        # iterate till pq is empty or number of iterations runs out    
        for _ in range(self.iterations):
            if not pq.isEmpty():
                currentState = pq.pop()
                if not self.mdp.isTerminal(currentState):
                    self.values[currentState] = self.computeQValueFromValues(currentState,  self.computeActionFromValues(currentState))
                for state in predecessors[currentState]:
                    if not self.mdp.isTerminal(state):
                        maximum = self.computeQValueFromValues(state, self.computeActionFromValues(state))
                        diff = abs(maximum - self.values[state])
                        if self.theta < diff:
                            pq.update(state, -diff)

