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
        # Write value iteration code here
        self.runValueIteration()

    def runValueIteration(self):
        values = util.Counter()

        for i in range(self.iterations):
            values = self.values.copy()

            for state in self.mdp.getStates():
                possibleVals = []

                if self.mdp.isTerminal(state):
                    self.values[state] = 0
   
                else:
                    for action in self.mdp.getPossibleActions(state):
                        tempValue = 0
                        
                        for t in self.mdp.getTransitionStatesAndProbs(state, action):
                            tempValue += t[1]*(self.mdp.getReward(state, action, t[0]) + self.discount * values[t[0]])
                        possibleVals.append(tempValue)

                    self.values[state] = max(possibleVals)


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
        value = 0
        for t in self.mdp.getTransitionStatesAndProbs(state, action):
            value += t[1] * (self.mdp.getReward(state, action, t[0]) + self.discount *self.values[t[0]])
        return value
        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        else:
            value = -float("inf")
            bestAction = None

            for action in self.mdp.getPossibleActions(state):
                tempVal = 0

                for t in self.mdp.getTransitionStatesAndProbs(state, action):
                    tempVal += t[1] * (self.mdp.getReward(state, action, t[0]) + self.discount * self.values[t[0]])
                if tempVal > value:
                    value = tempVal
                    bestAction = action
            return bestAction



    def getPolicy(self, state):
        value = 0
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
        values = util.Counter()
        states = self.mdp.getStates()

        for i in range(self.iterations):
            currentState = states[i%len(states)]
            values = self.values.copy()
            possibleVals = []

            if self.mdp.isTerminal(currentState):
                self.values[currentState] = 0
   
            else:
                for action in self.mdp.getPossibleActions(currentState):
                    tempValue = 0
                    
                    for t in self.mdp.getTransitionStatesAndProbs(currentState, action):
                        tempValue += t[1]*(self.mdp.getReward(currentState, action, t[0]) + self.discount * values[t[0]])
                    possibleVals.append(tempValue)

                self.values[currentState] = max(possibleVals)

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
        "Algorithm:"
        values = util.Counter()
        predSets =[]
        #initialize an empty priority queue
        queue = util.PriorityQueue()


        #For each state s do
        for state in self.mdp.getStates():
            
            #find predecessors of state
            predecessors = set()
            for s in self.mdp.getStates():
                for action in self.mdp.getPossibleActions(s):
                    for t in self.mdp.getTransitionStatesAndProbs(s, action):
                        if t[0] == state:
                            predecessors.add(s)
            predSets.append(predecessors)
            #if terminal state do the 0 thing
            if self.mdp.isTerminal(state):
                self.values[state] = 0

            else:
                #Get highest possible Q value from state
                maxQVal = -float("inf")
                for action in self.mdp.getPossibleActions(state):
                    maxQVal = max(self.getQValue(state, action), maxQVal)

                #Find the absolute difference between current s and q val
                diff = self.getValue(state) - maxQVal

                #push s into the priority queue qith priority -diff
                queue.push(state, -diff)
 
        for i in range(self.iterations):
            #if the priority queue is empty, then terminate
            if queue.isEmpty() == False:
                #Pop a state off the priority queue
                state = queue.pop()
                
                #If not terminal, update state's value in self.values
                if self.mdp.isTerminal(state) == False:
                    self.values[state] - self.getValue(state)
                #for each predecessor pred of state:
                for p in predSets[i]:
                    #Get highest possible Q value from state
                    maxQVal = -float("inf")
                    for action in self.mdp.getPossibleActions(state):
                        maxQVal = max(self.getQValue(state, action), maxQVal)
                    diff = self.getValue(state) - maxQVal

                    if diff > self.theta:
                        queue.push(p, -diff)

