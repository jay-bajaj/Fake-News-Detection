import random

# DO NOT import any other modules.
# DO NOT change the prototypes of any of the functions.
# Sample test cases given
# Grading will be based on hidden tests


# Cost function to be optimised
# Takes a list of elements
# Return the total sum of squares of even-indexed elements and inverse squares of odd-indexed elements
def cost_function(X): # 0.25 Marks
    odd=0
    even=0
    for i in range(0,len(X)):
        if i%2==0:
            even+=X[i]*X[i]
        if i%2 ==1:
            odd+=1/(X[i]*X[i])
    cost=even+odd
    return cost       
    


    # Takes length of vector as input
    # Returns 4 values - initial_position, initial_velocity, best_position and best_cost
    # Initialises position to a list with random values between [-10, 10] and velocity to a list with random values between [-1, 1]
    # best_position is an empty list and best cost is set to -1 initially
def initialise(length): # 0.25 Marks
    # your code goes here
    best_cost=-1
    best_position=[]
    initial_position=[]
    initial_velocity=[]
    for i in range(length):
        initial_position.append(random.uniform(-10,10))
        initial_velocity.append(random.uniform(-1,1))
    position=initial_position
    velocity=initial_velocity
    # best_position[:]=position
    return initial_position, initial_velocity, best_position, best_cost


# Evaluates the position vector based on the input func
# On getting a better cost, best_position is updated in-place
# Returns the better cost 
def assess(position, best_position, best_cost, func): # 0.25 Marks
    if func(position) < best_cost or best_cost==-1:
        best_position[:]=position
        best_cost=func(position)
    return best_cost


# Updates velocity in-place by the given formula for each element:
# vel = w*vel + c1*r1*(best_position-position) + c2*r2*(best_group_position-position)
# where r1 and r2 are random numbers between 0 and 1 (not same for each element of the list)
# No return value
def velocity_update(w, c1, c2, velocity, position, best_position, best_group_position): # 0.5 Marks
    # print(len(best_position))
    # print(len(best_group_position))
    for i in range(len(velocity)):
        r1=random.random()
        r2=random.random()
        velocity[i] = w*velocity[i] + c1*r1*(best_position[i]-position[i]) + c2*r2*(best_group_position[i]-position[i])


# Input - position, velocity, limits(list of two elements - [min, max])
# Updates position in-place by the given formula for each element:
# pos = pos + vel
# Position element set to limit if it crosses either limit value
# No return value
def position_update(position, velocity, limits): # 0.5 Marks
    # print(len(velocity))
    sum=[a+b for a,b in zip(position,velocity)]
    position[:]=sum
    # C = list(map(lambda x: (float(5)/9)*(x-32), F))
    position[:]= list(map(lambda x: limits[1] if x>limits[1] else (limits[0] if x<limits[0] else x ) , position))
    # for i in range(len(position)):
    #     if i>limits[1]:





# swarm is a list of particles each of which is a list containing current_position, current_velocity, best_position and best_cost
# Initialise these using the function written above
# In every iteration for every swarm particle, evaluate the current position using the assess function (use the cost function you have defined) and update the particle's best cost if needed
# Update the best group cost and best group position based on performance of that particle
# Then for every swarm particle, first update its velocity then its position
# Return the best position and cost for the group
def optimise(vector_length, swarm_size, w, c1, c2, limits, max_iterations, initial_best_group_position=[], initial_best_group_cost=-1): # 1.25 Marks
    best_group_position=[]
    best_group_cost=initial_best_group_cost
    current_position=[]
    current_velocity=[]
    best_position=[]
    best_cost=[]
    for j in range(swarm_size):
        position, velocity, bestposition,bestcost=initialise(vector_length)
        current_position.append(position)
        current_velocity.append(velocity)
        best_position.append(bestposition)
        best_cost.append(bestcost)
        # best_group_position=bestposition
    for i in range(max_iterations):
        for j in range(swarm_size):
            best_cost[j]=assess(current_position[j], best_position[j], best_cost[j], cost_function)
            if best_cost[j]<best_group_cost or best_group_cost==-1:
                best_group_cost=best_cost[j]
                best_group_position=best_position[j]
        for j in range(swarm_size):
            velocity_update(w,c1,c2,current_velocity[j],current_position[j],best_position[j],best_group_position)
            position_update(current_position[j],current_velocity[j],limits)
    # print(best_group_position)
    return best_group_position , best_group_cost    
            
