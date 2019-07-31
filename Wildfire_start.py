# Forest fire simulation
# Author: Cameron Millspaugh, OSC, 2019
# Using translated Javascript Code from Aaron Weeden, Shodor, 2016-2018

import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import sys

#matplotlib.use('TkAgg') # Uncomment this line to plot chart in an external window
matplotlib.use('module://ipykernel.pylab.backend_inline') # Uncomment this to plot graph as frames in the console

# Global Constants - Conversions
acre_to_sq_ft = 43560 # 1 acre = 43560 sq ft
sq_mile_to_acre = 640 # 1 sq_mile = 640 acres

#-------- US Wildfire Data --------
# Global Constants - From Wildfire Data Files - Cal Fire Redbooks
CAL_AVG_NUM_WFIRES = 3111 # California 2007-2016 state average
CAL_AVG_NUM_ACRES_BURNED = 191060 # California 2007-2016 state average
CAL_AVG_DAMAGES_USD = 373928203 # California 2007-2016 state average 

# Global Constants - From Wildfire Data Files - SuppCosts
US_WFIRE_SUPP_COST_USD = 3143256000 # United States 2007-2016 average Wildfire supression costs
US_AVG_NUM_WFIRES = 70388 # United States 2007-2016 average
US_AVG_NUM_ACRES_BURNED = 6554647 # United States 2007-2016 average

#-------- SIMULATION CONSTANTS --------
# Global Constants - Simulation Parameters
SEED_START = False # Set fire starting positions; True/False
DISPLAY = False # Show matplotlib graphs of simulation
SHOW_STATS = True # Show numeric simulation results
FOREST_DENSITY = 80 # Percent chance to add a tree to the forest (0-100)

# Global Constants - Canvas
TREE_COUNT_X = 40 # Number of trees in x direction
TREE_COUNT_Y = 40 # Number of trees in y direction
TIME_INTERVAL = .05 # Length of time step interval (seconds)

# Global Constants - Fire
BURN_CHANCE = 0  # Burn chance (0-100)
NUM_FIRES = 4 # Number of starting fires

# Global Constants - Wind
TOWARDS_TOP = 0 # Wind blows from bottom to top on canvas
TOWARDS_RIGHT = 1 # Wind blows from left to right on canvas
TOWARDS_BOTTOM = 2 # Wind blows from top to bottom on canvas
TOWARDS_LEFT = 3 # Wind blows from right to left on canvas
WIND_DIRECTION = TOWARDS_BOTTOM # Direction of the wind
WIND_SPEED = 0 # Magnitude of wind speed (0,100)

# Global Constants - Tree Attributes
COLOR_GREEN = .2 # matplotlib "tab10" cmap, with vmin = 0, vmax = 1
COLOR_RED = .3 # matplotlib "tab10" cmap, with vmin = 0, vmax = 1
COLOR_GRAY = .7 # matplotlib "tab10" cmap, with vmin = 0, vmax = 1
COLOR_BROWN = .6 # matplotlib "tab10" cmap, with vmin = 0, vmax = 1
NON_BURNING_STATE = COLOR_GREEN # Tree tile color
BURNING_STATE = COLOR_RED # Tree tile color
BURNT_STATE = COLOR_GRAY # Tree tile color
FLOOR_STATE = COLOR_BROWN # Tree tile color


# Define Tree object
class Tree : 
    current_state = None
    next_state = None
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Declare global variables
tree_colors = np.zeros(shape=(TREE_COUNT_X, TREE_COUNT_Y)) # 2D array of RGB values
forest = np.empty(shape=(TREE_COUNT_X, TREE_COUNT_Y), dtype=Tree) # 2D array of tree objects
init_burn_trees = np.empty(shape=(1,NUM_FIRES), dtype=Tree) # x position of initially-burning trees
forest_burnt = np.ones(shape=(TREE_COUNT_X, TREE_COUNT_Y)) * COLOR_GRAY # 2D array of burnt trees



''' ------------ Input Parsing Functions ------------ '''
# Return True if a string x is a real number, false otherwise
def is_num(x):
    flag = False
    try:
        float(x)
        flag = True
    except:
        TypeError
    return flag
# end function is_num(x)


# Converts a string to an integer
def to_int(x):
    return int(float(x))
# end function to_int(x)



''' ------------ Tree Management Functions ------------ '''
# Define a function to define tree positions and states
def reset_forest() :
    # Force compiler to recognize global variables
    global forest, tree_colors

    # (Re)define an initially-empty 2D array to store tree objects
    forest = np.empty(shape=(TREE_COUNT_X, TREE_COUNT_Y), dtype=Tree)
    tree_colors = np.zeros(shape=(TREE_COUNT_X, TREE_COUNT_Y))
    
    # Loop over the forest in the x direction
    for x in range(TREE_COUNT_X) :

        # Loop over the forest in the y direction
        for y in range(TREE_COUNT_Y) :
        
            # Define a new tree object
            tree = Tree(x,y)
            
            # Add the new tree object to the array
            forest[x, y] = tree
            
            # Set the state of the tree
            set_forest_state(tree, tree_colors)
    
    set_init_burning()
# end function reset_forest()


# Define a function to initialize the forest that are set on fire
def set_init_burning() :
    global init_burn_trees, tree_colors
    # Initialize/clear burning tree array
    init_burn_trees = np.empty(shape=(1,NUM_FIRES), dtype=Tree)

    for h in range(NUM_FIRES) :
        # Set random fire starting locations
        x = np.random.randint(TREE_COUNT_X)
        y = np.random.randint(TREE_COUNT_Y)
        # Assign trees to burning tree array
        init_burn_trees[0,h] = forest[x,y]
        # Update state for assigned trees
        set_forest_state(forest[x,y], tree_colors)
# end function set_init_burning()


# Define a function to set the state of a tree
def set_forest_state(tree, tree_colors) :
    # If the tree is on the border,
    if (is_on_border(tree)) :
    
        # Set the tree's state to burnt
        tree.current_state = BURNT_STATE
        tree_colors[tree.x, tree.y] = BURNT_STATE
    
    # Otherwise, if the tree is initially burning
    elif (is_init_burning(tree)) :
    
        # Set the tree's state to burning
        tree.current_state = BURNING_STATE
        tree_colors[tree.x, tree.y] = BURNING_STATE

    
    # Otherwise, the tree is sparse (floor)
    elif not is_init_tree():
        tree.current_state = FLOOR_STATE
        tree_colors[tree.x, tree.y] = FLOOR_STATE

    # Otherwise, the tree element is a living tree
    else:

        # Indicate the current tree is non-burning
        tree.current_state = NON_BURNING_STATE
        tree_colors[tree.x, tree.y] = NON_BURNING_STATE

    """ Make a copy of the tree's state, because when we advance the
    model, we don't want to update a tree's state until its neighbors have had a chance
    to check it. """
    tree.next_state = tree.current_state
# end function set_forest_state(tree, tree_colors)


# Define a function to set the states of forest in the next time step
def set_next_states() :

    # Loop over all tree columns, excluding the borders
    for x in range(1, TREE_COUNT_X - 1) :

        # Loop over all tree rows, excluding the borders
        for y in range(1, TREE_COUNT_Y - 1) :
        
            # Create a variable to store the current tree
            tree = forest[x, y]
        
            # If the tree is currently burning
            if (tree.current_state == BURNING_STATE):
        
                # Set it to be burnt in the next time step
                tree.next_state = BURNT_STATE
        
            # Otherwise, if the tree is not currently burning
            else :
        
                # Keep the tree in its current state for the next time step
                tree.next_state = tree.current_state
        
            # If the tree will be non-burning in the next time step
            if (tree.next_state == NON_BURNING_STATE) :

                # Try to spread the fire to the tree from each of its 4 neighbors
                try_spread(tree)
# end function set_next_states()


# Define a function for trying to spread the fire to a tree from each of its 4 neighbors
def try_spread(tree) :

    # If the tree's neighbor is on fire and the burn chance happens,
    if (spread_from_top(tree) or spread_from_left(tree) or spread_from_right(tree) or spread_from_bottom(tree)):

        # Light the tree on fire in the next time step
        tree.next_state = BURNING_STATE
# end function try_spread(tree)



''' ------------ Fire spread Functions ------------ '''
# Define a function that returns whether a tree is initially burning
def is_init_burning(tree) :
    # Get a random x and a random y for the burning tree
    return True if tree in init_burn_trees else False
# end function is_init_burning(tree)


# Define a function to see if fire spreads from the tree at the given x,y position
def spread(x, y, effective_burn_chance) :
    return tree_is_on_fire(forest[x, y]) and burn_chance_happens(BURN_CHANCE + effective_burn_chance)
# end function spread(x, y, effective_burn_chance)


# Define a function that returns whether fire spreads to current tree from the top neighbor
def spread_from_top(tree) :
    return spread(tree.x - 1, tree.y, WIND_SPEED if WIND_DIRECTION == TOWARDS_BOTTOM else -WIND_SPEED)
# end function spread_from_top(tree)


# Define a function that returns whether fire spreads to current tree from the right neighbor
def spread_from_right(tree) :
    return spread(tree.x, tree.y + 1, WIND_SPEED if WIND_DIRECTION == TOWARDS_LEFT else -WIND_SPEED)
# end function spread_from_right(tree)


# Define a function that returns whether fire spreads to current tree from the bottom neighbor
def spread_from_bottom(tree) :
    return spread(tree.x + 1, tree.y, WIND_SPEED if WIND_DIRECTION == TOWARDS_TOP else -WIND_SPEED)
# end function spread_from_bottom(tree)


# Define a function that returns whether fire spreads to current tree from the left neighbor
def spread_from_left(tree) :
    return spread(tree.x, tree.y - 1, WIND_SPEED if WIND_DIRECTION == TOWARDS_RIGHT else -WIND_SPEED)
# end function spread_from_left(tree)



''' --- Status Check Functions --- '''
# Define a function that returns whether a tree is on the border of the forest
def is_on_border(tree) :
    return is_in_left_border_column(tree) or is_in_top_border_row(tree) or is_in_right_border_column(tree) or is_in_bottom_border_row(tree)
# end function is_in_left_border_column(tree)


# Define a function that returns wheter a tree is in the bottom border row
def is_in_bottom_border_row(tree) :
    return tree.y == (TREE_COUNT_Y - 1)
# end function is_in_bottom_border_row(tree)


# Define a function that returns whether a tree is in the left border column
def is_in_left_border_column(tree) :
    return tree.x == 0
# end function is_in_left_border_column(tree)


# Define a function that returns whether a tree is in the right border column
def is_in_right_border_column(tree) :
    return tree.x == (TREE_COUNT_X - 1)
# end function is_in_right_border_column(tree)


# Define a function that returns whether a tree is in the top border row
def is_in_top_border_row(tree) :
    return tree.y == 0
# end function is_in_top_border_row(tree)

    
# Checks if the given x,y coordinate is a valid position in the forest matrix
def is_valid_point(x, y) :
    if is_in_range(x, y) and not is_on_border(forest[x, y]) and not is_init_burning(forest[x, y]) :
        return True
    return False
# end function is_valid_point(x, y)


# Checks if the given x,y coordinate is within the range of the forest matrix
def is_in_range(x, y) :
    if x > 0 and y > 0 and x < TREE_COUNT_X and y < TREE_COUNT_Y :
        return True
    return False
# end function is_in_range(x, y)


# Define a function that returns whether a given tree is on fire
def tree_is_on_fire(tree) :
    return forest[tree.x, tree.y].current_state == BURNING_STATE
# end function tree_is_on_fire(tree)


# Define a function for checking if any forest are still on fire
def still_burning() :
    # Loop over the forest in the x direction
    for x in forest :
        # Loop over the forest in the y direction
        for y in x :
            # Return True if the current state is different from the last state
            if y.current_state != y.next_state :
                return True
    
    return False
# end function still_burning()



''' ------------ Display Functions ------------ '''
# Define a function that runs the simulation until no fire remains
def play() :
    global DISPLAY
    set_next_states()
    while (still_burning()) : 
        # advance the model by swapping in each tree's current state for it's next state
        advance()
    
        # Call the function to draw all forest on the canvas
        if DISPLAY :
            draw()

        # Set the states of forest in the next time step
        set_next_states()
    if DISPLAY:
        plt.pause(1)
        plt.close()
# end function play()


# Function advances the model by swapping tree's current state for it's next state
def advance() :
    global tree_colors
    for x in range(TREE_COUNT_X - 1) :
        for y in range(TREE_COUNT_Y - 1) :
            tree = forest[x, y]
            tree.current_state = tree.next_state
            tree_colors[x,y] = tree.current_state
# end function advance()


# Define a function to draw all forest on the canvas
def draw() :
    # draw all forest
    plt.minorticks_on()
    plt.grid(False, which="both", axis="both")
    plt.axis("on")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.imshow(tree_colors, cmap="tab10",vmin="0",vmax="1")
    plt.pause(TIME_INTERVAL)
    plt.show()
# end function draw()


# Define a function that provides user with simulation statistics
def print_stats(num_live_trees_before, num_live_trees_after) :
    forest_destruction = 1 - (num_live_trees_after/num_live_trees_before)
    print("-----------------------------")
    print("Before Simulation:")
    print("- living trees: ", num_live_trees_before)
    print("- burn chance: ", BURN_CHANCE)
    print("- fires: ", NUM_FIRES)
    print("- seeded: ", SEED_START)
    print("-----------------------------")
    print("After simulation:")
    print("- living trees: ", num_live_trees_after)
    print("- percent destruction: {:.2f}".format(100 * forest_destruction))
    print("-----------------------------")
# end function print_stats(num_live_trees_before, num_live_trees_after)


# Define a function to count the number of live trees
def count_live_trees() :
    live_count = 0
    for x in range(TREE_COUNT_X - 1) :
        for y in range(TREE_COUNT_Y - 1) :
            if forest[x,y].current_state == NON_BURNING_STATE :
                live_count = live_count + 1
    return live_count
# end function count_live_trees()


# Define a function to run and control the wildfire simulation
def run_simulation():
    reset_forest()
    num_live_trees_before = count_live_trees()
    if DISPLAY :
        draw()
    play()
    num_live_trees_after = count_live_trees()
    if SHOW_STATS:
        print_stats(num_live_trees_before, num_live_trees_after)
    return num_live_trees_before, num_live_trees_after
# end function run_simulation()

# Returns true if a burn chance passes.
def burn_chance_happens(burn_chance):
    return np.random.randint(0, 101) <= burn_chance

# Returns true if a burn chance passes.
def is_init_tree():
    return np.random.randint(0, 101) <= FOREST_DENSITY

# Run simulation with certain parameters.
# Returns the ratio of burned forests. (forest_destruction)
def run_custom_simulation(bchance=85, fdensity=80, wspeed=0, wdir='down', debug=False):
    global BURN_CHANCE, FOREST_DENSITY, WIND_SPEED, WIND_DIRECTION

    BURN_CHANCE = bchance
    FOREST_DENSITY = fdensity
    WIND_SPEED = wspeed

    if wdir == 'right':
        WIND_DIRECTION = TOWARDS_RIGHT
    elif wdir == 'left':
        WIND_DIRECTION = TOWARDS_LEFT
    elif wdir == 'down':
        WIND_DIRECTION = TOWARDS_BOTTOM
    else:
        WIND_DIRECTION = TOWARDS_TOP

    if debug:
        print('Simulation parameters:')
        print('burn_chance {0}'.format(BURN_CHANCE))
        print('forest_density {0}'.format(FOREST_DENSITY))
        print('wind_speed {0}'.format(WIND_SPEED))
        print('wind_direction {0}'.format(WIND_DIRECTION))

    print('burn chance: {0}'.format(BURN_CHANCE))
    before, after = run_simulation()
    forest_destruction = 1 - (after/before)

    if debug:
        print('Initial: {0} => Final {1}'.format(before, after))
        print('Destruction result: {0}'.format(forest_destruction))

    return forest_destruction

if __name__ == "__main__" :
    # Run simulations we want
    # burn chance

    sample_size = 10

    bchance_datasets = []
    for _ in range(0, sample_size):
        data = []
        for i in range(0, 101):
            data.append(run_custom_simulation(85, 80, i, 'right', True))
        bchance_datasets.append(data)

    # average bchance data
    bchance_avg = []
    for i in range(0, 101):
        bchance_avg.append(0)
        for j in range(0, sample_size):
            bchance_avg[i] += bchance_datasets[j][i]
        bchance_avg[i] /= sample_size

    print('Average windspeed data: {0}'.format(bchance_avg))

    # Run simulation with debug output
    run_custom_simulation(bchance, fdensity, wspeed, wdir, True)
