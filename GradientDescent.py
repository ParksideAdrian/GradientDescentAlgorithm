
from numpy import *


""" 
This function will compute the error for a line, given a data set named "points"
It will take a line's slope value, m
and that line's y-intercept, b , as inputs
"""


def compute_error_for_line_given_points(b, m, points):
    total_error = 0                              #Initializing totalError to 0
    for i in range(0, len(points)):              #Iteratively find the sum of errors for every data point in "points"
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2    #Total Error is defined as the sum:
    return total_error / float(len(points))      #Of all (y - (m*x + b))**2 in the dataset
                                                 #return the totalError / by the number of points in the dataset


'''
This function will find a more optimal line of best fit, given another line of best fit
This function accepts the slope and y-int of a line of best fit to data
This function also accepts a specified learning rate (tuning knob)

'''


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0                                                      # Initialize b_gradient to 0
    m_gradient = 0                                                      # Initialize m_gradient to 0
    N = float(len(points))                                              # Let N be the number of data points
    for i in range(0, len(points)):                                     # Iterate through data set "Points"
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))      # gradient is calculated as the derivative of the function with respect to b, at the ith data point
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))  # gradient is calculated as the derivative of the function with respect to m, at the ith data point
    new_b = b_current - (learning_rate * b_gradient)                     # New b and m values are obtained by subtracting the (gradient*learningRate) from original b and m values
    new_m = m_current - (learning_rate * m_gradient)
    return[new_b, new_m]


'''
This function will find the ideal b and m values for a data set, by accepting starting values for b and m,
and performing the step_gradient() function as many times as num_iterations is set to
When the step_gradient() function is called, it takes b and m values, and returns ones with a lower error value
gradient_descent_runner() will perform that as many times as specified, and return the optimal b and m values for a function
'''


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b                                                  # initialize b and m values
    m = starting_m
    for i in range(num_iterations):                                 # num_iterations is the amount of times
        b, m = step_gradient(b, m, array(points), learning_rate)    # new b and m values will be calculated
        print('After {0} iterations b = {1}, m = {2}. error = {3}'  #prints live updates as gradient is descended
              .format(i, b, m, compute_error_for_line_given_points(b, m, points)))
    return [b, m]                                                   # most accurate b and m values are returned


'''
This function will declare the variable points as a list, with the data from data.csv
this function specifies the learning rate, and original "guesses" of b and m

'''


def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001          # This is the optimal learning rate, but try 0.00001 and 0.000001
                                    # To see how long it takes to reach the minimum error value
    initial_b = 0                   # initial b value
    initial_m = 0                   # initial m value
    num_iterations = 1000           # Another tuning knob, tweak to see its effects
    print('Starting Gradient Descent at b = {0}, m = {1}, error = {2}'      # Prints the starting b and m values, and the error associated with them
          .format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running . . . ")
    [b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)    # This will iteratively adjust the b and m values until they become optimal
    print('After {0} iterations b = {1}, m = {2}. error = {3}'
          .format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))


if __name__ == '__main__':

    run()
