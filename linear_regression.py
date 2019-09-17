import numpy as np
from matplotlib import pyplot
import math
#from mplkit.mplot3d import axes3d


# Load the data
data = np.loadtxt('linear_regression.txt', delimiter = ',')
#separate predictor from target variable
xd = np.c_[np.ones(data.shape[0]), data[:,0]]
yd = np.c_[data[:,1]]
print(yd.shape)
print(xd.shape)

# First appraoch - Normal equation

def normalEquation(x,y):
    """
    Parameteres: input variables (Table) , Target vector
    Instructions: Complete the code to compute the closed form solution to linear regression and 	save the result in theta.
    Return: coefficinets 
    """
    theta=[]
    x_t=np.transpose(x)
    theta=np.linalg.inv(x_t.dot(x))
    theta=theta.dot(x_t)
    theta=theta.dot(y)
    return theta

# Iterative Approach - Gradient Descent 

'''
Following paramteres need to be set by you - you may need to run your code multiple times to find the best combination 
'''
theta=np.matrix([[0],[0]])
i=1000
r=0.01

def gradient_descent(x,y,theta,i,r):
    """
    Paramters: input variable , Target variable, theta, number of iteration, learning_rate
    Instructions: Complete the code to compute the iterative solution to linear regression, in each iteration you will 
    add the cost of the iteration to a an empty list name cost_hisotry and update the theta.
    Return: theta, cost_history 
    """
    n=float(len(y))
    cost_history = []
    theta1=[]
    theta2=[]
    # Your code goes here
    x_t=np.transpose(x)
    for i in range(0,i):
        theta1.append(theta.item(0,0))
        theta2.append(theta.item(1,0))
        ytrain=x.dot(theta)
        ydiff=y-ytrain
        cost=(np.transpose(ydiff).dot(ydiff))/(2*n)
        cost_history.append(float(cost))
        thetagrad=((x_t).dot(ydiff))/n
        theta=theta+r*(thetagrad)
    return theta, cost_history,theta1,theta2

print(gradient_descent(xd,yd,theta,i,r)[0])
print(normalEquation(xd,yd))


# Plot the cost over number of iterations

#Your plot should be similar to the provided plot

x=range(0,1000)
pyplot.plot(x,gradient_descent(xd,yd,theta,i,r)[1])
pyplot.show()
"""
theta, cost_history,theta1,theta2=gradient_descent(xd,yd,theta,i,r)

# Plot the linear regression line for both gradient approach and normal equation in same plot
'''
hints: your x-axis will be your predictor variable and y-axis will be your target variable. plot a
scatter plot and draw the regression line using the theta calculated from both approaches. Your plot
should be similar to what provided.
'''



# Plot contour plot and 3d plot for the gradient descent approach

'''
your plots should be similar to our plots.

'''

#pyplot.contour(theta1, theta2, cost_history, colors='black');

"""
pyplot.plot(xd,yd,'ro')
pyplot.plot(xd,xd.dot(normalEquation(xd,yd)))
pyplot.plot(xd,xd.dot((gradient_descent(xd,yd,theta,i,r))[0]))
pyplot.show()
