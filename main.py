import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import *
from matplotlib.widgets import Slider

# Function to compute the derivative at a given point (x, y)
def derivative(x, y):
    return x+y*x

# Euler's Method
def euler(x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h*derivative(x[i], y[i])
        x[i+1] = x[i] + h
    return x, y

# RK2 Method
def rk2(x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        k1 = h*derivative(x[i], y[i])
        k2 = h*derivative(x[i]+h/2, y[i]+k1/2)
        y[i+1] = y[i] + k2
        x[i+1] = x[i] + h
    return x, y

# RK4 Method
def rk4(x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        k1 = h*derivative(x[i], y[i])
        k2 = h*derivative(x[i]+h/2, y[i]+k1/2)
        k3 = h*derivative(x[i]+h/2, y[i]+k2/2)
        k4 = h*derivative(x[i]+h, y[i]+k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        x[i+1] = x[i] + h
    return x, y

#Exact/LSODA Method
def lsoda(x0,y0,h,n):
    x = np.linspace(x0,x0+n*h)
    y = odeint(derivative, y0, x, tfirst=True)
    return x, y

#Exact Solution
def exact(x0, y0, h, n):
    xn = x0+h*n
    x = np.linspace(x0,xn)

    #Enter the exact function here
    y = np.sin(x)

    return x,y

    
# Set the initial conditions and step size
x0 = 0
y0 = 0
x1 = 5
h = 0.5
n = int((x1-x0)/h)

x_euler, y_euler = euler(x0, y0, h, n)
x_rk2, y_rk2 = rk2(x0, y0, h, n)
x_rk4, y_rk4 = rk4(x0, y0, h, n)
x_lsoda, y_lsoda = lsoda(x0,y0,h,n)
x_exact, y_exact = exact(x0,y0,h,n)

x = np.array([x_euler,x_rk2,x_rk4]).T
y = np.array([y_euler,y_rk2,y_rk4]).T

#Plotting the points from different methods
fig, ax = plt.subplots()
lines = ax.plot(x,y, 'o-',lw=2,label=["Euler","RK2","RK4"])
lsoda_line = ax.plot(x_lsoda,y_lsoda,label="Exact")
ax.legend(loc="upper right")

#Displaying absolute error value for each methods
errorText = "Error in Euler's Method: %.4f\nError in RK2's Method: %.4f\nError in RK4's Method: %.4f"
text = fig.text(0.15,0.9,errorText%(abs(y_euler[-1]-y_lsoda[-1])[0],abs(y_rk2[-1]-y_lsoda[-1])[0],abs(y_rk4[-1]-y_lsoda[-1])[0]),fontdict={"fontsize":12})

# Make a horizontal slider to control the step_size.
axH = fig.add_axes([0.15, 0.03, 0.7, 0.03])
h_slider = Slider(
    ax=axH,
    label='StepSize',
    valmin=0.05,
    valmax=1,
    valinit=h,
    valstep=0.05
)


# Update the plot when slider is changed
def update(h):
    n = round((x1-x0)/h)
    x_euler, y_euler = euler(x0, y0, h, n)
    x_rk2, y_rk2 = rk2(x0, y0, h, n)
    x_rk4, y_rk4 = rk4(x0, y0, h, n)


    x = np.array([x_euler,x_rk2,x_rk4]).T
    y = np.array([y_euler,y_rk2,y_rk4]).T
    for i in range(3):
        lines[i].set_data(x[:,i],y[:,i])
    text.set_text(errorText%(abs(y_euler[-1]-y_lsoda[-1])[0],abs(y_rk2[-1]-y_lsoda[-1])[0],abs(y_rk4[-1]-y_lsoda[-1])[0]))
        
h_slider.on_changed(update)

plt.show()

