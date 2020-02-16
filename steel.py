# MEE322 Lab 1 - Data Plots for Steel

import pandas as pd
import os
import matplotlib.pyplot as plt
import math

d_0 = 2.513 # in mm
d_f = 1.54 # in mm

area = (d_0/2 * 10 ** -3) ** 2 * math.pi # Cross-sectional area of material, m^2
area_f = (d_f/2 * 10 ** -3) ** 2 * math.pi # final area, m^2
length = 21.11 # Original length, mm

dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv (dir_path + r'\steel.csv')

disp = df['Displacement (mm)']
stress = df['Engineering Stress (GPa)']

strain = disp / length # Convert to engineering strain
stress *= 1000 # Convert from GPa to MPa

### a ----------------------------------

fig1 = plt.figure()
plt.plot(strain, stress)
plt.grid(b=True, which='major', color='#666666', linestyle='-')

xlim1, xlim2, ylim1, ylim2 = plt.axis()

plt.xlim([0, xlim2])
plt.ylim([0, ylim2])

plt.title('Engineering Stress-Strain for Steel Sample')
plt.xlabel('Engineering Strain')
plt.ylabel('Engineering Stress (MPa)')

### b, c, d, e, f ----------------------------------

import numpy as np

begin = 50
threshold = 200 # first n values to perform regression on
offset = 0.002 # Axis offset for finding yield strength
least_error = 1000000 # Large dummy value

strain_l = np.array(strain[begin:threshold]) # Linear elastic portion
stress_l = np.array(stress[begin:threshold])

def lin_reg(arr_x, arr_y): # Performs linear regression with linear algebra
    A = [np.ones(len(arr_x)), arr_x]
    b = arr_y

    A_trans = np.array(A)
    A = np.transpose(A)
    coeff = np.linalg.inv(A_trans.dot(A))

    return (coeff.dot(A_trans)).dot(b)  # Gets y-int, slope of function

def line_graph(x, u):
    return (x) * u[1] + u[0]

u_1 = lin_reg(strain_l, stress_l)

x_1 = np.array([0.,1.]) # X-range for linear plot, only two points needed

y_1 = line_graph(x_1, u_1) # Corresponding y-values of linear regression
x_1 += offset

plt.plot(x_1, y_1)

for i in range(len(strain)):
    error = abs(line_graph(strain[i]-offset, u_1)) # Finds error between offset linear regression and actual point
    if (error < least_error):
        yield_stress = stress[i]
        yield_point = i
        least_error = error

for i in range(yield_point + int(len(strain)/4), len(strain)): # Finds significant deviation from slope pattern in given area
    slope1 = (stress[i-3]-stress[i-6])/(strain[i-3]-strain[i-6])
    slope2 = (stress[i]-stress[i-3])/(strain[i]-strain[i-3])

    if (abs(slope2/slope1)>4):
        fracture_stress = stress[i-3]
        break_point = i-3
        break;

fracture_strength = fracture_stress*area/area_f
elongation = strain[break_point] * 100 # %elongation

ultimate_stress = max(stress)

area_reduction = abs(area_f-area)/area * 100

print("Young's Modulus: ", u_1[1]/1000, "GPa")
print("Yield Strength: ", yield_stress, "MPa")
print("Ultimate Strength: ", ultimate_stress, "MPa")
print("Fracture Strength: ", fracture_strength, "MPa")
print("Percent Elongation:", elongation, "%")
print("Percent Reduction in Area:", area_reduction, "%")

### g, h, i ----------------------------------

def find_index(array, value):
    for i in range(len(array)):
        if (array[i] == value):
            index = i
    return index

ultimate_index = find_index(stress, ultimate_stress) # Finds ultimate value of engineering stress

fig2 = plt.figure()

true_strain = np.log(1 + strain)
true_stress = stress*(1 + strain)

plt.plot(true_strain[:ultimate_index], true_stress[:ultimate_index])

plt.grid(b=True, which='major', color='#666666', linestyle='-')

xlim1, xlim2, ylim1, ylim2 = plt.axis()
plt.xlim([0, xlim2])
plt.ylim([0, ylim2])

plt.title('True Stress-Strain for Steel Sample')
plt.xlabel('True Strain')
plt.ylabel('True Stress (MPa)')

yield_index = find_index(stress, yield_stress)

log_strain = np.log(true_strain)
log_stress = np.log(true_stress)

fig3 = plt.figure()

plt.plot(log_strain[yield_index:ultimate_index], log_stress[yield_index:ultimate_index])

plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.title('Log-Log True Stress-Strain for Steel Sample')
plt.xlabel('Log True Strain')
plt.ylabel('Log True Stress')

u_2 = lin_reg(log_strain[yield_index:ultimate_index], log_stress[yield_index:ultimate_index]) # Linear regression
                                                                        # for log-log plot

x_2 = np.array([log_strain[yield_index], log_strain[ultimate_index]]) # X-range for linear plot, only two points needed

y_2 = line_graph(x_2, u_2) # Corresponding y-values of linear regression

plt.plot(x_2, y_2)

stress_coefficient = math.exp(u_2[0]) # Uses y-intercept of log-log graph, which is e^(y-int)
hardening_coefficient = u_2[1] # slope of log-log graph

print("Stress Coefficient:", stress_coefficient)
print("Hardening Coefficient:", hardening_coefficient, "MPa")

plt.show()
