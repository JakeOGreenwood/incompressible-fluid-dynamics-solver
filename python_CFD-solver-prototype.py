import numpy as np
import tkinter

from tqdm import tqdm
import matplotlib.pyplot as plt

np.seterr(all='raise')

# Some constants
relaxationParameter = 1.7# Show as omega- described page 37
reynolds = 150
dx = 0.1
dy = 0.1
dt = 0.005
max_it = 100


#initialising u and v
grid_len = 20
grid_height = 20
# Velocities in format [height, length, [u,v]]
vel = np.zeros((grid_height+2 ,grid_len+2 , 2))# simulation values are within the center grid, outer values are based on boundary conditions.
momentums = np.zeros((grid_height+2, grid_len+2, 2))# The momentums are in the center, with the boundaries being set depending on boundary conditions used

# Right hand side of pressure equation
rhs = np.zeros((grid_height+2,grid_len+2))

# Pressures in format [j,i]
pressures = np.zeros((grid_height+2, grid_len+2))# The same size as momentums and vel, however with boundary of zero values

#initialising p
p = np.zeros((grid_height,grid_len))

def du2_dx(i, j):
    ''' d(u^2)/d(x), the first calculation in 3.19a'''
    try:
        # The first part of du^2/dx
        val_pt1 = (1/dx) * (((vel[j,i,0] + vel[j,i+1,0])/2) ** 2 - ((vel[j,i-1,0] + vel[j,i,0])/2) ** 2)
        # The gamma / dx before the open bracket
        val_pt2 = upwind_discretization_parameter(i,j) / dx
        # The first half of the large bracket
        val_pt3 = (abs(vel[j,i,0] + vel[j, i+1, 0])/2) * ((vel[j,i,0] - vel[j,i+1,0])/2)
        # The second half of the large bracket
        val_pt4 = (abs(vel[j,i-1,0] + vel[j, i, 0])/2) * ((vel[j,i-1,0] - vel[j,i,0])/2)

        val = val_pt1 + (val_pt2 * (val_pt3 - val_pt4))
        return val
    except:
        print("likely overflow error")
        print(vel)
        print(i,j)
        print(vel[j,i])
        print(vel[j,i-1])
        print(vel[j,i+1])



def dv2_dy(i, j):
    ''' d(v^2)/d(y), from 3.19a'''
    # The first part of dv^2/dy
    val_pt1 = (1/dy) * (((vel[j,i,1] + vel[j+1,i,1])/2) ** 2 - ((vel[j-1,i,1] + vel[j,i,1])/2) ** 2)
    # The gamma / dx before the open bracket
    val_pt2 = upwind_discretization_parameter(i,j) / dy
    # The first half of the large bracket
    val_pt3 = (abs(vel[j,i,1] + vel[j+1, i, 1])/2) * ((vel[j,i,1] - vel[j+1,i,1])/2)
    # The second half of the large bracket
    val_pt4 = (abs(vel[j-1,i,1] + vel[j, i, 1])/2) * ((vel[j-1,i,1] - vel[j,i,1])/2)

    val = val_pt1 + (val_pt2 * (val_pt3 - val_pt4))
    return val

def duv_dy(i,j):
    ''' d(uv)/dy, the second calculation of 3.19a '''
    # the first section of the first brackets
    val_pt1 = ((vel[j,i,1] + vel[j,i+1,1]) / 2) * ((vel[j,i,0] + vel[j+1,i,0]) / 2)
    # The second section of the first brackets
    val_pt2 = ((vel[j-1,i,1] + vel[j-1,i+1,1]) / 2) * ((vel[j-1,i,0] + vel[j,i,0]) / 2)
    # The gamma / dx before second brackets
    val_pt3 = upwind_discretization_parameter(i,j) / dy
    # The first half of the second bracket
    val_pt4 = (abs(vel[j,i,1] + vel[j, i+1, 1])/2) * ((vel[j,i,0] - vel[j+1,i,0])/2)
    # The second half of the second bracket
    val_pt5 = (abs(vel[j-1,i,1] + vel[j-1, i+1, 1])/2) * ((vel[j-1,i,0] - vel[j,i,0])/2)

    val = (1/dy) * (val_pt1 - val_pt2) + val_pt3 * (val_pt4 - val_pt5)
    return val

def duv_dx(i,j):
    ''' d(uv)/dx, in 3.19a '''
    # the first section of the first brackets
    val_pt1 = ((vel[j,i,0] + vel[j+1,i,0]) / 2) * ((vel[j,i,1] + vel[j,i+1,1]) / 2)
    # The second section of the first brackets
    val_pt2 = ((vel[j,i-1,0] + vel[j+1,i-1,0]) / 2) * ((vel[j,i-1,1] + vel[j,i,1]) / 2)
    # The gamma / dx before second brackets
    val_pt3 = upwind_discretization_parameter(i,j) / dx
    # The first half of the second bracket
    val_pt4 = (abs(vel[j,i,0] + vel[j+1, i, 0])/2) * ((vel[j,i,1] - vel[j,i+1,1])/2)
    # The second half of the second bracket
    val_pt5 = (abs(vel[j,i-1,0] + vel[j+1, i-1, 0])/2) * ((vel[j,i-1,1] - vel[j,i,1])/2)

    val = (1/dy) * (val_pt1 - val_pt2) + val_pt3 * (val_pt4 - val_pt5)
    return val

def d2u_dx2(i,j):
    ''' the second derivative of du/dx '''
    val = (vel[j,i+1,0] - 2 * vel[j,i,0] + vel[j,i-1,0]) / (dx ** 2)
    return val

def d2u_dy2(i,j):
    ''' the second derivative of du/dy '''
    val = (vel[j+1,i,0] - 2 * vel[j,i,0] + vel[j-1,i,0]) / (dy ** 2)
    return val

def d2v_dx2(i,j):
    ''' the second derivative of dv/dx '''
    val = (vel[j,i+1,1] - 2 * vel[j,i,1] + vel[j,i-1,1]) / (dx ** 2)

    return val

def d2v_dy2(i,j):
    ''' the second derivative of dv/dy '''
    val = (vel[j+1,i,1] - 2 * vel[j,i,1] + vel[j-1,i,1]) / (dy ** 2)

    return val

def dp_dx(i,j):
    val = (p[j,i+1] - p[j,i]) / dx

    return val

def dp_dy(i,j):
    val = (p[j+1,i] - p[j,i]) / dy
    return val

def upwind_discretization_parameter(i,j):
    ''' Page 30 for details'''
    val_u = vel[j,i,0] * dt / dx
    val_v = vel[j,i,1] * dt / dy
    if val_u < val_v:
        return val_v
    else:
        return val_u

def g(i,j):
    ''' Placeholder for later whole body forces. eg gravity'''
    return 0

def bigF(i,j):
    # The first part inside the brackets
    val_pt1 = (1 / reynolds) * (d2u_dx2(i,j) + d2u_dy2(i,j))

    val_pt2 = 0 - du2_dx(i,j) - duv_dy(i,j) + g(i,j)

    val = vel[j,i,0] + (dt * (val_pt1 + val_pt2))

    return val

def bigG(i,j):
    #First part in brackets
    val_pt1 = (1 / reynolds) * (d2v_dx2(i,j) + d2v_dy2(i,j))

    val_pt2 = g(i,j) - duv_dx(i,j) - dv2_dy(i,j)

    val = vel[j,i,1] + (dt * (val_pt1 + val_pt2))
    return val

def show_graph():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    cell_vel = velocities_average()

    ax.quiver(cell_vel[:,:,0], cell_vel[:,:,1])

    min_pressure = np.min(pressures[1:grid_height,1:grid_len])
    max_pressure = np.max(pressures[1:grid_height,1:grid_len])

    plt.imshow(pressures, origin='lower',vmin=min_pressure,vmax=max_pressure)
    plt.colorbar()
    plt.show()

def momentum_boundaries():
    ''' Sets boundary values for when new momentums have been calculated '''
    momentums[0,:,1] = momentums[1,:,1] * -1
    momentums[grid_height+1,:,1] = momentums[grid_height,:,1] * -1

    momentums[:,0,0] = momentums[:,1,0] * -1
    momentums[:,grid_len+1,0] = momentums[:,grid_len,0] * -1

def no_slip_boundaries():
    for a in range(grid_len):
        vel[0][a] = [0,0]
        vel[grid_height][a] = [0,0]
    for a in range(grid_height):
        vel[a][0] = [0,0]
        vel[a][grid_len] = [0,0]

def lid_driven_boundaries():
        for a in range(grid_len+2):
            vel[0][a] = [0,0]
            vel[grid_height+1][a] = [0,0]

        for a in range(1,grid_len+1):
            vel[grid_height][a] = [2,0]


        for a in range(grid_height+2):
            vel[a][0] = [0,0]
            vel[a][grid_len+1] = [0,0]

def wall_driven_boundary():
    for j in range(grid_height+2):
        vel[j,0,:] = 0
        vel[j,grid_len+1,:] = 0

    for i in range(grid_len + 2):
        vel[0,i,:] = 0
        vel[grid_height+1,i,:] = 0

    for j in range(1,grid_height):
        vel[j,1] = [0,2]

def update_momentums():
    '''
    Calculates momentums in each spot of the matrix - in format [j][i][F,G]
    Has an outer boundary of values dependend on boundary conditions, is of same size as vel
    '''
    for a in range(1, grid_height+1):
        for b in range(1, grid_len+1):
            momentums[a,b,0] = bigF(b,a)
            momentums[a,b,1] = bigG(b,a)
    momentum_boundaries()

def update_rhs():
    '''
    right hand side of poisson equation on 3.38
    '''
    for j in range(1, grid_height+1):
        for i in range(1, grid_len+1):
            val_pt1 = (momentums[j,i,0] - momentums[j,i-1,0]) / dx
            val_pt2 = (momentums[j,i,1] - momentums[j-1,i,1]) / dy
            rhs[j,i] = (1/dt) * (val_pt1 + val_pt2)

def epsilon_north(j):
    '''
    Returns 0 if cell is on top row
    returns 1 if otherwise
    '''
    if j == grid_height:
        return 0
    else:
        return 1
def epsilon_south(j):
    '''
    Returns 0 if cell is on bottom row
    returns 1 if otherwise
    '''
    if j == 1:
        return 0
    else:
        return 1
def epsilon_east(i):
    if i == grid_len:
        return 0
    else:
        return 1
def epsilon_west(i):
    if i == 1:
        return 0
    else:
        return 1

def velocities_average():
    cell_vel = np.zeros((grid_height+2,grid_len+2,2))

    for j in range(1, grid_height+1):
        for i in range(1, grid_len+1):
            cell_vel[j,i,0] = (vel[j,i-1,0] + vel[j,i,0])/2
            cell_vel[j,i,1] = (vel[j-1,i,1] + vel[j,i,1])/2

    return cell_vel

def sor_iteration():
    '''
    calculates the sor method on a single cell - from 3.44

    efficiency could be improved by checking if any epsilon values aren't 1 first

    pressure values at 0 and max+1 are set in the update pressures equation
    '''
    next_pressures = np.zeros((grid_height+2, grid_len+2))# The same size as momentums and vel, however with boundary of zero values

    for i in range(1,grid_len+1):
        for j in range(1, grid_height+1):

                val_pt1 = (1 - relaxationParameter) * pressures[j,i]
                val_pt2 = relaxationParameter / ((epsilon_east(i)+epsilon_west(i))/(dx ** 2) + (epsilon_north(j) + epsilon_south(j)) / (dy ** 2))

                # Section in brackets
                val_pt3 = (epsilon_east(i) * pressures[j,i+1] + epsilon_west(i) * next_pressures[j,i-1]) / (dx ** 2)
                val_pt4 = (epsilon_north(j) * pressures[j+1,i] + epsilon_south(j) * next_pressures[j-1,i]) / (dy ** 2)

                val_pt5 = rhs[j,i]

                next_pressures[j,i] = val_pt1 + val_pt2 * (val_pt3 + val_pt4 - val_pt5)

    pressures[:,:] = next_pressures[:,:]

def update_velocities():
    '''
    updates the velocities from 3.31
    '''
    for i in range(1,grid_len):
        for j in range(1,grid_height):
            vel[j][i][0] = momentums[j,i,0] - (dt/dx) * (pressures[j,i+1] - pressures[j,i])

            vel[j][i][1] = momentums[j,i,1] - (dt/dx) * (pressures[j+1,i] - pressures[j,i])

    # Select boundary conditions

    #wall_driven_boundary()
    lid_driven_boundaries()
    #no_slip_boundaries()

def iteration():
    pressures[:,:] = 0
    update_momentums()
    update_rhs()

    for it in range(max_it):
        sor_iteration()
    update_velocities()

for a in tqdm(range(1000)):
    iteration()

show_graph()
