import numpy as np
import matplotlib.pyplot as plt
import random
import math
from numba import jit, float64

import time
start_time = time.time()
data = np.loadtxt('parameters.txt', dtype={'names': ('key', 'value'), 'formats': ('S5', 'f4')})
n = int(data[data['key'] == b'n']['value'][0])
M = data[data['key'] == b'm']['value'][0]
a = data[data['key'] == b'a']['value'][0]
f = data[data['key'] == b'f']['value'][0]
epsilon = data[data['key'] == b'e']['value'][0]
L = 1.22*a*(n-1) +0.01*a
r_max = data[data['key'] == b'r']['value'][0]
T = data[data['key'] == b'T']['value'][0]
S_d = data[data['key'] == b'S_d']['value'][0]
S_o = data[data['key'] == b'S_o']['value'][0]
#number of atoms
N = n**3

print(L)
#elementary cell
b_0 = np.array([a,0,0])
b_1 = np.array([a/2,a*np.sqrt(3)/2,0])
b_2 = np.array([a/2,a*np.sqrt(3)/6,a*np.sqrt(2/3)])
#initalize r(t=0)
@jit(nopython=True)
def init():
    R = np.zeros((N,3), dtype=np.float64)
    # for iii in range(n):
    #      for ii in range(n):
    #          for ic in range(n):
    #              i=ic+ii*n+iii*n**2
    #              R[i]=(ic-(n-1)/2)*b_0+(ii-(n-1)/2)*b_1+(iii-(n-1)/2)*b_2
    indices = np.arange(N)
    ic = indices % n
    ii = (indices // n) % n
    iii = indices // (n ** 2)
    R = (ic - (n - 1) / 2)[:, np.newaxis] * b_0 + (ii - (n - 1) / 2)[:, np.newaxis] * b_1 + (iii - (n - 1) / 2)[:,np.newaxis] * b_2
    return R
R = init()
#thermal energy
@jit(nopython=True)
def E(T):
    k = 8.31/1000
    w = random.uniform(0.00000001,1)
    return -0.5*k*T*np.log(w)
###momentum and initial momentum calculation
@jit(nopython=True)
def momentum(T):
    g = random.uniform(0.0000001,1)
    if g >= 0.5:
        p_x = np.sqrt(2*M*E(T))
    else:
        p_x = -np.sqrt(2*M*E(T))
    g = random.uniform(0.0000001,1)
    if g >= 0.5:
        p_y = np.sqrt(2*M*E(T))
    else:
        p_y = -np.sqrt(2*M*E(T))
    g = random.uniform(0.0000001,1)
    if g >= 0.5:
        p_z = np.sqrt(2*M*E(T))
    else:
        p_z = -np.sqrt(2*M*E(T))
    vec = np.array([p_x,p_y,p_z])
    return vec
#print( R)

@jit(nopython=True)
def init_momentum():
    x = np.zeros((N,3), dtype=np.float64)
    ped = np.zeros((N,3), dtype=np.float64)
    xx = 0
    yy = 0
    zz = 0
    for i in range(N):
        vec = momentum(T)
        x[i,0] = vec[0]
        xx = xx + vec[0]
        x[i,1] = vec [1]
        yy = yy + vec[1]
        x[i,2] = vec[2]
        zz = zz + vec[2]
    big_P = np.array([xx,yy,zz], dtype=np.float64)
    for i in range(N):
        ped[i,0] = x[i,0] - big_P[0]/N
        ped[i,1] = x[i,1] - big_P[1]/N
        ped[i,2] = x[i,2] - big_P[2]/N
    return ped
ped = init_momentum()

###potential
@jit(nopython=True)
def V_p(r):
    return epsilon*((r_max/r)*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r) - 2*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r)*(r_max/r))
@jit(nopython=True)
def V_s(r):
    if r < L:
        x = 0
    elif r >= L:
        x = 0.5*f*(r - L)*(r - L)
    return x
@jit(nopython=True)
def V_all(r):
    g = 0
    h = 0
    for i in range(0, N):
        x_prim = r[i, 0]
        y_prim = r[i, 1]
        z_prim = r[i, 2]
        r_prim = np.sqrt(x_prim *x_prim + y_prim *y_prim + z_prim *z_prim)
        # print(r_prim)
        h = h + V_s(r_prim)
        for j in range(i):
            x = r[i, 0] - r[j, 0]
            y = r[i, 1] - r[j, 1]
            z = r[i, 2] - r[j, 2]
            rr = np.sqrt(x * x + y * y + z * z)
            # print(r_max/r)
            g += V_p(rr)
    return g+h


#forces acting on atoms
@jit(nopython=True)
def F_p(xi,yi,zi,xj,yj,zj):
    rij = np.sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj))
    vec = np.array([xi - xj, yi - yj, zi - zj], dtype=np.float64)
    hm = (12*epsilon*((r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij) - (r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)*(r_max/rij)))/(rij*rij)
    return vec*hm
@jit(nopython=True)
def F_s(xi,yi,zi):
    mod = np.sqrt(xi*xi + yi*yi + zi*zi)
    ri = np.array([xi, yi, zi], dtype=np.float64)
    if mod < L:
        x = np.array([0, 0, 0], dtype=np.float64)
    elif mod >= L:
        a = f*(L - mod)/mod
        x = np.array([a*xi, a*yi, a*zi], dtype=np.float64)
    return x
@jit(nopython=True)
def F_all(r,ile):
    k = np.zeros((ile, 3), dtype=np.float64)
    const = 1 / (4 * np.pi * L *L)
    pressure = 0.0
    for i in range(ile):
        for j in range(i+1,ile,1):
            force = F_p(r[i, 0], r[i, 1], r[i, 2], r[j, 0], r[j, 1], r[j, 2])
            k[i] = k[i] + force
            k[j] = k[j] - force
        s_force = F_s(r[i, 0], r[i, 1], r[i, 2])
        k[i] += s_force

        pressure = pressure + const*np.sqrt(s_force[0]*s_force[0] + s_force[1]*s_force[1] + s_force[2]*s_force[2])
    return k, pressure


time_step = data[data['key'] == b'tau']['value'][0]
time_sim = data[data['key'] == b't']['value'][0]
# solving the motion equations
@jit(nopython=True)
def r_all():
    momentums = np.zeros((int(N * time_sim / time_step), 3))
    momentums_help = np.zeros((int(N * time_sim / time_step), 3))
    rs = np.zeros((int(N * time_sim / time_step), 3), dtype=np.float64)
    G = np.copy(R)
    energy = np.zeros(int(time_sim / time_step), dtype=np.float64)
    kinetic = np.zeros(int(N * time_sim / time_step), dtype=np.float64)
    pressure = np.zeros(int(N * time_sim / time_step), dtype=np.float64)
    temperature = np.zeros(int(N * time_sim / time_step), dtype=np.float64)
    k = 8.31 / 1000
    const = 2 / (3 * N * k)
    potential = 0
    forces = np.zeros((N, 3), dtype=np.float64)
    rs[:N] = G[:N]
    momentums[:N] = ped[:N]
    potential = V_all(G)
    forces, pressure[0] = F_all(G, N)
    kinetic[0] = (np.sum((momentums[0:N, 0] * momentums[0:N, 0] + momentums[0:N, 1] * momentums[0:N, 1] + momentums[0:N, 2] * momentums[0:N, 2])))/(2*M)
    energy[0] = kinetic[0] + potential
    temperature[0] = const*kinetic[0]
    m = 1

    for i in range(1,int(time_sim/time_step) -1):
        k = m*N
        l = (m - 1)*N
        momentums_help[l:l + N] = momentums[l:l + N] + 0.5 * forces * time_step
        rs[k:k + N] = rs[l:l + N] + (momentums_help[l:l+N] / M) * time_step
        forces, pressure[i] = F_all(rs[k:k +N], N)
        momentums[k:k + N]= momentums_help[l :l +N] + 0.5 * forces * time_step
        kinetic[i] = (np.sum((momentums[k:k + N, 0] * momentums[k:k + N, 0] + momentums[k:k + N, 1] * momentums[k:k + N, 1] + momentums[k:k + N, 2] * momentums[k:k + N, 2])))/(2*M)
        potential = V_all(rs[k:k+N])
        energy[i] = kinetic[i] + potential
        temperature[i] = const*kinetic[i]
        m = m+ 1

    return rs,energy,kinetic,pressure,temperature
rr,energy,kinetic,pressure,temp = r_all()
###save r's to file
with open("motion_r.xyz", "w") as outfile:
    for j in range(0,int(time_sim/time_step) -1,100):
        print(N, file=outfile)
        print('argon', file=outfile)
        for i in range(N):
            print('Ar', '\t', rr[j*N + i][0], '\t', rr[j*N + i][1], '\t', rr[j*N + i][2], file=outfile)
    outfile.close()

#################################

with open("energy_total.txt", "w") as outfile_1:
    with open("energy_kinetic.txt", "w") as outfile_2:
        with open("temperature.txt", "w") as outfile_3:
            with open("energy_potential.txt", "w") as outfile_4:
                with open("pressure.txt", "w") as outfile_5:
                    for i in range(0,len(energy) -1,10):
                        print(10*i/(time_sim/time_step),'\t',energy[i], file=outfile_1)
                        print(10 * i / (time_sim / time_step), '\t', kinetic[i], file=outfile_2)
                        print(10 * i / (time_sim / time_step), '\t', temp[i], file=outfile_3)
                        print(10 * i / (time_sim / time_step), '\t', energy[i] - kinetic[i], file=outfile_4)
                        print(10 * i / (time_sim / time_step), '\t', pressure[i], file=outfile_5)
outfile_1.close()
outfile_2.close()
outfile_3.close()
outfile_4.close()
outfile_5.close()

#mean temperature
# def mean_temp():
#     T = 0
#     for i in range(int(S_o),int(S_d +S_o) - 1,1):
#         T = T + temp[i]
#     return T/S_d

# print('Kinet',kinetic)
# print('Mean temp:',mean_temp())
end_time = time.time()

# Oblicz czas trwania działania programu
elapsed_time = end_time - start_time
print('Time: ',elapsed_time)