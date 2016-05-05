
# coding: utf-8

# In[1]:

#Se importan las librerias necesarias
from numpy import *
from matplotlib.pyplot import *
from matplotlib import animation
import maf_IO as IO

# In[ ]:

def U_vec(rho, u, v, E):
    return [rho, rho*u, rho*v, rho*E]

def A_vec(rho, u, v, E, P, tau_xx, tau_xy, tau_yy):
    return [rho*u, rho*u*u + P - tau_xx, rho*u*v - tau_xy, (rho*E + P)*u-tau_xx*u-tau_xy*v]

def B_vec(rho, u, v, E, P, tau_xx, tau_xy, tau_yy):
    return [rho*v, rho*u*v - tau_xy, rho*v*v + P - tau_yy, (rho*E + P)*v-tau_xy*u-tau_yy*v]

def F_vec(rho, u, v, F, G):
    return [zeros_like(u), rho*F, rho*G, - rho*F*u - rho*G*v]

def Tau(nu, u, v, dx, dy):
    tau_xx, tau_xy, tau_yy = zeros_like(u), zeros_like(u), zeros_like(u)
    tau_xx[1:-1,1:-1] = 2.0*nu*(u[1:-1,2:]-u[1:-1,0:-2])/(3.0*dx) - nu*(v[2:,1:-1]-v[0:-2,1:-1])/(3.0*dy)
    tau_xy[1:-1,1:-1] = nu*(v[1:-1,2:]-v[1:-1,0:-2])/(2.0*dx) + nu*(u[2:,1:-1]-u[0:-2,1:-1])/(2.0*dy)
    tau_yy[1:-1,1:-1] = 2.0*nu*(v[2:,1:-1]-v[0:-2,1:-1])/(3.0*dx) - nu*(u[1:-1,2:]-u[1:-1,0:-2])/(3.0*dx)
    return tau_xx, tau_xy, tau_yy

def Uvec_to_Vars(u_vec):
    return [u_vec[0], u_vec[1]/u_vec[0], u_vec[2]/u_vec[0], u_vec[3]/u_vec[0]]

def Ec_estado_p_rho(rho, u, v):
    P0 = 1.01325    #Pa
    K0 = 2.15e3    #Pa
    rho0 = 1.0     #g/cm
    n = 7.0
    p = zeros_like(rho)
    p[:,:] = P0 + K0*((rho[:,:]/rho0)**n-1)/n
    return p

def Condiciones_Fronter(u_pred_N, rho, u, v):
    #Condiciones de frontera para rho
    u_pred_N[0][:,0] = rho[:,1]
    u_pred_N[0][:,-1] = rho[:,-2]
    u_pred_N[0][0,:] = rho[1,:]
    u_pred_N[0][-1,:] = rho[-2,:]
    
    #Condiciones de frontera para u
    u_pred_N[1][:,0] = 0
    u_pred_N[1][:,-1] = 0
    u_pred_N[1][0,:] = u_pred_N[1][0,:]
    u_pred_N[1][-1,:] = u_pred_N[1][-1,:]
    
    #Condiciones de frontera para v
    u_pred_N[2][:,0] = u_pred_N[2][:,1]
    u_pred_N[2][:,-1] = u_pred_N[2][:,-2]
    u_pred_N[2][0,:] = 0
    u_pred_N[2][-1,:] = 0
    
    #Condiciones de frontera para E
    u_pred_N[3][:,0] = u_pred_N[3][:,1]
    u_pred_N[3][:,-1] = u_pred_N[3][:,-2]
    u_pred_N[3][0,:] = u_pred_N[3][1,:]
    u_pred_N[3][-1,:] = u_pred_N[3][-2,:]
    
    return u_pred_N

def MacCormack(nu, rhon, un, vn, En, Pn, f, g, dx, dy, dt, ciclo):
    u_pred_n = [zeros_like(rhon), zeros_like(rhon), zeros_like(rhon), zeros_like(rhon)]
    u_vec_n1 = [zeros_like(rhon), zeros_like(rhon), zeros_like(rhon), zeros_like(rhon)]
    
    tau_XX, tau_XY, tau_YY = Tau(nu, un, vn, dx, dy)
    u_vec_n = U_vec(rhon, un, vn, En)
    a_vec_n = A_vec(rhon, un, vn, En, Pn, tau_XX, tau_XY, tau_YY)
    b_vec_n = B_vec(rhon, un, vn, En, Pn, tau_XX, tau_XY, tau_YY)
    f_vec_n = F_vec(rhon, un, vn, f, g)

    if ciclo%5<>0:
        for i in range(4): 
            u_pred_n[i][1:-1,1:-1] = u_vec_n[i][1:-1,1:-1]                                    - dt/dx * (a_vec_n[i][1:-1,2:]-a_vec_n[i][1:-1,1:-1])                                    - dt/dy * (b_vec_n[i][2:,1:-1]-b_vec_n[i][1:-1,1:-1])                                    + dt * f_vec_n[i][1:-1,1:-1]
    else:
        for i in range(4): 
            u_pred_n[i][1:-1,1:-1] = 0.25*(u_vec_n[i][1:-1,0:-2]+u_vec_n[i][1:-1,2:]                                    +u_vec_n[i][0:-2,1:-1]+u_vec_n[i][2:,1:-1])                                    - dt/dx * (a_vec_n[i][1:-1,2:]-a_vec_n[i][1:-1,1:-1])                                    - dt/dy * (b_vec_n[i][2:,1:-1]-b_vec_n[i][1:-1,1:-1])                                    + dt * f_vec_n[i][1:-1,1:-1]
                    
    u_pred_n = Condiciones_Fronter(u_pred_n, rhon, un, vn)
    Vars = Uvec_to_Vars(u_pred_n)
    tau_XX_pred, tau_XY_pred, tau_YY_pred = Tau(nu, Vars[1], Vars[2], dx, dy)
    Pn_pred = Ec_estado_p_rho(Vars[0], Vars[1], Vars[2])

    a_pred_n = A_vec(Vars[0], Vars[1], Vars[2], Vars[3], Pn_pred, tau_XX, tau_XY, tau_YY)
    b_pred_n = B_vec(Vars[0], Vars[1], Vars[2], Vars[3], Pn_pred, tau_XX, tau_XY, tau_YY)
    f_pred_n = F_vec(Vars[0], Vars[1], Vars[2], f, g)
    
    for i in range(4): 
        u_vec_n1[i][1:-1,1:-1] = 0.5*(u_vec_n[i][1:-1,1:-1] + u_pred_n[i][1:-1,1:-1]                                      - dt/dx * (a_pred_n[i][1:-1,2:]-a_pred_n[i][1:-1,1:-1])                                      - dt/dy * (b_pred_n[i][2:,1:-1]-b_pred_n[i][1:-1,1:-1])
                                      + dt * f_pred_n[i][1:-1,1:-1])    

    u_vec_n1 = Condiciones_Fronter(u_vec_n1, rhon, un, vn)
    
    Varsn1 = Uvec_to_Vars(u_vec_n1)
    Pn1 = Ec_estado_p_rho(Varsn1[0], Varsn1[1], Varsn1[2])
    
    return Varsn1[0], Varsn1[1], Varsn1[2], Varsn1[3], Pn1

def vorticidad(un, vn, dx, dy):
    Xi = zeros_like(un)
    Xi[2:-2,2:-2] = (un[0:-4,2:-2]-8*un[1:-3,2:-2]+8*un[3:-1,2:-2]-un[4:,2:-2])/(12*dy)                    -(vn[2:-2,0:-4]-8*vn[2:-2,1:-3]+8*vn[2:-2,3:-1]-vn[2:-2,4:])/(12*dx)
    return Xi

def Avance_en_tiempo(u0, v0, p0, rho0, E0, nu, f, g, dx, dy, x, y, sigma, NT, ny, nx, flag, min_step, tol, f_prefix):
    t = 0.0
    un = zeros((ny, nx))
    vn = zeros((ny, nx))
    Pn = zeros((ny, nx))
    rhon = zeros((ny, nx))
    En = zeros((ny, nx))
    Xin = zeros((ny, nx))
    
    if flag: 
        Uhist=[]
        Vhist=[]
        Phist=[]
        Rhist=[]
        Ehist=[]
        Xhist=[]
        thist=[]
    
    u = u0
    v = v0
    P = p0
    rho = rho0
    E = E0
    
    for n in range(NT):        
        if n%100==0: print "Ciclo:"+str(n)
            
        #Se copia el valor de la funcion u en el arreglo un
        un = u.copy()
        vn = v.copy()
        Pn = P.copy()
        rhon = rho.copy()
        En = E.copy()
        Xin = vorticidad(un, vn, dx, dy)
        
        if flag:                         
            Uhist.append(un)
            Vhist.append(vn)
            Phist.append(Pn)
            Rhist.append(rhon)
            Ehist.append(En)
            Xhist.append(Xin)
            thist.append(t)

        ReD = min(amin(rho*un)*dx, amin(rho*vn)*dy)/nu
        dt_cfl = 0.9/(amax(un/dx)+amax(vn/dy)+sqrt(amax(un)+amax(vn))*sqrt(1.0/dx**2+1.0/dy**2))
        dt = sigma*ReD*dt_cfl/(ReD+2.0)
        
        t = t+dt
        rho, u, v, E, P = MacCormack(nu, rhon, un, vn, En, Pn, f, g, dx, dy, dt, n)
        
        Xin = vorticidad(u, v, dx, dy)
        if max(amax(u), amax(v)) > 1000:
            raise RuntimeError, u"Tuve que terminar la sesión, ciclo {}".format(n)
        # aqui guardar los datos y limpiar
       # if n%100==0:
       #     space = [array(thist), y, x]
       #     data = array([Rhist, Uhist, Vhist, Ehist, Phist, Xhist])
       #     IO.outwrite(space, data, samp_steps=[min_step, None, None], fprefix=f_prefix, tol=tol)
    if flag:
        return Rhist, Uhist, Vhist, Ehist, Phist, Xhist, thist
    else:
        return rho, u, v, E, P, Xin, t


# In[ ]:

lx = 4.
ly = 1.
nx = 100
ny = 100
NT = 10000

dx = lx/(nx-1)
dy = ly/(ny-1)
sigma = 0.01
nu = 0.01

salto = 50

N = 5
    
x = linspace(0., lx, nx)
y = linspace(0., ly, ny)
X, Y = meshgrid(x, y)

v0 = zeros((ny,nx))
u0 = 105.0*tanh(50.0*(Y-0.5*ly))*exp(-100.0*(Y-0.5*ly)**2)

v0 = 25*sin(9.0*pi*X/lx)

rho0 = 1.0-0.001*tanh(50.0*(Y-0.5*ly))

E0 = 10*ones((ny,nx))+rho0*(u0*u0+v0*v0)
p0 = Ec_estado_p_rho(rho0, u0, v0)

ReD = min(amin(rho0[1:-1,1:-1]*u0[1:-1,1:-1]*dx),amin(rho0[1:-1,1:-1]*v0[1:-1,1:-1]*dy))/nu
dt_cfl = 0.9/(amax(u0[1:-1,1:-1]/dx)+amax(v0[1:-1,1:-1]/dy)                +sqrt(amax(u0[1:-1,1:-1])+amax(v0[1:-1,1:-1]))*sqrt(1.0/dx**2+1.0/dy**2))
dt = sigma*dt_cfl/(1.0+2.0/ReD)

print "dt = "+str(dt)

F = -1000.0*cos(pi/3.0)*ones((ny, nx))
G = -1000.0*sin(pi/3.0)*ones((ny, nx))

min_step = 10*dt
tol = 0.15
f_prefix = "test"


Rhoh, Uh, Vh, Eh, Ph, Xih, th =  Avance_en_tiempo(u0, v0, p0, rho0, E0, nu, F, G, dx, dy, x, y, sigma, NT+1, ny, nx, True, min_step, tol, f_prefix)

UMag = [sqrt(Uh[i]**2+Vh[i]**2) for i in range(0,len(Uh))]

figanim = figure(figsize=(8,8));
Ax = figanim.add_subplot(211);
Bx = figanim.add_subplot(212);

cbar_ax = figanim.add_axes([0.85, 0.57, 0.02, 0.3]);
cbar_bx = figanim.add_axes([0.85, 0.13, 0.02, 0.3]);
subplots_adjust(wspace = 0.5);


XihMax = amax(Xih)
XihMin = amin(Xih)

RhohMax = amax(Rhoh)
RhohMin = amin(Rhoh)

LevelsXi = linspace(XihMin, XihMax, 25)
Levelsrho = linspace(RhohMin,RhohMax, 25)

def init():
    Ax.cla()
    Bx.cla()
    campo_vectorial = Ax.quiver([],[],[],[])
    return campo_vectorial,

# Esta funcion se llama de manera secuencial para cada elemento i.
def animate(i,ax,bx,fig):
    NN = i*salto
    ax.cla()
    bx.cla()
    cbar_ax.cla()
    cbar_bx.cla()
    ax.set_title(u'Campo de vorticidad en t='+str(th[NN]))
    bx.set_title(u'Campo de densidad en t='+str(th[NN]))
    figanim.subplots_adjust(right=0.8)
    contornof = ax.contourf(X, Y, Xih[NN], LevelsXi)
    figanim.colorbar(contornof, cbar_ax)
    contornof2 = bx.contourf(X, Y, Rhoh[NN], Levelsrho)
    figanim.colorbar(contornof2, cbar_bx)
    campo_vectorial = ax.quiver(X[::N,::N], Y[::N,::N], Uh[NN][::N,::N], Vh[NN][::N,::N])
    return contornof, contornof2, campo_vectorial,

#anim=animation.FuncAnimation(figanim, animate, init_func=init, frames=len(Uh)/salto, 
                               fargs=(Ax, Bx, figanim), interval=5);
#anim.save("mi_anim.mp4",fps=20);