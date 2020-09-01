#! /usr/bin/env python

import numpy as np
import quaternion

from mayavi import mlab

def plot_axes():
    # rot = quaternion.as_rotation_matrix(q)
    mlab.quiver3d(0,0,0,1,0,0,color=(1,0,0),mode='arrow', scale_factor=1, scale_mode='scalar')
    mlab.quiver3d(0,0,0,0,1,0,color=(0,1,0),mode='arrow', scale_factor=1, scale_mode='scalar')
    mlab.quiver3d(0,0,0,0,0,1,color=(0,0,1),mode='arrow', scale_factor=1, scale_mode='scalar')


def plot_interp(q, opacity=1):

    rot = quaternion.as_rotation_matrix(q)
    z =  np.zeros_like(rot)
    mlab.quiver3d(z[0,0,0], z[0,1,0], z[0,2,0], rot[0,0,0], rot[0,1,0], rot[0,2,0], color=(1,0,0), mode='2dhooked_arrow', opacity=opacity)
    mlab.quiver3d(z[0,0,1], z[0,1,1], z[0,2,1], rot[0,0,1], rot[0,1,1], rot[0,2,1], color=(0,1,0), mode='2dhooked_arrow', opacity=opacity)
    mlab.quiver3d(z[0,0,2], z[0,1,2], z[0,2,2], rot[0,0,2], rot[0,1,2], rot[0,2,2], color=(0,0,1), mode='2dhooked_arrow', opacity=opacity)

    mlab.quiver3d(z[-1,0,0], z[-1,1,0], z[-1,2,0], rot[-1,0,0], rot[-1,1,0], rot[-1,2,0], color=(1,0,0), opacity=opacity)
    mlab.quiver3d(z[-1,0,1], z[-1,1,1], z[-1,2,1], rot[-1,0,1], rot[-1,1,1], rot[-1,2,1], color=(0,1,0), opacity=opacity)
    mlab.quiver3d(z[-1,0,2], z[-1,1,2], z[-1,2,2], rot[-1,0,2], rot[-1,1,2], rot[-1,2,2], color=(0,0,1), opacity=opacity)

    mlab.plot3d(rot[:,0,0], rot[:,1,0], rot[:,2,0], color=(1,0,0), tube_radius=0.005, opacity=opacity)
    mlab.plot3d(rot[:,0,1], rot[:,1,1], rot[:,2,1], color=(0,1,0), tube_radius=0.005, opacity=opacity)
    mlab.plot3d(rot[:,0,2], rot[:,1,2], rot[:,2,2], color=(0,0,1), tube_radius=0.005, opacity=opacity)

    mlab.points3d(rot[:,0,0], rot[:,1,0], rot[:,2,0], color=(1,0,0), scale_factor=0.025, opacity=opacity)
    mlab.points3d(rot[:,0,1], rot[:,1,1], rot[:,2,1], color=(0,1,0), scale_factor=0.025, opacity=opacity)
    mlab.points3d(rot[:,0,2], rot[:,1,2], rot[:,2,2], color=(0,0,1), scale_factor=0.025, opacity=opacity)

    r = quaternion.as_float_array(np.log(q))
    z = np.zeros_like(r)
    mlab.quiver3d(z[0,1], z[0,2], z[0,3], r[0,1], r[0,2], r[0,3], scale_factor=1, color=(1,1,1), mode='2dhooked_arrow', opacity=opacity)
    mlab.quiver3d(z[-1,1], z[-1,2], z[-1,3], r[-1,1], r[-1,2], r[-1,3], scale_factor=1, color=(1,1,1), opacity=opacity)
    mlab.points3d(r[:,1], r[:,2], r[:,3], scale_factor=0.025, color=(1,1,1), opacity=opacity)
    mlab.plot3d(r[:,1], r[:,2], r[:,3], tube_radius=0.005, color=(1,1,1), opacity=opacity)


def velocity(q,h):
    return np.abs(q[1:]-q[:-1])/(h[1:]-h[:-1])
    # return np.abs(q[1:-1]-q[:-2])/2 + np.abs(q[1:-1]-q[2:])/2


def nlerp(q0,q1,h):
    ql = q0*(1-h) + q1*h # eq(6.3)
    # return ql
    return ql/np.abs(ql)


def slerp(q0,q1,h):
    return q0 * (q0.conj()*q1)**h  # Slerp eq(6.4)
    # return (q1*q0.conj())**h * q0  # Slerp eq(6.6)
    # return q0**(1-h) * q1**h


def twerp(q0,q1,h):
    r0,r1 = np.log(q0),np.log(q1)
    return np.exp((r1-r0)*h + r0)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    # references
    # https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/functions/exponent/index.htm
    # https://web.mit.edu/2.998/www/QuaternionReport1.pdf
    # http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/ (coordinate free derivation of slerp, and nlerp, normalized lerp)
    # https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf (see sec4.1 for applicatin of differential control) Twerp = exponential map
    
    # r0 = quaternion.from_float_array(np.r_[0,np.random.rand(3)*2-1])
    # r1 = quaternion.from_float_array(np.r_[0,np.random.rand(3)*2-1])

    phi0, phi1 = np.pi/4, np.pi/2
    n0, n1 = np.quaternion(0,1,3,0), np.quaternion(0,-1,2,0)
    n0, n1 = n0/np.abs(n0), n1/np.abs(n1)
    r0, r1 = n0*phi0, n1*phi1

    q0, q1 = np.exp(r0), np.exp(r1)

    # n = np.quaternion(0,0,0,1)
    # phi = 0
    # om = 1
    # dt = 1
    # q0 = np.exp(n*phi)

    # q1 = q0/(1-om*n*dt)
    # q1 = q1/np.abs(q1)

    # print(q0,np.abs(q0))
    # print(q1,np.abs(q1))


    # define interpolation parameter
    h = np.linspace(0,1,11)

    # Twerp
    qt = twerp(q0,q1,h)
    vt = velocity(qt,h)

    # Slerp
    qs = slerp(q0,q1,h)
    vs = velocity(qs,h)

    # Lerp
    ql = nlerp(q0,q1,h)
    vl = velocity(ql,h)

    ri = quaternion.as_float_array(ql)
    x,y,z = ri[:,0], ri[:,1], ri[:,2]

    # sphere
    # pi = np.pi
    # cos = np.cos
    # sin = np.sin
    # phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

    # u = np.sin(phi) * np.cos(theta)
    # v = np.sin(phi) * np.sin(theta)
    # w = np.cos(phi)

    # mlab.plot3d(x,y,z,tube_radius=0.02)
    # mlab.points3d(x,y,z,scale_factor=0.05)
    # mlab.mesh(u,v,w,opacity=0.75)
    # mlab.show()


    # q01 = twerp(q0,q1,h)
    # q10 = twerp(q1,q0,h)
    # print(np.abs(q10-q01))

    # import matplotlib.pyplot as plt
    # plt.plot(h[:-1], vl/np.pi, '-', label='Lerp')
    # plt.plot(h[:-1], vs/np.pi, '--', label='Slerp')
    # plt.plot(h[:-1], vt/np.pi, '-.',  label='Twerp')
    # plt.grid(True)
    # plt.legend()
    # plt.xlabel('Interplation parameter, h')
    # plt.ylabel('Approximate velocity')
    # plt.title('Quaternion Interpolation')
    # plt.show()

    plot_axes()
    # plot_interp(q)
    plot_interp(ql, opacity=0.5)
    mlab.show()
