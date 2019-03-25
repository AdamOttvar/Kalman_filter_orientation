import time
import numpy as np
from numpy import linalg as LA

class EKFOrientationFilter(object):
    """docstring for EKFOrientationFilter"""
    def __init__(self, Ra, Rw, Rm, m0, g0):
        super(EKFOrientationFilter, self).__init__()
        self.Ra = Ra
        self.Rw = Rw
        self.Rm = Rm
        self.m0 = m0
        self.g0 = g0
        self.L = 0
        self.alpha = 0.001
        self.gyr_time_start = 0

    def run_filter(self, x, P, acc, gyro, mag):
        if np.isnan(gyro).any():
            x, P = self.tu_qw_NaN(x, P, gyro, 0.01, self.Rw)
            self.gyr_time_start = time.time()
        else:
            x, P = self.tu_qw(x, P, gyro, 0.01, self.Rw)
            self.gyr_time_start = time.time()

        x, P = self.mu_normalizeQ(x, P)

        if not np.isnan(acc).any():
            x, P = self.mu_g(x, P, acc, self.Ra, self.g0)

        x, P = self.mu_normalizeQ(x, P)


        if not np.isnan(mag).any():
            norm_mag = LA.norm(mag)
            if not self.L == 0:
                self.L = (1-self.alpha)*self.L + self.alpha*norm_mag
            else:
                self.L = norm_mag

            if not ((self.L-10) > norm_mag or norm_mag > (self.L+10)):
                x, P = self.mu_m(x, P, mag, self.Rm, self.m0)

        x, P = self.mu_normalizeQ(x, P)


        return [x, P]

    def Somega(self, w):
        wx, wy, wz = np.array(w).flatten()

        S = np.array([[0, -wx, -wy, -wz],
                    [wx, 0, wz, -wy],
                    [wy, -wz, 0, wx],
                    [wz, wy, -wx, 0]])

        return np.matrix(S)

    def Sq(self, q):
        q0, q1, q2, q3 = np.array(q).flatten()
        S = np.array([[-q1, -q2, -q3],
                    [q0, -q3, q2],
                    [q3, q0, -q1],
                    [-q2, q1, q0]])

        return np.matrix(S)

    def tu_qw(self, x, P, omega, T, Rw):
        A = np.eye(len(x)) + (T/2)*self.Somega(omega)
        G = (T/2)*self.Sq(x)
        Q = G*Rw*np.transpose(G)

        x = A*x
        P = A*P*np.transpose(A) + Q

        return [x, P]

    def tu_qw_NaN(self, x, P, omega, T, Rw):
        A = np.matrix(np.eye(len(x)))
        G = (T/2)*self.Sq(x)
        Q = G*Rw*np.transpose(G)

        x = A*x
        P = A*P*np.transpose(A) + Q

        return [x, P]

    def dQqdq(self, q):
        q0, q1, q2, q3 = np.array(q).flatten()

        Q0 = 2 * np.matrix([[2*q0, -q3, q2],
                         [q3, 2*q0, -q1],
                         [-q2, q1, 2*q0]])

        Q1 = 2 * np.matrix([[2*q1, q2, q3],
                         [q2, 0., -q0],
                         [q3, q0, 0.]])

        Q2 = 2 * np.matrix([[0., q1, q0],
                         [q1, 2*q2, q3],
                         [-q0, q3, 0.]])

        Q3 = 2 * np.matrix([[0., -q0, q1],
                         [q0, 0., q2],
                         [q1, q2, 2*q3]])

        return [Q0, Q1, Q2, Q3]

    def Qq(self, q):
        q0, q1, q2, q3 = np.array(q).flatten()

        Q = np.array([[2*(np.power(q0, 2) + np.power(q1, 2)) - 1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                    [2*(q1*q2 + q0*q3), 2*(np.power(q0, 2) + np.power(q2, 2)) - 1, 2*(q2*q3 - q0*q1)],
                    [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(np.power(q0, 2) + np.power(q3, 2)) - 1]])

        return np.matrix(Q)

    def mu_g(self, x, P, yacc, Ra, g0):
        Q0, Q1, Q2, Q3 = self.dQqdq(x)

        H = np.matrix(np.c_[np.transpose(Q0)*g0, np.transpose(Q1)*g0, np.transpose(Q2)*g0, np.transpose(Q3)*g0])

        S = H*P*np.transpose(H) + Ra
        K = np.dot(P*np.transpose(H), np.linalg.pinv(S))
        P = P - K*S*np.transpose(K)
        x = x + K*(yacc - np.transpose(self.Qq(x))*g0)

        return [x, P]

    def mu_m(self, x, P, mag, Rm, m0):
        Q0, Q1, Q2, Q3 = self.dQqdq(x)
        H = np.matrix(np.c_[np.transpose(Q0)*m0, np.transpose(Q1)*m0, np.transpose(Q2)*m0, np.transpose(Q3)*m0])

        S = H*P*np.transpose(H) + Rm
        K = np.dot(P*np.transpose(H), np.linalg.pinv(S))
        P = P - K*S*np.transpose(K)
        x = x + K*(mag - np.transpose(self.Qq(x))*m0)

        return [x, P]

    def mu_normalizeQ(self, x, P):
        x = x / LA.norm(x)

        if x[0] < 0:
          x = -x

        return [x, P]

if __name__ == "__main__":
    t = np.load("t.npy")
    acc = np.load("acc.npy")
    gyr = np.load("gyr.npy")
    mag = np.load("mag.npy")

    Ra = (10**(-3))*np.matrix('0.14 -0.02 -0.02;-0.02 0.13 0.01;-0.02 0.01 0.28')
    Rw = (10**(-5))*np.matrix('0.11 -0.01 0 ;-0.01 0.11 0;0.01 0 0.11')
    Rm = np.matrix('0.05 -0.02 0.02; -0.02 0.02 -0.0036;0.02 -0.004 0.08')

    m0 = np.matrix(np.array([[0], [14.2597], [-28.3893]]))
    g0 = np.matrix(np.array([[0], [0], [9.82]]))

    ekf = EKFOrientationFilter(Ra, Rw, Rm, m0, g0)


    x = np.matrix(np.array([[1.],[0.],[0.],[0.]]))
    P = np.matrix(np.eye(4, 4))
    xhat = np.array(x)

    ekf.gyr_time_start = time.time()

    for ind in range(t.shape[1]):
        acc_cur = np.matrix(acc[:, ind]).transpose()
        gyr_cur = np.matrix(gyr[:, ind]).transpose()
        mag_cur = np.matrix(mag[:, ind]).transpose()
        x, P = ekf.run_filter(x, P, acc_cur, gyr_cur, mag_cur)

        xhat = np.c_[xhat, x]

    np.save("xhat.npy", xhat)
