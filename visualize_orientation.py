import time
import numpy as np
import math
import matplotlib.pyplot as plt
import CubeAxes
import EKFOrientationFilter
import serial

# If set to True the script will read inputs from a 
# serial port, with format:
# ax, ay, az, gx, gy, gz, mx, my, mz
# If set to False it will read saved data and plot
# Google orientation estimation against own EKF
LIVE = False

if __name__ == "__main__":
    
    if not LIVE:
        fig1 = plt.figure(figsize=(4, 4))
        fig1.suptitle('Google orientation')
        ax1 = CubeAxes.CubeAxes(fig1)
        fig1.add_axes(ax1)
        

        fig2 = plt.figure(figsize=(4, 4))
        fig2.suptitle('Own orientation')
        ax2 = CubeAxes.CubeAxes(fig2)
        fig2.add_axes(ax2)

        # orientations.npy contains the google calculated orientations
        # which are used for comparison of our filter
        orientations = np.load("input/orientations.npy")
        #orientations = np.load("orient_for_figs.npy")

        # Loading the inputs to the filter
        t = np.load("input/t.npy")[:,4:]
        acc = np.load("input/acc.npy")[:,4:]
        gyr = np.load("input/gyr.npy")[:,4:]
        mag = np.load("input/mag.npy")[:,4:]

        # Filter coefficients
        Ra = (10**(-3))*np.matrix('0.14 -0.02 -0.02;-0.02 0.13 0.01;-0.02 0.01 0.28')
        Rw = (10**(-5))*np.matrix('0.11 -0.01 0 ;-0.01 0.11 0;0.01 0 0.11')
        Rm = np.matrix('0.05 -0.02 0.02; -0.02 0.02 -0.0036;0.02 -0.004 0.08')
        m0 = np.matrix(np.array([[0.0], [12.90], [-43.61]]))
        g0 = np.matrix(np.array([[0.038], [0.14], [9.77]]))

        # The EKF filter class
        ekf = EKFOrientationFilter.EKFOrientationFilter(Ra, Rw, Rm, m0, g0)

        # Start state
        x = np.matrix(np.array([[1.],[0.],[0.],[0.]]))
        P = np.matrix(np.eye(4, 4))
        xhat = np.array(x)


        for col in range(orientations.shape[1]):
            ax1.draw_cube()
            ax1.current_rot = CubeAxes.Quaternion(orientations[:, col])
            ax2.draw_cube()

            acc_cur = np.matrix(acc[:, col]).transpose()
            gyr_cur = np.matrix(gyr[:, col]).transpose()
            mag_cur = np.matrix(mag[:, col]).transpose()
            x, P = ekf.run_filter(x, P, acc_cur, gyr_cur, mag_cur)
            ax2.current_rot = CubeAxes.Quaternion(np.array(x).flatten())
            
            plt.pause(0.0001)

        plt.show()
    
    else:

        fig1 = plt.figure(figsize=(4, 4))
        fig1.suptitle('Live orientation')
        ax1 = CubeAxes.CubeAxes(fig1)
        fig1.add_axes(ax1)

        # Filter coefficients
        Ra = np.matrix(np.array([[[ 3.24199269e-05 , 3.72194970e-06 , 1.90389054e-06],
                                    [ 3.72194970e-06 , 4.89102728e-05 ,-2.73997851e-06],
                                    [ 1.90389054e-06, -2.73997851e-06,  3.93395622e-05]]]))
        Rw = 1000*np.matrix(np.array([[2.20908885e-05, 3.79946899e-07, 1.02907305e-06],
                                    [3.79946899e-07, 7.16164970e-06, 1.47690202e-07],
                                    [1.02907305e-06, 1.47690202e-07, 2.35129789e-05]]))
        Rm = 100*np.matrix(np.array([[ 0.09463006,  0.01083826, -0.00551019],
                                [ 0.01083826,  0.12989043,  0.05544334],
                                [-0.00551019,  0.05544334,  0.23533449]]))
        m0 = np.matrix(np.array([[29.89], [-47.62], [12.36]]))
        #g0 = np.matrix(np.array([[-6.9e-03], [-4.77e-01], [9.36]]))
        g0 = np.matrix(np.array([[-4.77e-01], [9.36], [-6.9e-03]]))

        # The EKF filter class
        ekf = EKFOrientationFilter.EKFOrientationFilter(Ra, Rw, Rm, m0, g0)

        # Start state
        x = np.matrix(np.array([[1.],[0.],[0.],[0.]]))
        P = np.matrix(np.eye(4, 4))
        xhat = np.array(x)

        ser = serial.Serial('COM4', 115200)
        ser.flushInput()

        while True:
            ser_bytes = ser.readline()
            try: 
                decoded_bytes = ser_bytes.decode("utf-8")
                ax, ay, az, gx, gy, gz, mx, my, mz = decoded_bytes.strip().split(",")
            except UnicodeDecodeError:
                print("Error decoding: ")
                print(ser_bytes)
                ser.flushInput()
                continue
            except ValueError:
                print("Error splitting: ")
                print(decoded_bytes)
                continue

            try:
                acc = np.array([[float(ay)], [float(az)], [float(ax)]])
                gyr = np.array([[float(gy)], [float(gz)], [float(gx)]])
                mag = np.array([[float(my)], [float(mz)], [float(mx)]])
            except ValueError as VE:
                print(str(VE))
                continue

            ax1.draw_cube()

            x, P = ekf.run_filter(x, P, acc, gyr, mag)
            ax1.current_rot = CubeAxes.Quaternion(np.array(x).flatten())
            
            plt.pause(0.0001)

        plt.show()


