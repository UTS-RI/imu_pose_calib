# This scripts allows to perform extrinsic calibration between IMU and pose (from Vicon or other system)
# The data is collected in ros bags and then processed by this script
# The script will output the transformation between IMU and pose sensor as well as the IMU trajectory and the estimated biases

import os
import numpy as np
import argparse
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

import pose_utils

import rosbag

def invSkewSym(v):
    return np.array([v[2, 1], v[0, 2], v[1, 0]])

def skewSym(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def rightJacobianSO3(r):
    theta = np.linalg.norm(r)
    if theta < 1e-10:
        return np.eye(3)
    else:
        r_hat = skewSym(r)
        return np.eye(3) - ((1-np.cos(theta))/(theta**2)) * r_hat + ((theta-np.sin(theta))/(theta**3)) * (r_hat @ r_hat)

def getAccData(bag, topic):
    acc_data = []
    acc_time = []
    for topic, msg, t in bag.read_messages(topics=[topic]):
        acc_data.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        acc_time.append(msg.header.stamp)
    return np.array(acc_data), acc_time

def getGyrData(bag, topic):
    gyr_data = []
    gyr_time = []
    for topic, msg, t in bag.read_messages(topics=[topic]):
        gyr_data.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        gyr_time.append(msg.header.stamp)
    return np.array(gyr_data), gyr_time



def kernelSE(X1, X2, hyper):
    return (hyper['sf']**2)*np.exp(-(pairwise_distances(X1,X2,'sqeuclidean'))/(2*(hyper['l']**2)))

def kernelSEDiff(X1, X2, hyper, axis=0):
    k = (hyper['sf']**2)*np.exp(-pairwise_distances(X1,X2,'sqeuclidean')/(2*(hyper['l']**2)))
    return -k*((X1[:,axis][:,np.newaxis])-(X2[:,axis][np.newaxis,:])) / (hyper['l']**2)

def kernelSEDiff2(X1, X2, hyper, axis=0):
    k = (hyper['sf']**2)*np.exp(-pairwise_distances(X1,X2,'sqeuclidean')/(2*(hyper['l']**2)))
    return k*(((X1[:,axis][:,np.newaxis])-(X2[:,axis][np.newaxis,:]))**2/(hyper['l']**4) - 1/(hyper['l']**2))


def filterWithGP(x, y, chunck_size = 5):
    h = {}
    h['l'] = 5 * np.median(np.diff(x))

    out = np.zeros(np.shape(y))

    for i in range(np.size(y,1)):
        smooth = np.convolve(y[:, i], np.ones((20,))/20, mode='same')
        h['sz'] = np.std(y[:, i]-smooth)
        inf_temp = []
        for t in np.arange(np.min(x), np.max(x), chunck_size):
            x_start = t - 20*h['l']
            x_end = t + chunck_size + 10*h['l']
            x_chunk = x[(x >= x_start) & (x < x_end)]
            y_chunk = y[(x >= x_start) & (x < x_end), i]
            h['sf'] = np.std(y_chunk)

            x_inf = x[(x >= t) & (x < (t+chunck_size))]

            avg = np.mean(y_chunk)
            y_chunk = y_chunk - avg

            K = kernelSE(x_chunk[:,np.newaxis], x_chunk[:,np.newaxis], h)
            K = K + h['sz']**2 * np.eye(np.size(x_chunk, 0))

            Ks = kernelSE(x_inf[:,np.newaxis], x_chunk[:,np.newaxis], h)

            alpha = np.linalg.solve(K, y_chunk)

            mu = Ks @ alpha

            inf_temp.append(mu + avg)

        out[:, i] = np.concatenate(inf_temp)        

    return out
    
def diffWithGP(x, y, chunck_size = 5):
    h = {}
    h['l'] = 5 * np.median(np.diff(x))

    out = np.zeros(np.shape(y))

    for i in range(np.size(y,1)):
        smooth = np.convolve(y[:, i], np.ones((20,))/20, mode='same')
        h['sz'] = np.std(y[:, i]-smooth)
        inf_temp = []
        for t in np.arange(np.min(x), np.max(x), chunck_size):
            x_start = t - 20*h['l']
            x_end = t + chunck_size + 10*h['l']
            x_chunk = x[(x >= x_start) & (x < x_end)]
            y_chunk = y[(x >= x_start) & (x < x_end), i]
            h['sf'] = np.std(y_chunk)

            x_inf = x[(x >= t) & (x < (t+chunck_size))]

            avg = np.mean(y_chunk)
            y_chunk = y_chunk - avg

            K = kernelSE(x_chunk[:,np.newaxis], x_chunk[:,np.newaxis], h)
            K = K + h['sz']**2 * np.eye(np.size(x_chunk, 0))

            Ks = kernelSEDiff(x_inf[:,np.newaxis], x_chunk[:,np.newaxis], h, axis=0)

            alpha = np.linalg.solve(K, y_chunk)

            mu = Ks @ alpha

            inf_temp.append(mu)

        out[:, i] = np.concatenate(inf_temp)        

    return out


def diff2WithGP(x, y, chunck_size = 5):
    h = {}
    h['l'] = 5 * np.mean(np.diff(x))

    out = np.zeros(np.shape(y))

    for i in range(np.size(y,1)):
        smooth = np.convolve(y[:, i], np.ones((10,))/10, mode='same')
        h['sz'] = np.std(y[:, i]-smooth)
        inf_temp = []
        for t in np.arange(np.min(x), np.max(x), chunck_size):
            x_start = t - 20*h['l']
            x_end = t + chunck_size + 10*h['l']
            x_chunk = x[(x >= x_start) & (x < x_end)]
            y_chunk = y[(x >= x_start) & (x < x_end), i]
            h['sf'] = np.std(y_chunk)

            x_inf = x[(x >= t) & (x < (t+chunck_size))]

            avg = np.mean(y_chunk)
            y_chunk = y_chunk - avg

            K = kernelSE(x_chunk[:,np.newaxis], x_chunk[:,np.newaxis], h)
            K = K + h['sz']**2 * np.eye(np.size(x_chunk, 0))

            Ks = kernelSEDiff2(x_inf[:,np.newaxis], x_chunk[:,np.newaxis], h, axis=0)

            alpha = np.linalg.solve(K, y_chunk)

            mu = Ks @ alpha

            inf_temp.append(mu)

        out[:, i] = np.concatenate(inf_temp)        

    return out

def addNPi(r, n):
    return (r/np.linalg.norm(r))*(2*n*np.pi + np.linalg.norm(r))

def fromQuatTrajToRotVec(q):
    r = np.zeros((np.size(q, 0), 3))
    r[0,:] = R.from_quat(q[0,:]).as_rotvec()
    nb_rev = 0
    for i in range(1, np.size(q, 0)):
        r_temp = R.from_quat(q[i,:]).as_rotvec()

        r0 = addNPi(r_temp, nb_rev)
        r1 = addNPi(r_temp, nb_rev+1)
        r_1 = addNPi(r_temp, nb_rev-1)

        d0 = np.linalg.norm(r0-r)
        d1 = np.linalg.norm(r1-r)
        d_1 = np.linalg.norm(r_1-r)

        if d0 < d1 and d0 < d_1:
            r[i,:] = r0
        elif d1 < d0 and d1 < d_1:
            r[i,:] = r1
            nb_rev += 1
        else:
            r[i,:] = r_1
            nb_rev -= 1
    return r




if __name__ == '__main__':
    # Get the configuration file path from the command line
    parser = argparse.ArgumentParser(description='Calibrate IMU to pose.')
    parser.add_argument('-c', '--config', dest='config', type=str, help='The configuration file. (default: config.yaml)', default='scripts/config.yaml', required=False)
    args = parser.parse_args()

    # Read YAML file
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)


    print("Reading data (python)")

    # Load the data
    bag = rosbag.Bag(config['bag_path'])



    #[imu_data, imu_time]  = getImuData(bag, config['imu_topic'])
    [gyr_data, gyr_time]  = getGyrData(bag, config['gyr_topic'])
    [acc_data, acc_time]  = getAccData(bag, config['acc_topic'])

    [pose_time, pose_data] = pose_utils.readPosesRosbag(config, as_quat=True)

    bag.close()

    # If there is a prior calibration, load it
    r = np.zeros(3)
    p = np.zeros(3)
    if ('init_pos_x' in config) and ('init_pos_y' in config) and ('init_pos_z' in config) and ('init_rot_x' in config) and ('init_rot_y' in config) and ('init_rot_z' in config):
        p[0] = config['init_pos_x']
        p[1] = config['init_pos_y']
        p[2] = config['init_pos_z']
        r[0] = config['init_rot_x']
        r[1] = config['init_rot_y']
        r[2] = config['init_rot_z']

    # Get acc and gyro frequency
    min_time = np.max([pose_time[0], acc_time[0], gyr_time[0]])
    max_time = np.min([pose_time[-1], acc_time[-1], gyr_time[-1]])

    # Convert the time into seconds with 0 as the first time stamp
    temp = np.empty(len(pose_time))
    for i in range(len(pose_time)):
        temp[i] = (pose_time[i]-min_time).to_sec()
    pose_time = temp
    temp = np.empty(len(acc_time))
    for i in range(len(acc_time)):
        temp[i] = (acc_time[i]-min_time).to_sec()
    acc_time = temp
    temp = np.empty(len(gyr_time))
    for i in range(len(gyr_time)):
        temp[i] = (gyr_time[i]-min_time).to_sec()
    gyr_time = temp

    max_time = (max_time - min_time).to_sec()

    # Crop data to the same time interval
    pose_data = pose_data[pose_time <= max_time, :]
    pose_time = pose_time[pose_time <= max_time]

    pose_freq = 1.0/np.median(np.diff(pose_time))
    acc_freq = 1.0/np.median(np.diff(acc_time))
    gyr_freq = 1.0/np.median(np.diff(gyr_time))

    margin_factor = 20.0
    margin = margin_factor/pose_freq
    acc_data = acc_data[(acc_time >= margin) & (acc_time <= (max_time - margin)), :]
    acc_time = acc_time[(acc_time >= margin) & (acc_time <= (max_time - margin))]
    gyr_data = gyr_data[(gyr_time >= margin) & (gyr_time <= (max_time - margin)), :]
    gyr_time = gyr_time[(gyr_time >= margin) & (gyr_time <= (max_time - margin))]

    # For each acc or gyr measurement, check if there are enough pose measurements in the temporal neighborhood
    acc_mask = np.zeros(len(acc_time), dtype=bool)
    for i in range(len(acc_time)):
        acc_mask[i] = np.sum((pose_time >= (acc_time[i] - margin)) & (pose_time <= (acc_time[i] + margin))) >= 2*margin_factor
    gyr_mask = np.zeros(len(gyr_time), dtype=bool)
    for i in range(len(gyr_time)):
        gyr_mask[i] = np.sum((pose_time >= (gyr_time[i] - margin)) & (pose_time <= (gyr_time[i] + margin))) >= 2*margin_factor
    
    acc_data = acc_data[acc_mask, :]
    acc_time = acc_time[acc_mask]
    gyr_data = gyr_data[gyr_mask, :]
    gyr_time = gyr_time[gyr_mask]


    # Create the weights
    acc_dt = np.diff(acc_time)
    gyr_dt = np.diff(gyr_time)

    pos_prior=p
    rot_prior=r
    gravity=config['gravity']
    acc_std=config['acc_std']
    gyr_std=config['gyr_std']
    acc_walk=config['acc_walk']
    gyr_walk=config['gyr_walk']

    g_vec = np.array([0, 0, -gravity])
    acc_weight = np.ones((len(acc_time)))/(acc_std*np.sqrt(1.0/acc_freq))
    gyr_weight = np.ones((len(gyr_time)))/(gyr_std*np.sqrt(1.0/gyr_freq))
    acc_walk_weight = (1.0/(acc_walk*np.sqrt(1.0/acc_freq)))*acc_dt
    gyr_walk_weight = (1.0/(gyr_walk*np.sqrt(1.0/gyr_freq)))*gyr_dt



    rot_vec = fromQuatTrajToRotVec(pose_data[:, 3:7])
    rot_vec = filterWithGP(pose_time, rot_vec)
    diff_rot_vec = diffWithGP(pose_time, rot_vec)

    pose_data_quat = pose_data.copy()
    pose_data = np.hstack((pose_data[:, 0:3], rot_vec))

    pose_ang_vel_t = pose_time.copy()
    pose_ang_vel = np.zeros((np.size(pose_data, 0), 3))

    for i in range(0, np.size(pose_data,0)):
        pose_ang_vel[i, :] = (rightJacobianSO3(rot_vec[i, :]) @ (diff_rot_vec[i, :].reshape(-1,1))).squeeze()


    world_ang_acc = diff2WithGP(pose_ang_vel_t, rot_vec)
    world_ang_vel = diffWithGP(pose_ang_vel_t, rot_vec)

    pose_acc = diff2WithGP(pose_time, pose_data[:, 0:3])

    for i in range(0, np.size(pose_acc, 0)):
        rot_mat = R.from_quat(pose_data_quat[i, 3:7]).as_matrix()
        pose_acc[i, :] = rot_mat.T @ pose_acc[i, :]


    # Write to file and call the C++ code

    # Check if ../temp exists and create it if not
    if not os.path.exists('temp'):
        os.makedirs('temp')

    # Write the data to file
    np.savetxt('temp/acc_data.csv', np.hstack((acc_time.reshape(-1, 1), acc_data)), delimiter=',')
    np.savetxt('temp/gyr_data.csv', np.hstack((gyr_time.reshape(-1, 1), gyr_data)), delimiter=',')
    np.savetxt('temp/acc_bias_weights.csv', acc_walk_weight, delimiter=',')
    np.savetxt('temp/gyr_bias_weights.csv', gyr_walk_weight, delimiter=',')
    np.savetxt('temp/acc_weights.csv', acc_weight, delimiter=',')
    np.savetxt('temp/gyr_weights.csv', gyr_weight, delimiter=',')
    np.savetxt('temp/rot_prior.csv', rot_prior, delimiter=',')
    np.savetxt('temp/pos_prior.csv', pos_prior, delimiter=',')
    np.savetxt('temp/pose.csv', np.hstack( (pose_time.reshape(-1, 1) , pose_data) ), delimiter=',')
    np.savetxt('temp/gravity.csv', g_vec, delimiter=',')


    # Call the C++ code
    os.system('cd build && ./imu_pose_calib')

    # Read the results
    r_c = np.loadtxt('temp/rot_calib.csv', delimiter=',')
    p_c = np.loadtxt('temp/pos_calib.csv', delimiter=',')
    dt_dt_drift = np.loadtxt('temp/dt.csv', delimiter=',')
    r_0 = np.loadtxt('temp/rot_0.csv', delimiter=',')
    gyr_bias = np.loadtxt('temp/gyr_biases.csv', delimiter=',')
    acc_bias = np.loadtxt('temp/acc_biases.csv', delimiter=',')


    dt = dt_dt_drift

    # Correct the IMU time stamps
    #imu_time = imu_time + dt
    acc_time = acc_time + dt
    gyr_time = gyr_time + dt


    # Query (interpolate) the pose data at the IMU time stamps
    acc_pose = np.zeros((np.size(acc_data, 0), 7))
    for i in range(3):
        acc_pose[:, i] = np.interp(acc_time, pose_time, pose_data[:, i])
    acc_pose[:, 3:7] = Slerp(pose_time, R.from_quat(pose_data_quat[:, 3:7]))(acc_time).as_quat()

    gyr_pose = np.zeros((np.size(gyr_data, 0), 7))
    for i in range(3):
        gyr_pose[:, i] = np.interp(gyr_time, pose_time, pose_data[:, i])
    gyr_pose[:, 3:7] = Slerp(pose_time, R.from_quat(pose_data_quat[:, 3:7]))(gyr_time).as_quat()

    # Apply the rotation and translation corrections
    R_c = R.from_rotvec(r_c).as_matrix()
    for i in range(np.size(acc_data, 0)):
        R_i = R.from_quat(acc_pose[i, 3:7]).as_matrix()
        acc_pose[i, 0:3] = (acc_pose[i, 0:3] + (R_i@p_c))
        R_i = R_i @ R_c
        acc_pose[i, 3:7] = R.from_matrix(R_i).as_quat()
    
    for i in range(np.size(gyr_data, 0)):
        R_i = R.from_quat(gyr_pose[i, 3:7]).as_matrix()
        gyr_pose[i, 0:3] = (gyr_pose[i, 0:3] + (R_i@p_c))
        R_i = R_i @ R_c
        gyr_pose[i, 3:7] = R.from_matrix(R_i).as_quat()

    # Combine the acc and gyr data
    imu_time = np.concatenate((acc_time, gyr_time))
    imu_pose = np.concatenate((acc_pose, gyr_pose))
    # Sort the data by time
    sort_idx = np.argsort(imu_time)
    imu_time = imu_time[sort_idx]
    imu_pose = imu_pose[sort_idx, :]




    # Get the name of the bag without the path and the extension
    bag_name = os.path.splitext(os.path.basename(config['bag_path']))[0]
    # Get the path of the bag without the name
    bag_path = os.path.dirname(config['bag_path'])

    # Save the calibration results in a csv file
    write_path = bag_path + '/' + bag_name + '_imu_pose_calib.csv'

    np.savetxt(write_path, np.hstack((p_c.reshape(1,-1), r_c.reshape(1,-1), dt.reshape(1,-1))), delimiter=',', fmt='%.9f', header='pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, delta_t (Calibration so that the points projection is as x_marker = Exp(rot)*x_imu + pos)')





    # Rotate the IMU gyro measurements
    temp_gyr = np.zeros((np.size(gyr_data, 0), 3))
    temp_gyr_prior = np.zeros((np.size(gyr_data, 0), 3))
    R_c = R.from_rotvec(r_c).as_matrix()
    R_prior = R.from_rotvec(rot_prior).as_matrix()
    for i in range(np.size(gyr_data, 0)):
        temp_gyr[i, :] = (R_c @ (gyr_data[i, :]-gyr_bias[i,:])[:,np.newaxis]).T
        temp_gyr_prior[i, :] = (R_prior @ (gyr_data[i, :])[:,np.newaxis]).T


    
    mask = (pose_ang_vel_t > np.min(pose_ang_vel_t) + 0.5*margin) & (pose_ang_vel_t < np.max(pose_ang_vel_t) - 0.5*margin)
    # Plot the results in 3 different subplots
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(gyr_time, temp_gyr_prior[:, 0], label='prior')
    ax[0].plot(pose_ang_vel_t[mask], pose_ang_vel[mask, 0], label='vicon')
    ax[0].plot(gyr_time, temp_gyr[:, 0], label='imu')
    ax[0].set_title('Gyro x')
    ax[0].legend()
    ax[1].plot(gyr_time, temp_gyr_prior[:, 1], label='prior')
    ax[1].plot(pose_ang_vel_t[mask], pose_ang_vel[mask, 1], label='vicon')
    ax[1].plot(gyr_time, temp_gyr[:, 1], label='imu')
    ax[1].set_title('Gyro y')
    ax[1].legend()
    ax[2].plot(gyr_time, temp_gyr_prior[:, 2], label='prior')
    ax[2].plot(pose_ang_vel_t[mask], pose_ang_vel[mask, 2], label='vicon')
    ax[2].plot(gyr_time, temp_gyr[:, 2], label='imu')
    ax[2].set_title('Gyro z')
    ax[2].legend()
    fig.suptitle('Gyro measurements vs Vicon-inferred angular velocity')
    
    plt.show()

    # Get the prior on R_0
    g_unit = -g_vec/np.linalg.norm(g_vec)
    avg_acc = np.mean(acc_data[0:10, 0:3], axis=0)
    avg_acc_unit = avg_acc/np.linalg.norm(avg_acc)
    imu_r_0 = np.cross(g_unit, avg_acc_unit)
    imu_r_0 = imu_r_0/np.linalg.norm(imu_r_0)
    imu_r_0 = np.arccos(np.dot(g_unit, avg_acc_unit))*imu_r_0
    imu_r_0 = imu_r_0 if np.linalg.norm(R.from_rotvec(imu_r_0).as_matrix()@avg_acc_unit - g_unit) < np.linalg.norm(R.from_rotvec(-imu_r_0).as_matrix()@avg_acc_unit - g_unit) else -imu_r_0
    
    R_0_prior = R.from_rotvec(imu_r_0).as_matrix()@ (R.from_rotvec(rot_prior).as_matrix().T) @ (R.from_quat(pose_data_quat[0, 3:7]).as_matrix().T)

    pose_reading = np.zeros((np.size(world_ang_acc, 0), 3))
    pose_reading_prior = np.zeros((np.size(world_ang_acc, 0), 3))
    R_0 = R.from_rotvec(r_0).as_matrix()
    for i in range(0, np.size(world_ang_acc, 0)):
        rot_mat = R.from_quat(pose_data_quat[i, 3:7]).as_matrix()

        pose_reading[i, :] = R_c.T@(pose_acc[i, :] + rot_mat.T@np.cross(world_ang_acc[i,:], p_c) + rot_mat.T@np.cross(world_ang_vel[i,:],np.cross(world_ang_vel[i,:], p_c)) - rot_mat.T @ R_0.T @ g_vec)
        pose_reading_prior[i, :] = R_prior.T@(pose_acc[i, :] + rot_mat.T@np.cross(world_ang_acc[i,:], pos_prior) + rot_mat.T@np.cross(world_ang_vel[i,:],np.cross(world_ang_vel[i,:], pos_prior)) - rot_mat.T @ R_0_prior.T @ g_vec)



    fig, axs = plt.subplots(3, 1)
    axs[0].plot(pose_ang_vel_t[mask], pose_reading_prior[mask, 0], label='prior')
    axs[0].plot(pose_ang_vel_t[mask], pose_reading[mask, 0], label='vicon')
    axs[0].plot(acc_time, acc_data[:, 0] - acc_bias[:,0], label='imu')
    axs[0].set_title('Acc x')
    axs[0].legend()
    axs[1].plot(pose_ang_vel_t[mask], pose_reading_prior[mask, 1], label='prior')
    axs[1].plot(pose_ang_vel_t[mask], pose_reading[mask, 1], label='vicon')
    axs[1].plot(acc_time, acc_data[:, 1] - acc_bias[:,1], label='imu')
    axs[1].set_title('Acc y')
    axs[1].legend()
    axs[2].plot(pose_ang_vel_t[mask], pose_reading_prior[mask, 2], label='prior')
    axs[2].plot(pose_ang_vel_t[mask], pose_reading[mask, 2], label='vicon')
    axs[2].plot(acc_time, acc_data[:, 2] - acc_bias[:,2], label='imu')
    axs[2].set_title('Acc z')
    axs[2].legend()
    fig.suptitle('Accelerometer measurements vs Vicon-inferred acceleration')
    plt.show()


    # Remove files in the temp folder
    for file in os.listdir('temp'):
        os.remove('temp/' + file)
    os.rmdir('temp')


    print("Done")



