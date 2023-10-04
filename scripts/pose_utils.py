from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import numpy as np
import rospy
import tf

# Poses are stored as N x 6 numpy arrays, where each row is a pose (x, y, z, rx, ry, rz) in meters and radians (rx, ry, rz are the components of the rotation vector)



# Read poses from a rosbag
# config = {
#     'bag_path' (required): path to the rosbag
#     'pose_topic' or both 'tf_parent' and 'tf_child' (required): topic name if using a TransformStamped topic or tf frames of base and end of the chain if using tf
# }
def readPosesRosbag(config, static_only=False, as_quat=False):
    import rosbag

    bag = rosbag.Bag(config['bag_path'])

    vicon_timestamps = []
    vicon_poses = []
    if 'pose_topic' in config:
        for topic, msg, t in bag.read_messages(topics=[config['pose_topic']]):
            if as_quat:
                r_vec = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]
            else:
                r_vec = R.from_quat([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]).as_rotvec()
            pos = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
            vicon_poses.append(np.concatenate((pos, r_vec)))
            vicon_timestamps.append(msg.header.stamp)

    else:
        tf_tree = tf.Transformer(True, rospy.Duration(5000000))
        if static_only:
            for topic, msg, tt in bag.read_messages(topics=['/tf_static']):
                for transform in msg.transforms:
                    temp_t = transform
                    temp_t.header.stamp = msg.transforms[0].header.stamp
                    tf_tree.setTransform(temp_t)
                if tf_tree.canTransform(config['tf_parent'], config['tf_child'], temp_t.header.stamp):
                    T = tf_tree.lookupTransform(config['tf_parent'], config['tf_child'], temp_t.header.stamp)
                    if as_quat:
                        r_vec = T[1]
                    else:
                        r_vec = R.from_quat(T[1]).as_rotvec()
                    pos = np.array(T[0])
                    vicon_poses.append(np.concatenate((pos, r_vec)))
                    vicon_timestamps.append(transform.header.stamp)
        else:
            # Get the pose between the config['tf_parent'] and config['tf_child'] frames
            for topic, msg, t in bag.read_messages(topics=['/tf']):
                tf_tree.clear()
                for ttopic, mmsg, tt in bag.read_messages(topics=['/tf_static']):
                    for ttransform in mmsg.transforms:
                        temp_tt = ttransform
                        temp_tt.header.stamp = msg.transforms[0].header.stamp
                        tf_tree.setTransform(temp_tt)
                for transform in msg.transforms:
                    temp_transform = transform
                    temp_transform.header.stamp = msg.transforms[0].header.stamp
                    tf_tree.setTransform(temp_transform)
                    if tf_tree.canTransform(config['tf_parent'], config['tf_child'], temp_transform.header.stamp):
                        T = tf_tree.lookupTransform(config['tf_parent'], config['tf_child'], temp_transform.header.stamp)
                        if as_quat:
                            r_vec = T[1]
                        else:
                            r_vec = R.from_quat(T[1]).as_rotvec()
                        pos = np.array(T[0])
                        vicon_poses.append(np.concatenate((pos, r_vec)))
                        vicon_timestamps.append(transform.header.stamp)


    vicon_poses = np.array(vicon_poses)

    bag.close()

    return vicon_timestamps, vicon_poses


# Interpolate pose at query timestamp using linear interpolation for position and slerp for orientation (data need to be sorted by timestamp)
# t_poses: timestamps of the poses (N numpy array float)
# poses: poses (N x 6 numpy array)
# t_query: timestamp of the query (float)
def interpolatePose(t_poses, poses, t_query):
    temp_index = np.argmin(np.abs(t_poses - t_query))
    if t_poses[temp_index] > t_query:
        temp_index -= 1
    if temp_index < 0:
        temp_index = 0
    if temp_index >= len(t_poses) - 1:
        temp_index = len(t_poses) - 2

    temp_pose = poses[temp_index,:]
    temp_pose_next = poses[temp_index+1,:]

    pos_out = temp_pose[:3] + (temp_pose_next[:3] - temp_pose[:3]) * (t_query - t_poses[temp_index]) / (t_poses[temp_index+1] - t_poses[temp_index])


    slerp = Slerp([0,1], R.from_rotvec(poses[[temp_index, temp_index+1],3:]))
    quat_out = slerp((t_query - t_poses[temp_index]) / (t_poses[temp_index+1] - t_poses[temp_index])).as_quat()

    pose_out = np.concatenate((pos_out, R.from_quat(quat_out).as_rotvec()))

    return pose_out

def prToT(pose):
    T = np.eye(4)
    T[:3,:3] = R.from_rotvec(pose[3:6]).as_matrix()
    T[:3,3] = pose[:3]
    return T