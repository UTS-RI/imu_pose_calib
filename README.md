# Pose/Vicon-IMU calibration

Author: le.gentil.cedric@gmail.com

This repository provides a python/C++ toolbox for IMU-to-pose (typically Vicon/motion-capture systems, or robot arm) extrinsic and temporal calibration.
It uses Gaussian Processes with linear operators to deal with differentiation of the positions data.

TODO: Put paper link and bibtex when/if published.

## Warning

This is research code that is not optimised neither for performance nor maintainability.

## Install

Clone the repository
```
git clone git@github.com:clegenti/imu_vicon_calib.git
```

Install the C++ dependencies (you'll also need the standard cmake and other compilation tools not listed here)
```
sudo apt-get install libboost-program-options-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libceres-dev
```

__If there is something missing please let me know__ (if `libboost-program-options-dev` is an issue maybe try with `libboost-all-dev`)


Create a build folder and compile
```
cd imu_vicon_calib
mkdir build
cd build
cmake ../
make -j4
cd ..
```

For running this package you also need to install some python dependencies. The required packages are shown in `scripts/requirements.txt`.


## Run

You need to update the configuration file `scripts/config.yaml` with the rosbag path, topic names, writing path, initial guess of the extrinsic transformation, etc. The rotation initial guess is given with the rotation vector representation (often called axis-angle).

Note that the IMU topic must be of type `sensor_msgs/Imu` and the poses `geometry_msgs/TransformStamped` or from the _tf_ tree (I am not an expert with the \tf topic with python, I made it work but it is probably not very elegant. Please let me know if you have a better way :) ).


Then run
```
python scripts/calibrate.py -c scripts/config.yaml
```

There will be two sets of plots that will show-up. The calibration should be good if the IMU and Vicon/pose measurements curves overlap nicely (should be the orange and green lines).
The program will terminate when both windows are closed.

With the provided configuration file and data sample, the script should write down the IMU-to-pose extrinsics and time-shift in `data/imu_vicon_calib.csv`.
The format of the former is `pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, delta_t`.
