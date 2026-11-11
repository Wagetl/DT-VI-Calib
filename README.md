## DT-VI-Calib
Discrete-Time Visual-Inertial Calibration

## Description
This repo contains the code for Ultrafast Target-based IMU-Camera Spatial-Temporal Calibration. The algorithm is designed with Discrete-Time State Representation.

## Reference
Song, Junlin, Antoine Richard, and Miguel Olivares-Mendez. ["**Unleashing the Power of Discrete-Time State Representation: Ultrafast Target-based IMU-Camera Spatial-Temporal Calibration**."](https://arxiv.org/abs/2509.12846) To be appeared at 2026 IEEE International Conference on Robotics and Automation (ICRA).

```
@article{song2025unleashing,
  title={Unleashing the Power of Discrete-Time State Representation: Ultrafast Target-based IMU-Camera Spatial-Temporal Calibration},
  author={Song, Junlin and Richard, Antoine and Olivares-Mendez, Miguel},
  journal={arXiv preprint arXiv:2509.12846},
  year={2025}
}

@phdthesis{song2025calibration,
  title={Calibration for Vision-based Multi-sensor System},
  author={Song, Junlin},
  year={2025},
  school={Space Robotics (SpaceR) Research Group, Interdisciplinary Centre for Security, Reliability and Trust (SnT), University of Luxembourg},
  note ={\url{https://hdl.handle.net/10993/67005}}
}
```

## Installation
### Docker on macOS
Apple Silicon Mac users can follow [Docker setup and calibration commands](doc/DockerMac.md).

### Tested for Ubuntu 20.04
```
git clone https://github.com/JunlinSong/DT-VI-Calib
cd DT-VI-Calib
./scripts/install_deps.sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=Yes
make -j8
```

## How to use?
1, download a calibration file: dataset-calib-imu1_512_16.bag
```
https://cdn2.vision.in.tum.de/tumvi/calibrated/512_16/
```

2, calibrate stereo extrinsics and intrinsics
```
./basalt_calibrate --dataset-path /home/junlin/Downloads/dataset-calib-imu1_512_16.bag --dataset-type bag --aprilgrid /home/junlin/Calib/DT-VI-Calib/data/aprilgrid_6x6.json --result-path /home/junlin/Calib/calib_result/ --cam-types ds ds
```

3, perform IMU-Camera Spatial-Temporal Calibration
```
./basalt_calibrate_imu --dataset-path /home/junlin/Downloads/dataset-calib-imu1_512_16.bag --dataset-type bag --aprilgrid /home/junlin/Calib/DT-VI-Calib/data/aprilgrid_6x6.json --result-path /home/junlin/Calib/calib_result/ --gyro-noise-std 0.000282 --accel-noise-std 0.016 --skip-images 1
```

For --aprilgrid, one can find explanation from [kalibr](https://github.com/ethz-asl/kalibr/wiki/calibration-targets).  
For --skip-images, one can choose 1, 2, 4, referring to 20hz, 10hz, 5hz for image frequency, if the original image frequency is 20hz.  
To try different time offset (unit: ms), one can modify [td_set] in calibrate_imu.cpp.  
Suggestions on potential improvements of this repo can be found in Sec. 6.8 of ["**Calibration for Vision-based Multi-sensor System**."](https://hdl.handle.net/10993/67005)

## Acknowledgment
This repo is based on [basalt](https://gitlab.com/VladyslavUsenko/basalt).
