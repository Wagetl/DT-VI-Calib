/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <basalt/optimization/accumulator.h>
#include <basalt/optimization/spline_linearize.h>

#include <basalt/calibration/calibration.hpp>

#include <basalt/camera/unified_camera.hpp>

#include <basalt/image/image.h>

#include <basalt/serialization/headers_serialization.h>

#include <tbb/parallel_reduce.h>

#include <chrono>

#include <basalt/io/colors.h>

namespace basalt {

template <int N, typename Scalar>
class SplineOptimization {
 public:
  typedef LinearizeSplineOpt<N, Scalar, SparseHashAccumulator<Scalar>>
      LinearizeT;

  typedef typename LinearizeT::SE3 SE3;
  typedef typename LinearizeT::Vector2 Vector2;
  typedef typename LinearizeT::Vector3 Vector3;
  typedef typename LinearizeT::Vector4 Vector4;
  typedef typename LinearizeT::VectorX VectorX;
  typedef typename LinearizeT::Matrix24 Matrix24;

  static const int POSE_SIZE = LinearizeT::POSE_SIZE;
  static const int POS_SIZE = LinearizeT::POS_SIZE;
  static const int POS_OFFSET = LinearizeT::POS_OFFSET;
  static const int ROT_SIZE = LinearizeT::ROT_SIZE;
  static const int ROT_OFFSET = LinearizeT::ROT_OFFSET;
  static const int ACCEL_BIAS_SIZE = LinearizeT::ACCEL_BIAS_SIZE;
  static const int GYRO_BIAS_SIZE = LinearizeT::GYRO_BIAS_SIZE;
  static const int G_SIZE = LinearizeT::G_SIZE;

  static const int ACCEL_BIAS_OFFSET = LinearizeT::ACCEL_BIAS_OFFSET;
  static const int GYRO_BIAS_OFFSET = LinearizeT::GYRO_BIAS_OFFSET;
  static const int G_OFFSET = LinearizeT::G_OFFSET;

  const Scalar pose_var;

  Scalar pose_var_inv;

  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  typedef Se3Spline<N, Scalar> SplineT;

  SplineOptimization(int64_t dt_ns = 1e7, double init_lambda = 1e-12)
      : pose_var(1e-4),
        mocap_initialized(false),
        lambda(init_lambda),
        min_lambda(1e-18),
        max_lambda(100),
        lambda_vee(2),
        spline(dt_ns),
        dt_ns(dt_ns) {
    pose_var_inv = 1.0 / pose_var;

    // reasonable default values
    // calib.accelerometer_noise_var = 2e-2;
    // calib.gyroscope_noise_var = 1e-4;

    g.setZero();

    min_time_us = std::numeric_limits<int64_t>::max();
    max_time_us = std::numeric_limits<int64_t>::min();
  }

  int64_t getCamTimeOffsetNs() const { return calib->cam_time_offset_ns; }
  int64_t getMocapTimeOffsetNs() const {
    return mocap_calib->mocap_time_offset_ns;
  }

  const SE3& getCamT_i_c(size_t i) const { return calib->T_i_c[i]; }
  SE3& getCamT_i_c(size_t i) { return calib->T_i_c[i]; }

  VectorX getIntrinsics(size_t i) const {
    return calib->intrinsics[i].getParam();
  }

  const CalibAccelBias<Scalar>& getAccelBias() const {
    return calib->calib_accel_bias;
  }
  const CalibGyroBias<Scalar>& getGyroBias() const {
    return calib->calib_gyro_bias;
  }

  void resetCalib(size_t num_cams, const std::vector<std::string>& cam_types) {
    BASALT_ASSERT(cam_types.size() == num_cams);

    calib.reset(new Calibration<Scalar>);

    calib->intrinsics.resize(num_cams);
    calib->T_i_c.resize(num_cams);

    mocap_calib.reset(new MocapCalibration<Scalar>);
  }

  void resetMocapCalib() { mocap_calib.reset(new MocapCalibration<Scalar>); }

  void loadCalib(const std::string& p) {
    std::string path = p + "calibration.json";

    std::ifstream is(path);

    if (is.good()) {
      cereal::JSONInputArchive archive(is);
      calib.reset(new Calibration<Scalar>);
      archive(*calib);
      std::cout << "Loaded calibration from: " << path << std::endl;
    } else {
      std::cerr << "No calibration found. Run camera calibration first!!!"
                << std::endl;
    }
  }

  void saveCalib(const std::string& path) const {
    if (calib) {
      std::ofstream os(path + "calibration.json");
      cereal::JSONOutputArchive archive(os);

      archive(*calib);
    }
  }

  void saveMocapCalib(const std::string& path,
                      int64_t mocap_to_imu_offset_ns = 0) const {
    if (calib) {
      std::ofstream os(path + "mocap_calibration.json");
      cereal::JSONOutputArchive archive(os);

      mocap_calib->mocap_to_imu_offset_ns = mocap_to_imu_offset_ns;

      archive(*mocap_calib);
    }
  }

  bool calibInitialized() const { return calib != nullptr; }

  bool initialized() const { return spline.numKnots() > 0; }

  void initSpline(const SE3& pose, int num_knots) {
    spline.setKnots(pose, num_knots);
  }

  void initSpline(const SplineT& other) {
    spline.setKnots(other);

    // std::cerr << "spline.numKnots() " << spline.numKnots() << std::endl;
    // std::cerr << "other.numKnots() " << other.numKnots() << std::endl;

    size_t num_knots = spline.numKnots();

    for (size_t i = 0; i < num_knots; i++) {
      Eigen::Vector3d rot_rand_inc = Eigen::Vector3d::Random() / 20;
      Eigen::Vector3d trans_rand_inc = Eigen::Vector3d::Random();

      // std::cerr << "rot_rand_inc " << rot_rand_inc.transpose() << std::endl;
      // std::cerr << "trans_rand_inc " << trans_rand_inc.transpose() <<
      // std::endl;

      spline.getKnotSO3(i) *= Sophus::SO3d::exp(rot_rand_inc);
      spline.getKnotPos(i) += trans_rand_inc;
    }
  }

  const SplineT& getSpline() const { return spline; }

  Vector3 getG() const { return g; }

  void setG(const Vector3& g_a) { 
    g = g_a;
    g = (9.81 / g.norm()) * g;
    get_g_sphere(g, g_sphere);
  }

  void set_opt_option(bool opt_intrinsics, bool opt_cam_time_offset, bool opt_imu_scale)
  {
    ccd.opt_intrinsics = opt_intrinsics;
    ccd.opt_cam_time_offset = opt_cam_time_offset;
    ccd.opt_imu_scale = opt_imu_scale;
  }

  // const Calibration& getCalib() const { return calib; }
  // void setCalib(const Calibration& c) { calib = c; }

  SE3 getT_w_moc() const { return mocap_calib->T_moc_w.inverse(); }
  void setT_w_moc(const SE3& val) { mocap_calib->T_moc_w = val.inverse(); }

  SE3 getT_mark_i() const { return mocap_calib->T_i_mark.inverse(); }
  void setT_mark_i(const SE3& val) { mocap_calib->T_i_mark = val.inverse(); }

  Eigen::Vector3d getTransAccelWorld(int64_t t_ns) const {
    return spline.transAccelWorld(t_ns);
  }

  Eigen::Vector3d getRotVelBody(int64_t t_ns) const {
    return spline.rotVelBody(t_ns);
  }

  SE3 getT_w_i(int64_t t_ns) { return spline.pose(t_ns); }

  void setAprilgridCorners3d(const Eigen::aligned_vector<Eigen::Vector4d>& v) {
    aprilgrid_corner_pos_3d = v;
  }

  void addPoseMeasurement(int64_t t_ns, const SE3& pose) {
    min_time_us = std::min(min_time_us, t_ns);
    max_time_us = std::max(max_time_us, t_ns);

    pose_measurements.emplace_back();
    pose_measurements.back().timestamp_ns = t_ns;
    pose_measurements.back().data = pose;

    timestam_to_pose[t_ns] = pose;
  }

  void addMocapMeasurement(int64_t t_ns, const SE3& pose) {
    mocap_measurements.emplace_back();
    mocap_measurements.back().timestamp_ns = t_ns;
    mocap_measurements.back().data = pose;
  }

  void addAccelMeasurement(int64_t t_ns, const Vector3& meas) {
    min_time_us = std::min(min_time_us, t_ns);
    max_time_us = std::max(max_time_us, t_ns);

    accel_measurements.emplace_back();
    accel_measurements.back().timestamp_ns = t_ns;
    accel_measurements.back().data = meas;
  }

  void addGyroMeasurement(int64_t t_ns, const Vector3& meas) {
    min_time_us = std::min(min_time_us, t_ns);
    max_time_us = std::max(max_time_us, t_ns);

    gyro_measurements.emplace_back();
    gyro_measurements.back().timestamp_ns = t_ns;
    gyro_measurements.back().data = meas;
  }

  void addAprilgridMeasurement(
      int64_t t_ns, int cam_id,
      const Eigen::aligned_vector<Eigen::Vector2d>& corners_pos,
      const std::vector<int>& corner_id) {
    min_time_us = std::min(min_time_us, t_ns);
    max_time_us = std::max(max_time_us, t_ns);

    aprilgrid_corners_measurements.emplace_back();
    aprilgrid_corners_measurements.back().timestamp_ns = t_ns;
    aprilgrid_corners_measurements.back().cam_id = cam_id;
    aprilgrid_corners_measurements.back().corner_pos = corners_pos;
    aprilgrid_corners_measurements.back().corner_id = corner_id;
  }

  Scalar getMinTime() const { return min_time_us * 1e-9; }
  Scalar getMaxTime() const { return max_time_us * 1e-9; }

  int64_t getMinTimeNs() const { return min_time_us; }
  int64_t getMaxTimeNs() const { return max_time_us; }

  // ImuData<double>::Ptr popFromImuDataQueue() {
  //   ImuData<double>::Ptr data;
  //   imu_data_queue.pop(data);
  //   return data;
  // }

  void init() {
    // int64_t time_interval_us = max_time_us - min_time_us;

    // if (spline.numKnots() == 0) {
    //   spline.setStartTimeNs(min_time_us);
    //   spline.setKnots(pose_measurements.front().data,
    //                   time_interval_us / dt_ns + N + 1);
    // }

    // Vector3 gyro_bias;
    // Matrix3 gyro_scale;
    // calib->calib_gyro_bias.getBiasAndScale(gyro_bias, gyro_scale);

    // Vector3 accel_bias;
    // Matrix3 accel_scale;
    // calib->calib_accel_bias.getBiasAndScale(accel_bias, accel_scale);

    timestam_to_frame_states.clear();
    imu_measurements.clear();

    const Eigen::Matrix<double, 3, 1> accel_cov = calib->dicrete_time_accel_noise_std().array().square();
    const Eigen::Matrix<double, 3, 1> gyro_cov = calib->dicrete_time_gyro_noise_std().array().square();

    bool first_img = true;
    int64_t prev_frame_t = -1;
    Sophus::SE3d prev_frame_pose;
    bool first_pre_imu = true;
    size_t start_imu_idx = 0;
    for (const auto &kv : timestam_to_pose) {
      if (first_img)
      {
        prev_frame_t = kv.first;
        prev_frame_pose = kv.second;
        first_img = false;
      }
      else
      {
        int64_t curr_frame_t = kv.first;
        Sophus::SE3d curr_frame_pose = kv.second;
        double dt = (curr_frame_t - prev_frame_t) * 1e-9;
        Eigen::Vector3d vel_w_i = (curr_frame_pose.translation() - prev_frame_pose.translation()) / dt;

        ///////////////////////////////////////////////////////////////////////
        IntegratedImuMeasurement<double>::Ptr meas;
        meas.reset(new IntegratedImuMeasurement<double>(
          prev_frame_t, common_bg, common_ba));

        BASALT_ASSERT_MSG(prev_frame_t < curr_frame_t,
                          "duplicate frame timestamps?! zero time delta leads "
                          "to invalid IMU integration.");

        // ImuData<double>::Ptr data = popFromImuDataQueue();
        // BASALT_ASSERT_MSG(data, "first IMU measurment is nullptr");

        // // data->accel = calib->calib_accel_bias.getCalibrated(data->accel);
        // // data->gyro = calib->calib_gyro_bias.getCalibrated(data->gyro);

        // while (data->t_ns <= prev_frame_t) {
        //   data = popFromImuDataQueue();
        //   if (!data) break;
        //   // data->accel = calib->calib_accel_bias.getCalibrated(data->accel);
        //   // data->gyro = calib->calib_gyro_bias.getCalibrated(data->gyro);
        // }

        // while (data->t_ns <= curr_frame_t) {
        //   meas->integrate(*data, accel_cov, gyro_cov);
        //   data = popFromImuDataQueue();
        //   if (!data) break;
        //   // data->accel = calib->calib_accel_bias.getCalibrated(data->accel);
        //   // data->gyro = calib->calib_gyro_bias.getCalibrated(data->gyro);
        // }

        get_imu_buff(start_imu_idx, prev_frame_t, curr_frame_t, meas);
        meas->reintegrate(accel_cov, gyro_cov);

        if (first_pre_imu)
        {
          PoseVelState<double> prev_state(prev_frame_t, prev_frame_pose, vel_w_i);
          prev_state.angular_vel = meas->imu_buff[0].gyro - common_bg;
          timestam_to_frame_states[prev_frame_t] = prev_state;
          first_pre_imu = false;
        }
        PoseVelState<double> curr_state(curr_frame_t, curr_frame_pose, vel_w_i);
        curr_state.angular_vel = meas->imu_buff.back().gyro - common_bg;
        timestam_to_frame_states[curr_frame_t] = curr_state;

        imu_measurements.push_back(meas);
        
        /////////////////////////////////////////////////////////////////////
        prev_frame_t = curr_frame_t;
        prev_frame_pose = curr_frame_pose;
      }
    }

    recompute_size();

    //    std::cout << "spline.minTimeNs() " << spline.minTimeNs() << std::endl;
    //    std::cout << "spline.maxTimeNs() " << spline.maxTimeNs() << std::endl;

    while (!mocap_measurements.empty() &&
           mocap_measurements.front().timestamp_ns <=
               spline.minTimeNs() + spline.getDtNs())
      mocap_measurements.pop_front();

    while (!mocap_measurements.empty() &&
           mocap_measurements.back().timestamp_ns >=
               spline.maxTimeNs() - spline.getDtNs())
      mocap_measurements.pop_back();

    ccd.calibration = calib.get();
    ccd.mocap_calibration = mocap_calib.get();
    ccd.aprilgrid_corner_pos_3d = &aprilgrid_corner_pos_3d;
    ccd.common_ba = &common_ba;
    ccd.common_bg = &common_bg;
    ccd.g = &g;
    ccd.offset_poses = &offset_poses;
    ccd.offset_intrinsics = &offset_cam_intrinsics;
    ccd.offset_T_i_c = &offset_T_i_c;
    ccd.bias_block_offset = bias_block_offset;
    // ccd.mocap_block_offset = mocap_block_offset;

    ccd.opt_g = true;

    // ccd.pose_var_inv = pose_var_inv;
    // ccd.gyro_var_inv =
    //     calib->dicrete_time_gyro_noise_std().array().square().inverse();
    // ccd.accel_var_inv =
    //     calib->dicrete_time_accel_noise_std().array().square().inverse();
    // ccd.mocap_var_inv = pose_var_inv;

    ccd.gyro_var = gyro_cov;
    ccd.accel_var = accel_cov;
  }

  void get_imu_buff(size_t& start_imu_idx, const int64_t prev_frame_t, const int64_t curr_frame_t, IntegratedImuMeasurement<double>::Ptr meas)
  {
    ImuData<double>::Ptr data = imu_data_queue[start_imu_idx];
    for (size_t i = start_imu_idx; i < imu_data_queue.size(); i++)
    {
      data = imu_data_queue[i];
      if (data->t_ns > prev_frame_t)
      {
        start_imu_idx = i;
        break;
      }
    }

    data = imu_data_queue[start_imu_idx - 1];
    if (data->t_ns == prev_frame_t)
    {
      meas->imu_buff.push_back(*data);
    }
    else if (data->t_ns < prev_frame_t)
    {
      ImuData<double> left_imu = *data;
      ImuData<double> right_imu = *imu_data_queue[start_imu_idx];

      ImuData<double> interpolated_imu;
      interpolated_imu.t_ns = prev_frame_t;
      Scalar interpolation_lamda = (prev_frame_t - left_imu.t_ns) * 1.0 / (right_imu.t_ns - left_imu.t_ns);
      interpolated_imu.accel = left_imu.accel + interpolation_lamda * (right_imu.accel - left_imu.accel);
      interpolated_imu.gyro = left_imu.gyro + interpolation_lamda * (right_imu.gyro - left_imu.gyro);

      meas->imu_buff.push_back(interpolated_imu);
    }
    else
    {
      printf("imu0 timestamp err!\n");
      getchar();
    }

    for (size_t i = start_imu_idx; i < imu_data_queue.size(); i++)
    {
      data = imu_data_queue[i];
      if (data->t_ns > prev_frame_t && data->t_ns <= curr_frame_t)
      {
        meas->imu_buff.push_back(*data);
      }

      if (data->t_ns > curr_frame_t)
      {
        start_imu_idx = i;
        break;
      }
    }

    data = imu_data_queue[start_imu_idx - 1];
    if (data->t_ns == curr_frame_t)
    {
      // do nothing;
    }
    else if (data->t_ns < curr_frame_t)
    {
      ImuData<double> left_imu = *data;
      ImuData<double> right_imu = *imu_data_queue[start_imu_idx];

      ImuData<double> interpolated_imu;
      interpolated_imu.t_ns = curr_frame_t;
      if (meas->integration_flag == 0)
      {
        interpolated_imu.accel = right_imu.accel;
        interpolated_imu.gyro = right_imu.gyro;
      }
      else
      {
        Scalar interpolation_lamda = (curr_frame_t - left_imu.t_ns) * 1.0 / (right_imu.t_ns - left_imu.t_ns);
        interpolated_imu.accel = left_imu.accel + interpolation_lamda * (right_imu.accel - left_imu.accel);
        interpolated_imu.gyro = left_imu.gyro + interpolation_lamda * (right_imu.gyro - left_imu.gyro);
      }

      meas->imu_buff.push_back(interpolated_imu);
    }
    else
    {
      printf("imu1 timestamp err!\n");
      getchar();
    }

    start_imu_idx = start_imu_idx - 1;

    if ((meas->imu_buff.front().t_ns != prev_frame_t) || (meas->imu_buff.back().t_ns != curr_frame_t))
    {
      printf("imu buff timestamp err!\n");
      getchar();
    }
  }

  void update_timeshifted_state()
  {
    std::vector<PoseVelState<double>> frame_states;
    for (const auto &kv : timestam_to_frame_states)
    {
      frame_states.push_back(kv.second);
    }

    // shift state
    size_t start_frame_idx = 1;
    for (auto &kv : timestam_to_frame_states) {
      int64_t curr_frame_t = kv.first + calib->cam_time_offset_ns;

      for (size_t i = start_frame_idx; i < frame_states.size(); i++)
      {
        PoseVelState<double> right_state = frame_states[i];
        if (right_state.t_ns >= curr_frame_t)
        {
          PoseVelState<double> left_state = frame_states[i - 1];
          if (left_state.t_ns > curr_frame_t)
          {
            start_frame_idx = i;
            break;
          }

          Scalar interpolation_lamda = (curr_frame_t - left_state.t_ns) * 1.0 / (right_state.t_ns - left_state.t_ns);
          const SE3 T_w_i0 = left_state.T_w_i;
          const SE3 T_w_i1 = right_state.T_w_i;
          Vector6 theta = Sophus::se3_logd(T_w_i1.inverse() * T_w_i0);
          SE3 T_i_w = Sophus::se3_expd(interpolation_lamda * theta) * T_w_i0.inverse();
          SE3 T_w_i = T_i_w.inverse();
          Vector3 vel_w_i = left_state.vel_w_i + interpolation_lamda * (right_state.vel_w_i - left_state.vel_w_i);

          PoseVelState<double> new_frame_state;
          new_frame_state.t_ns = kv.first;
          new_frame_state.T_w_i = T_w_i;
          new_frame_state.vel_w_i = vel_w_i;

          kv.second = new_frame_state;
          start_frame_idx = i;
          break;
        }
      }
    }
  }

  void reinit() {
    // update time-shifted state via propagation from old time to new time
    update_timeshifted_state();

    // update camera constraints.
    for (auto &kv : aprilgrid_corners_measurements) {
      kv.timestamp_ns += calib->cam_time_offset_ns;
    }

    // update imu preintegration constraints.
    imu_measurements.clear();
    Eigen::aligned_map<int64_t, PoseVelState<double>> updated_timestam_to_frame_states;

    bool first_img = true;
    int64_t prev_frame_t = -1;
    PoseVelState<double> prev_frame_pose_vel;
    bool first_pre_imu = true;
    size_t start_imu_idx = 0;
    for (const auto &kv : timestam_to_frame_states) {
      if (first_img)
      {
        prev_frame_t = kv.first + calib->cam_time_offset_ns;
        prev_frame_pose_vel = kv.second;
        prev_frame_pose_vel.t_ns = prev_frame_t;
        first_img = false;
      }
      else
      {
        int64_t curr_frame_t = kv.first + calib->cam_time_offset_ns;
        PoseVelState<double> curr_frame_pose_vel = kv.second;
        curr_frame_pose_vel.t_ns = curr_frame_t;

        if (prev_frame_t < min_imu_time || curr_frame_t > max_imu_time)
        {
          // don't forget to update!
          prev_frame_t = curr_frame_t;
          prev_frame_pose_vel = curr_frame_pose_vel;
          continue;
        }

        ///////////////////////////////////////////////////////////////////////
        IntegratedImuMeasurement<double>::Ptr meas;
        meas.reset(new IntegratedImuMeasurement<double>(
          prev_frame_t, common_bg, common_ba));

        get_imu_buff(start_imu_idx, prev_frame_t, curr_frame_t, meas);

        if (first_pre_imu)
        {
          prev_frame_pose_vel.angular_vel = meas->imu_buff.front().gyro - common_bg;
          updated_timestam_to_frame_states[prev_frame_t] = prev_frame_pose_vel;
          first_pre_imu = false;
        }
        curr_frame_pose_vel.angular_vel = meas->imu_buff.back().gyro - common_bg;
        updated_timestam_to_frame_states[curr_frame_t] = curr_frame_pose_vel;

        imu_measurements.push_back(meas);
        
        /////////////////////////////////////////////////////////////////////
        prev_frame_t = curr_frame_t;
        prev_frame_pose_vel = curr_frame_pose_vel;
      }
    }

    timestam_to_frame_states.clear();
    timestam_to_frame_states = updated_timestam_to_frame_states;
    recompute_size();

    // Without this, Segmentation fault (core dumped), corrupted double-linked list
    ccd.bias_block_offset = bias_block_offset;
  }

  void recompute_size() {
    offset_poses.clear();
    offset_T_i_c.clear();
    offset_cam_intrinsics.clear();

    // size_t num_knots = spline.numKnots();

    // bias_block_offset = POSE_SIZE * num_knots;

    size_t curr_offset = 0;

    for (const auto &kv : timestam_to_frame_states) {
      offset_poses[kv.first] = curr_offset;
      curr_offset += POSE_VEL_SIZE;
    }

    bias_block_offset = curr_offset;

    size_t T_i_c_block_offset =
        bias_block_offset + ACCEL_BIAS_SIZE + GYRO_BIAS_SIZE + G_SIZE;

    // size_t T_i_c_block_offset = bias_block_offset + G_SIZE;

    offset_T_i_c.emplace_back(T_i_c_block_offset);
    for (size_t i = 0; i < calib->T_i_c.size(); i++)
      offset_T_i_c.emplace_back(offset_T_i_c.back() + POSE_SIZE);

    offset_cam_intrinsics.emplace_back(offset_T_i_c.back());

    if (ccd.opt_intrinsics)
    {
      for (size_t i = 0; i < calib->intrinsics.size(); i++)
        offset_cam_intrinsics.emplace_back(offset_cam_intrinsics.back() +
                                          calib->intrinsics[i].getN());
    }

    // mocap_block_offset = offset_cam_intrinsics.back();

    // opt_size = mocap_block_offset + 2 * POSE_SIZE + 2;

    size_t dim_td = 0;
    if (ccd.opt_cam_time_offset)
    {
      dim_td = 1;
    }
    opt_size = offset_cam_intrinsics.back() + dim_td;

    //    std::cerr << "bias_block_offset " << bias_block_offset << std::endl;
    //    std::cerr << "mocap_block_offset " << mocap_block_offset << std::endl;
    //    std::cerr << "opt_size " << opt_size << std::endl;
    //    std::cerr << "offset_T_i_c.back() " << offset_T_i_c.back() <<
    //    std::endl; std::cerr << "offset_cam_intrinsics.back() " <<
    //    offset_cam_intrinsics.back()
    //              << std::endl;
  }

  // Returns true when converged
  bool optimize(bool use_poses, bool use_april_corners, bool use_mocap,
                double huber_thresh, double stop_thresh, double& error,
                int& num_points, double& reprojection_error,
                bool print_info = true) {
    // std::cerr << "optimize num_knots " << num_knots << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();

    ccd.huber_thresh = huber_thresh;

    LinearizeT lopt(opt_size, &spline, timestam_to_frame_states, ccd);

    tbb::blocked_range<PoseDataIter> pose_range(pose_measurements.begin(),
                                                pose_measurements.end());
    tbb::blocked_range<AprilgridCornersDataIter> april_range(
        aprilgrid_corners_measurements.begin(),
        aprilgrid_corners_measurements.end());

    tbb::blocked_range<MocapPoseDataIter> mocap_pose_range(
        mocap_measurements.begin(), mocap_measurements.end());

    tbb::blocked_range<AccelDataIter> accel_range(accel_measurements.begin(),
                                                  accel_measurements.end());

    tbb::blocked_range<GyroDataIter> gyro_range(gyro_measurements.begin(),
                                                gyro_measurements.end());
    
    tbb::blocked_range<ImuDataIter> imu_range(imu_measurements.begin(),
                                                imu_measurements.end());

    // if (use_poses) {
    //   tbb::parallel_reduce(pose_range, lopt);
    //   // lopt(pose_range);
    // }

    if (use_april_corners) {
      tbb::parallel_reduce(april_range, lopt);
      // lopt(april_range);
    }

    // if (use_mocap && mocap_initialized) {
    //   tbb::parallel_reduce(mocap_pose_range, lopt);
    //   // lopt(mocap_pose_range);
    // } else if (use_mocap && !mocap_initialized) {
    //   std::cout << "Mocap residuals are not used. Initialize Mocap first!"
    //             << std::endl;
    // }

    // tbb::parallel_reduce(accel_range, lopt);
    // tbb::parallel_reduce(gyro_range, lopt);

    tbb::parallel_reduce(imu_range, lopt);

    error = lopt.error;
    num_points = lopt.num_points;
    reprojection_error = lopt.reprojection_error;

    if (print_info)
      std::cout << "[LINEARIZE] Error: " << lopt.error << " num points "
                << lopt.num_points << std::endl;

    lopt.accum.setup_solver();
    Eigen::VectorXd Hdiag = lopt.accum.Hdiagonal();

    auto t2 = std::chrono::high_resolution_clock::now();
    double setup_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf(RED "setup_time %f ms.\n" RESET, setup_time);

    bool converged = false;
    bool step = false;
    int max_iter = 10;

    while (!step && max_iter > 0 && !converged) {
      Eigen::VectorXd Hdiag_lambda = Hdiag * lambda;
      for (int i = 0; i < Hdiag_lambda.size(); i++)
        Hdiag_lambda[i] = std::max(Hdiag_lambda[i], min_lambda);

      // VectorX inc_full = -lopt.accum.solve(&Hdiag_lambda);
      VectorX inc_full = -lopt.accum.solve_new(&Hdiag_lambda);

      double max_inc = inc_full.array().abs().maxCoeff();

      if (max_inc < stop_thresh) converged = true;

      Calibration<Scalar> calib_backup = *calib;
      MocapCalibration<Scalar> mocap_calib_backup = *mocap_calib;
      SplineT spline_backup = spline;
      Vector3 common_ba_backup = common_ba;
      Vector3 common_bg_backup = common_bg;
      Vector3 g_backup = g;

      applyInc(inc_full, offset_cam_intrinsics);

      if (ccd.opt_cam_time_offset)
      {
        // align imu measurements with camera measurements
        reinit();
      }
      tbb::blocked_range<ImuDataIter> updated_imu_range(imu_measurements.begin(), imu_measurements.end());

      ComputeErrorSplineOpt eopt(opt_size, &spline, timestam_to_frame_states, ccd);
      // if (use_poses) {
      //   tbb::parallel_reduce(pose_range, eopt);
      // }

      if (use_april_corners) {
        tbb::parallel_reduce(april_range, eopt);
      }

      // if (use_mocap && mocap_initialized) {
      //   tbb::parallel_reduce(mocap_pose_range, eopt);
      // } else if (use_mocap && !mocap_initialized) {
      //   std::cout << "Mocap residuals are not used. Initialize Mocap first!"
      //             << std::endl;
      // }

      // tbb::parallel_reduce(accel_range, eopt);
      // tbb::parallel_reduce(gyro_range, eopt);

      tbb::parallel_reduce(updated_imu_range, eopt);

      double f_diff = (lopt.error - eopt.error);
      double l_diff = 0.5 * inc_full.dot(inc_full * lambda - lopt.accum.getB());

      // std::cout << "f_diff " << f_diff << " l_diff " << l_diff << std::endl;

      double step_quality = f_diff / l_diff;

      if (step_quality < 0) {
        if (print_info)
          std::cout << "\t[REJECTED] lambda:" << lambda
                    << " step_quality: " << step_quality
                    << " max_inc: " << max_inc << " Error: " << eopt.error
                    << " num points " << eopt.num_points << std::endl;
        lambda = std::min(max_lambda, lambda_vee * lambda);
        lambda_vee *= 2;

        spline = spline_backup;
        *calib = calib_backup;
        *mocap_calib = mocap_calib_backup;
        common_ba = common_ba_backup;
        common_bg = common_bg_backup;
        g = g_backup;

      } else {
        if (print_info)
          std::cout << "\t[ACCEPTED] lambda:" << lambda
                    << " step_quality: " << step_quality
                    << " max_inc: " << max_inc << " Error: " << eopt.error
                    << " num points " << eopt.num_points << std::endl;

        lambda = std::max(
            min_lambda,
            lambda *
                std::max(1.0 / 3, 1 - std::pow(2 * step_quality - 1, 3.0)));
        lambda_vee = 2;

        error = eopt.error;
        num_points = eopt.num_points;
        reprojection_error = eopt.reprojection_error;

        step = true;
      }
      max_iter--;
    }

    if (converged && print_info) {
      std::cout << "[CONVERGED]" << std::endl;
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    printf(RED "iter_time %f ms.\n" RESET, iter_time);

    return converged;
  }

  typename Calibration<Scalar>::Ptr calib;
  typename MocapCalibration<Scalar>::Ptr mocap_calib;
  bool mocap_initialized;

  // refer to vio app
  std::vector<ImuData<double>::Ptr> imu_data_queue;
  int64_t min_imu_time = -1;
  int64_t max_imu_time = -1;
  double cam_time_offset = 0;
  Vector3 common_bg= Vector3::Zero();
  Vector3 common_ba = Vector3::Zero();
  size_t opt_size;
  Eigen::aligned_map<int64_t, PoseVelState<double>> timestam_to_frame_states;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  typedef typename Eigen::aligned_deque<PoseData>::const_iterator PoseDataIter;
  typedef typename Eigen::aligned_deque<GyroData>::const_iterator GyroDataIter;
  typedef
      typename Eigen::aligned_deque<AccelData>::const_iterator AccelDataIter;
  typedef typename Eigen::aligned_deque<AprilgridCornersData>::const_iterator
      AprilgridCornersDataIter;
  typedef typename Eigen::aligned_deque<MocapPoseData>::const_iterator
      MocapPoseDataIter;
  
  typedef typename Eigen::aligned_deque<IntegratedImuMeasurement<double>::Ptr>::const_iterator ImuDataIter;

  void applyInc(VectorX& inc_full,
                const std::vector<size_t>& offset_cam_intrinsics) {
    // size_t num_knots = spline.numKnots();

    // for (size_t i = 0; i < num_knots; i++) {
    //   Vector6 inc = inc_full.template segment<POSE_SIZE>(POSE_SIZE * i);

    //   // std::cerr << "i: " << i << " inc: " << inc.transpose() << std::endl;

    //   spline.applyInc(i, inc);
    // }

    // size_t bias_block_offset = POSE_SIZE * num_knots;

    for (auto &kv : timestam_to_frame_states) {
      // kv.second.T_w_i.translation() += inc_full.template segment<3>(offset_poses[kv.first]);
      // kv.second.T_w_i.so3() = Sophus::SO3d::exp(inc_full.template segment<3>(offset_poses[kv.first] + 3)) * kv.second.T_w_i.so3();
      kv.second.applyInc(inc_full.template segment<POSE_VEL_SIZE>(offset_poses[kv.first]));
    }

    // calib->calib_accel_bias += inc_full.template segment<ACCEL_BIAS_SIZE>(
    //     bias_block_offset + ACCEL_BIAS_OFFSET);

    common_ba += inc_full.template segment<ACCEL_BIAS_SIZE>(
        bias_block_offset + ACCEL_BIAS_OFFSET);

    // calib->calib_gyro_bias += inc_full.template segment<GYRO_BIAS_SIZE>(
    //     bias_block_offset + GYRO_BIAS_OFFSET);
    common_bg += inc_full.template segment<GYRO_BIAS_SIZE>(
        bias_block_offset + GYRO_BIAS_OFFSET);

    // update for 3dof g
    // g += inc_full.template segment<G_SIZE>(bias_block_offset + G_OFFSET);

    // update for 2dof g
    Vector2 g_inc = inc_full.template segment<G_SIZE>(bias_block_offset + G_OFFSET);
    g_sphere(0) += g_inc(0);
    g_sphere(1) += g_inc(1);
    get_g_xyz(g_sphere, g);

    size_t T_i_c_block_offset =
        bias_block_offset + ACCEL_BIAS_SIZE + GYRO_BIAS_SIZE + G_SIZE;
    
    // size_t T_i_c_block_offset = bias_block_offset + G_SIZE;

    for (size_t i = 0; i < calib->T_i_c.size(); i++) {
      calib->T_i_c[i] *= Sophus::se3_expd(inc_full.template segment<POSE_SIZE>(
          T_i_c_block_offset + i * POSE_SIZE));
    }

    if (ccd.opt_intrinsics)
    {
      for (size_t i = 0; i < calib->intrinsics.size(); i++) {
        calib->intrinsics[i].applyInc(inc_full.segment(
            offset_cam_intrinsics[i], calib->intrinsics[i].getN()));
      }
    }

    // size_t mocap_block_offset = offset_cam_intrinsics.back();

    // mocap_calib->T_moc_w *= Sophus::se3_expd(
    //     inc_full.template segment<POSE_SIZE>(mocap_block_offset));
    // mocap_calib->T_i_mark *= Sophus::se3_expd(
    //     inc_full.template segment<POSE_SIZE>(mocap_block_offset + POSE_SIZE));

    // mocap_calib->mocap_time_offset_ns +=
    //     1e9 * inc_full[mocap_block_offset + 2 * POSE_SIZE];

    // calib->cam_time_offset_ns +=
    //     1e9 * inc_full[mocap_block_offset + 2 * POSE_SIZE + 1];

    if (ccd.opt_cam_time_offset)
    {
      calib->cam_time_offset_ns = 1e9 * inc_full[offset_cam_intrinsics.back()];
    }

    cam_time_offset += calib->cam_time_offset_ns * 1e-9;
    printf(RED "cam_time_offset %f s.\n" RESET, cam_time_offset);

    //    std::cout << "bias_block_offset " << bias_block_offset << std::endl;
    //    std::cout << "mocap_block_offset " << mocap_block_offset << std::endl;
  }

  Scalar lambda, min_lambda, max_lambda, lambda_vee;

  int64_t min_time_us, max_time_us;

  Eigen::aligned_deque<PoseData> pose_measurements;
  Eigen::aligned_deque<GyroData> gyro_measurements;
  Eigen::aligned_deque<AccelData> accel_measurements;
  Eigen::aligned_deque<AprilgridCornersData> aprilgrid_corners_measurements;
  Eigen::aligned_deque<MocapPoseData> mocap_measurements;
  Eigen::aligned_deque<IntegratedImuMeasurement<double>::Ptr> imu_measurements;

  typename LinearizeT::CalibCommonData ccd;

  std::unordered_map<int64_t, size_t> offset_poses;
  std::vector<size_t> offset_cam_intrinsics;
  std::vector<size_t> offset_T_i_c;
  // size_t mocap_block_offset;
  size_t bias_block_offset;

  SplineT spline;
  Vector3 g;
  Vector3 g_sphere;

  // frame poses
  Eigen::aligned_map<int64_t, Sophus::SE3d> timestam_to_pose;

  Eigen::aligned_vector<Eigen::Vector4d> aprilgrid_corner_pos_3d;

  int64_t dt_ns;
};  // namespace basalt

}  // namespace basalt
