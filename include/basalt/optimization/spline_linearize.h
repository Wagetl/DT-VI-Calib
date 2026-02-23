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
#ifndef BASALT_SPLINE_LINEARIZE_H
#define BASALT_SPLINE_LINEARIZE_H

#include <basalt/io/dataset_io.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/camera/generic_camera.hpp>
#include <basalt/camera/stereographic_param.hpp>

#include <basalt/optimization/linearize.h>

#include <basalt/utils/test_utils.h>

#include <tbb/blocked_range.h>

namespace basalt {

template <int N, typename Scalar, typename AccumT>
struct LinearizeSplineOpt : public LinearizeBase<Scalar> {
  static const int POSE_SIZE = LinearizeBase<Scalar>::POSE_SIZE;
  static const int POS_SIZE = LinearizeBase<Scalar>::POS_SIZE;
  static const int POS_OFFSET = LinearizeBase<Scalar>::POS_OFFSET;
  static const int ROT_SIZE = LinearizeBase<Scalar>::ROT_SIZE;
  static const int ROT_OFFSET = LinearizeBase<Scalar>::ROT_OFFSET;
  static const int ACCEL_BIAS_SIZE = LinearizeBase<Scalar>::ACCEL_BIAS_SIZE;
  static const int GYRO_BIAS_SIZE = LinearizeBase<Scalar>::GYRO_BIAS_SIZE;
  static const int G_SIZE = LinearizeBase<Scalar>::G_SIZE;
  static const int TIME_SIZE = LinearizeBase<Scalar>::TIME_SIZE;
  static const int ACCEL_BIAS_OFFSET = LinearizeBase<Scalar>::ACCEL_BIAS_OFFSET;
  static const int GYRO_BIAS_OFFSET = LinearizeBase<Scalar>::GYRO_BIAS_OFFSET;
  static const int G_OFFSET = LinearizeBase<Scalar>::G_OFFSET;

  typedef Sophus::SE3<Scalar> SE3;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 6, 6> Matrix6;

  typedef Eigen::Matrix<Scalar, 2, 4> Matrix24;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  typedef Se3Spline<N, Scalar> SplineT;

  typedef typename Eigen::aligned_deque<PoseData>::const_iterator PoseDataIter;
  typedef typename Eigen::aligned_deque<GyroData>::const_iterator GyroDataIter;
  typedef
      typename Eigen::aligned_deque<AccelData>::const_iterator AccelDataIter;
  typedef typename Eigen::aligned_deque<AprilgridCornersData>::const_iterator
      AprilgridCornersDataIter;
  typedef typename Eigen::aligned_deque<MocapPoseData>::const_iterator
      MocapPoseDataIter;

  typedef typename Eigen::aligned_deque<IntegratedImuMeasurement<double>::Ptr>::const_iterator ImuDataIter;

  // typedef typename LinearizeBase<Scalar>::PoseCalibH PoseCalibH;
  typedef typename LinearizeBase<Scalar>::CalibCommonData CalibCommonData;

  AccumT accum;
  Scalar error;
  Scalar reprojection_error;
  int num_points;

  size_t opt_size;

  const SplineT* spline;

  const Eigen::aligned_map<int64_t, PoseVelState<double>>& timestam_to_frame_states;

  LinearizeSplineOpt(size_t opt_size, const SplineT* spl,
                     const Eigen::aligned_map<int64_t, PoseVelState<double>>& timestam_to_frame_states,
                     const CalibCommonData& common_data)
      : opt_size(opt_size), spline(spl), timestam_to_frame_states(timestam_to_frame_states) {
    this->common_data = common_data;

    accum.reset(opt_size);
    error = 0;
    reprojection_error = 0;
    num_points = 0;

    BASALT_ASSERT(spline);
  }

  LinearizeSplineOpt(const LinearizeSplineOpt& other, tbb::split)
      : opt_size(other.opt_size), spline(other.spline), timestam_to_frame_states(other.timestam_to_frame_states) {
    this->common_data = other.common_data;
    accum.reset(opt_size);
    error = 0;
    reprojection_error = 0;
    num_points = 0;
  }

  void operator()(const tbb::blocked_range<ImuDataIter>& r) {
    for (const IntegratedImuMeasurement<double>::Ptr imu_meas : r) {
      MatrixX Jp;
      VectorX r;
      Jp.resize(POSE_VEL_SIZE, 2 * POSE_VEL_SIZE + ACCEL_BIAS_SIZE + GYRO_BIAS_SIZE + G_SIZE);
      r.resize(POSE_VEL_SIZE);
      Jp.setZero();
      r.setZero();

      const int64_t start_t = imu_meas->get_start_t_ns();
      const int64_t end_t = imu_meas->imu_buff.back().t_ns;
      if (!timestam_to_frame_states.count(start_t))
      {
        printf("err: no start_t %f\n", start_t*1e-9);
        getchar();
        return;
      }
      if (!timestam_to_frame_states.count(end_t))
      {
        printf("err: no end_t %f\n", end_t*1e-9);
        getchar();
        return;
      }

      const size_t start_idx = 0;
      const size_t end_idx = POSE_VEL_SIZE;
      const size_t ba_idx = 2 * POSE_VEL_SIZE + ACCEL_BIAS_OFFSET;
      const size_t bg_idx = 2 * POSE_VEL_SIZE + GYRO_BIAS_OFFSET;
      const size_t g_idx = 2 * POSE_VEL_SIZE + G_OFFSET;

      PoseVelState<Scalar> start_state = timestam_to_frame_states.at(start_t);
      PoseVelState<Scalar> end_state = timestam_to_frame_states.at(end_t);

      typename IntegratedImuMeasurement<Scalar>::MatNN d_res_d_start, d_res_d_end;
      typename IntegratedImuMeasurement<Scalar>::MatN3 d_res_d_bg, d_res_d_ba;

      typename PoseVelState<Scalar>::VecN res = imu_meas->residual(
          start_state, *(this->common_data.g), end_state,
          *(this->common_data.common_bg),
          *(this->common_data.common_ba), &d_res_d_start, &d_res_d_end,
          &d_res_d_bg, &d_res_d_ba);

      // d_res_d_state0.template block<3, 3>(0, 6) = -R0_inv * dt;
      // d_res_d_state0.template block<3, 3>(6, 6) = -R0_inv;

      Scalar Delta_dt = imu_meas->getDeltaState().t_ns * Scalar(1e-9);
      Matrix3 dp_dg = 0.5 * Delta_dt * d_res_d_start.template block<3, 3>(0, 6);
      Matrix3 dv_dg = Delta_dt * d_res_d_start.template block<3, 3>(6, 6);
      Eigen::Matrix<Scalar, 9, 3> d_res_dg_orig;
      d_res_dg_orig.setZero();
      d_res_dg_orig.template block<3, 3>(0, 0) = dp_dg;
      d_res_dg_orig.template block<3, 3>(6, 0) = dv_dg;

      Eigen::Matrix3d H_f;
      get_g_jacobian_representation(*(this->common_data.g), H_f);
      Eigen::Matrix<double, 9, 2> d_res_dg = d_res_dg_orig * H_f.block(0, 0, 3, 2);

      Jp.template block<9, 2>(0, g_idx) =
          imu_meas->get_sqrt_cov_inv() * d_res_dg;

      // error
      Scalar imu_error =
          Scalar(0.5) * (imu_meas->get_sqrt_cov_inv() * res).squaredNorm();

      // imu residual linearization
      Jp.template block<9, 9>(0, start_idx) =
          imu_meas->get_sqrt_cov_inv() * d_res_d_start;
      Jp.template block<9, 9>(0, end_idx) =
          imu_meas->get_sqrt_cov_inv() * d_res_d_end;

      Jp.template block<9, 3>(0, bg_idx) =
          imu_meas->get_sqrt_cov_inv() * d_res_d_bg;
      Jp.template block<9, 3>(0, ba_idx) =
          imu_meas->get_sqrt_cov_inv() * d_res_d_ba;

      r.template segment<9>(0) = imu_meas->get_sqrt_cov_inv() * res;

      error += imu_error;

      const size_t start_i = this->common_data.offset_poses->at(start_t);
      const size_t end_i = this->common_data.offset_poses->at(end_t);
      const size_t ba_i = this->common_data.bias_block_offset + ACCEL_BIAS_OFFSET;
      const size_t bg_i = this->common_data.bias_block_offset + GYRO_BIAS_OFFSET;
      const size_t g_i = this->common_data.bias_block_offset + G_OFFSET;

      const MatrixX H = Jp.transpose() * Jp;
      const VectorX b = Jp.transpose() * r;

      // 1st column
      accum.template addH<POSE_VEL_SIZE, POSE_VEL_SIZE>(
          start_i, start_i,
          H.template block<POSE_VEL_SIZE, POSE_VEL_SIZE>(
            start_idx, start_idx));

      accum.template addH<POSE_VEL_SIZE, POSE_VEL_SIZE>(
          end_i, start_i,
          H.template block<POSE_VEL_SIZE, POSE_VEL_SIZE>(
              end_idx, start_idx));

      accum.template addH<ACCEL_BIAS_SIZE, POSE_VEL_SIZE>(
          ba_i, start_i,
          H.template block<ACCEL_BIAS_SIZE, POSE_VEL_SIZE>(
              ba_idx, start_idx));
      
      accum.template addH<GYRO_BIAS_SIZE, POSE_VEL_SIZE>(
          bg_i, start_i,
          H.template block<GYRO_BIAS_SIZE, POSE_VEL_SIZE>(
              bg_idx, start_idx));
      accum.template addH<G_SIZE, POSE_VEL_SIZE>(
          g_i, start_i,
          H.template block<G_SIZE, POSE_VEL_SIZE>(
              g_idx, start_idx));

      // 2nd column
      accum.template addH<POSE_VEL_SIZE, POSE_VEL_SIZE>(
          end_i, end_i,
          H.template block<POSE_VEL_SIZE, POSE_VEL_SIZE>(
              end_idx, end_idx));

      accum.template addH<ACCEL_BIAS_SIZE, POSE_VEL_SIZE>(
          ba_i, end_i,
          H.template block<ACCEL_BIAS_SIZE, POSE_VEL_SIZE>(
              ba_idx, end_idx));
      
      accum.template addH<GYRO_BIAS_SIZE, POSE_VEL_SIZE>(
          bg_i, end_i,
          H.template block<GYRO_BIAS_SIZE, POSE_VEL_SIZE>(
              bg_idx, end_idx));

      accum.template addH<G_SIZE, POSE_VEL_SIZE>(
          g_i, end_i,
          H.template block<G_SIZE, POSE_VEL_SIZE>(
              g_idx, end_idx));
      
      // 3rd column
      accum.template addH<ACCEL_BIAS_SIZE, ACCEL_BIAS_SIZE>(
          ba_i, ba_i,
          H.template block<ACCEL_BIAS_SIZE, ACCEL_BIAS_SIZE>(
              ba_idx, ba_idx));
      
      accum.template addH<GYRO_BIAS_SIZE, ACCEL_BIAS_SIZE>(
          bg_i, ba_i,
          H.template block<GYRO_BIAS_SIZE, ACCEL_BIAS_SIZE>(
              bg_idx, ba_idx));

      accum.template addH<G_SIZE, ACCEL_BIAS_SIZE>(
          g_i, ba_i,
          H.template block<G_SIZE, ACCEL_BIAS_SIZE>(
              g_idx, ba_idx));

      // 4th column
      accum.template addH<GYRO_BIAS_SIZE, GYRO_BIAS_SIZE>(
          bg_i, bg_i,
          H.template block<GYRO_BIAS_SIZE, GYRO_BIAS_SIZE>(
              bg_idx, bg_idx));

      accum.template addH<G_SIZE, GYRO_BIAS_SIZE>(
          g_i, bg_i,
          H.template block<G_SIZE, GYRO_BIAS_SIZE>(
              g_idx, bg_idx));

      // 5th column
      accum.template addH<G_SIZE, G_SIZE>(
          g_i, g_i,
          H.template block<G_SIZE, G_SIZE>(
              g_idx, g_idx));

      // residual part
      accum.template addB<POSE_VEL_SIZE>(
          start_i, b.template segment<POSE_VEL_SIZE>(start_idx));
      accum.template addB<POSE_VEL_SIZE>(
          end_i, b.template segment<POSE_VEL_SIZE>(end_idx));
      accum.template addB<ACCEL_BIAS_SIZE>(
          ba_i, b.template segment<ACCEL_BIAS_SIZE>(ba_idx));
      accum.template addB<GYRO_BIAS_SIZE>(
          bg_i, b.template segment<GYRO_BIAS_SIZE>(bg_idx));
      accum.template addB<G_SIZE>(
          g_i, b.template segment<G_SIZE>(g_idx));
    }
  }

  void operator()(const tbb::blocked_range<AprilgridCornersDataIter>& r) {
    for (const AprilgridCornersData& acd : r) {
      std::visit(
          [&](const auto& cam) {
            constexpr int INTRINSICS_SIZE =
                std::remove_reference<decltype(cam)>::type::N;

            // typename SplineT::PosePosSO3JacobianStruct J;

            // int64_t time_ns = acd.timestamp_ns +
            //                   this->common_data.calibration->cam_time_offset_ns;

            // if (time_ns < spline->minTimeNs() || time_ns >= spline->maxTimeNs())
            //   return;

            // SE3 T_w_i = spline->pose(time_ns, &J);

            // Vector6 d_T_wi_d_time;
            // spline->d_pose_d_t(time_ns, d_T_wi_d_time);

            if (!timestam_to_frame_states.count(acd.timestamp_ns))
            {
              return;
            }

            SE3 T_w_i = timestam_to_frame_states.at(acd.timestamp_ns).T_w_i;

            SE3 T_w_c =
                T_w_i * this->common_data.calibration->T_i_c[acd.cam_id];
            SE3 T_c_w = T_w_c.inverse();
            Eigen::Matrix4d T_c_w_m = T_c_w.matrix();

            typename LinearizeBase<Scalar>::template PoseCalibH<INTRINSICS_SIZE>
                cph;

            double err = 0;
            double reproj_err = 0;
            int num_inliers = 0;

            for (size_t i = 0; i < acd.corner_pos.size(); i++) {
              this->linearize_point(acd.corner_pos[i], acd.corner_id[i],
                                    T_c_w_m, cam, &cph, err, num_inliers,
                                    reproj_err);
            }

            error += err;
            reprojection_error += reproj_err;
            num_points += num_inliers;

            Matrix6 Adj = -this->common_data.calibration->T_i_c[acd.cam_id]
                               .inverse()
                               .Adj();
            
            Matrix3 R_inv = T_w_i.so3().inverse().matrix();

            Vector6 d_T_wi_d_time = Vector6::Zero();
            Vector3 vel_w_i = timestam_to_frame_states.at(acd.timestamp_ns).vel_w_i;
            Vector3 angular_vel = timestam_to_frame_states.at(acd.timestamp_ns).angular_vel;
            d_T_wi_d_time.template head<3>() = R_inv * vel_w_i;
            d_T_wi_d_time.template tail<3>() = angular_vel;
            Vector6 Adj_time = Adj * d_T_wi_d_time;
            // comment this to enable time offset estimation if it works.
            // Adj_time.setZero();

            Matrix6 d_full_d_split = Matrix6::Zero();
            d_full_d_split.block(0, 0, 3, 3) = R_inv;
            d_full_d_split.block(3, 3, 3, 3) = R_inv;
            Adj = Adj * d_full_d_split;

            Matrix6 A_H_p_A = Adj.transpose() * cph.H_pose_accum * Adj;

            size_t T_i_c_start = this->common_data.offset_T_i_c->at(acd.cam_id);
            
            size_t start_cam_time_offset = this->common_data.offset_intrinsics->back();
            
            // size_t start_cam_time_offset =
            //     this->common_data.mocap_block_offset + 2 * POSE_SIZE + 1;

            // for (int i = 0; i < N; i++) {
            //   int start_i = (J.start_idx + i) * POSE_SIZE;

            //   Matrix6 Ji_A_H_p_A = J.d_val_d_knot[i].transpose() * A_H_p_A;

            //   for (int j = 0; j <= i; j++) {
            //     int start_j = (J.start_idx + j) * POSE_SIZE;

            //     accum.template addH<POSE_SIZE, POSE_SIZE>(
            //         start_i, start_j, Ji_A_H_p_A * J.d_val_d_knot[j]);
            //   }

            //   accum.template addH<POSE_SIZE, POSE_SIZE>(
            //       T_i_c_start, start_i,
            //       -cph.H_pose_accum * Adj * J.d_val_d_knot[i]);

            //   if (this->common_data.opt_intrinsics)
            //     accum.template addH<INTRINSICS_SIZE, POSE_SIZE>(
            //         calib_start, start_i,
            //         cph.H_intr_pose_accum * Adj * J.d_val_d_knot[i]);

            //   if (this->common_data.opt_cam_time_offset)
            //     accum.template addH<TIME_SIZE, POSE_SIZE>(
            //         start_cam_time_offset, start_i,
            //         d_T_wi_d_time.transpose() * A_H_p_A * J.d_val_d_knot[i]);

            //   accum.template addB<POSE_SIZE>(
            //       start_i, J.d_val_d_knot[i].transpose() * Adj.transpose() *
            //                    cph.b_pose_accum);
            // }

            const size_t start_i =
                this->common_data.offset_poses->at(acd.timestamp_ns);

            accum.template addH<POSE_SIZE, POSE_SIZE>(
                start_i, start_i, A_H_p_A);

            accum.template addH<POSE_SIZE, POSE_SIZE>(
                T_i_c_start, start_i,
                -cph.H_pose_accum * Adj);

            if (this->common_data.opt_intrinsics)
            {
              size_t calib_start =
                this->common_data.offset_intrinsics->at(acd.cam_id);
              
              accum.template addH<INTRINSICS_SIZE, POSE_SIZE>(
                  calib_start, start_i,
                  cph.H_intr_pose_accum * Adj);
            }

            if (this->common_data.opt_cam_time_offset)
              accum.template addH<TIME_SIZE, POSE_SIZE>(
                  start_cam_time_offset, start_i,
                  Adj_time.transpose() * cph.H_pose_accum * Adj);

            accum.template addB<POSE_SIZE>(
                start_i, Adj.transpose() * cph.b_pose_accum);
            ///////////////////////////////////////////////////////////////////

            accum.template addH<POSE_SIZE, POSE_SIZE>(T_i_c_start, T_i_c_start,
                                                      cph.H_pose_accum);
            accum.template addB<POSE_SIZE>(T_i_c_start, -cph.b_pose_accum);

            if (this->common_data.opt_intrinsics) {
              size_t calib_start =
                this->common_data.offset_intrinsics->at(acd.cam_id);
              
              accum.template addH<INTRINSICS_SIZE, POSE_SIZE>(
                  calib_start, T_i_c_start, -cph.H_intr_pose_accum);
              accum.template addH<INTRINSICS_SIZE, INTRINSICS_SIZE>(
                  calib_start, calib_start, cph.H_intr_accum);
              accum.template addB<INTRINSICS_SIZE>(calib_start,
                                                   cph.b_intr_accum);
            }

            // if (this->common_data.opt_cam_time_offset) {
            //   accum.template addH<TIME_SIZE, POSE_SIZE>(
            //       start_cam_time_offset, T_i_c_start,
            //       -d_T_wi_d_time.transpose() * Adj.transpose() *
            //           cph.H_pose_accum);

            //   if (this->common_data.opt_intrinsics)
            //     accum.template addH<TIME_SIZE, INTRINSICS_SIZE>(
            //         start_cam_time_offset, calib_start,
            //         d_T_wi_d_time.transpose() * Adj.transpose() *
            //             cph.H_intr_pose_accum.transpose());

            //   accum.template addH<TIME_SIZE, TIME_SIZE>(
            //       start_cam_time_offset, start_cam_time_offset,
            //       d_T_wi_d_time.transpose() * A_H_p_A * d_T_wi_d_time);

            //   accum.template addB<TIME_SIZE>(start_cam_time_offset,
            //                                  d_T_wi_d_time.transpose() *
            //                                      Adj.transpose() *
            //                                      cph.b_pose_accum);
            // }

            if (this->common_data.opt_cam_time_offset) {
              accum.template addH<TIME_SIZE, POSE_SIZE>(
                  start_cam_time_offset, T_i_c_start,
                  -Adj_time.transpose() * cph.H_pose_accum);

              // if (this->common_data.opt_intrinsics)
              //   accum.template addH<TIME_SIZE, INTRINSICS_SIZE>(
              //       start_cam_time_offset, calib_start,
              //       Adj_time.transpose() * cph.H_intr_pose_accum.transpose());

              accum.template addH<TIME_SIZE, TIME_SIZE>(
                  start_cam_time_offset, start_cam_time_offset,
                  Adj_time.transpose() * cph.H_pose_accum * Adj_time);

              accum.template addB<TIME_SIZE>(start_cam_time_offset,
                                             Adj_time.transpose() * cph.b_pose_accum);
            }
          },
          this->common_data.calibration->intrinsics[acd.cam_id].variant);
    }
  }

  void join(LinearizeSplineOpt& rhs) {
    accum.join(rhs.accum);
    error += rhs.error;
    reprojection_error += rhs.reprojection_error;
    num_points += rhs.num_points;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <int N, typename Scalar>
struct ComputeErrorSplineOpt : public LinearizeBase<Scalar> {
  typedef Sophus::SE3<Scalar> SE3;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;

  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
  typedef Eigen::Matrix<Scalar, 6, 6> Matrix6;

  typedef Eigen::Matrix<Scalar, 2, 4> Matrix24;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  typedef Se3Spline<N, Scalar> SplineT;

  typedef typename Eigen::aligned_deque<PoseData>::const_iterator PoseDataIter;
  typedef typename Eigen::aligned_deque<GyroData>::const_iterator GyroDataIter;
  typedef
      typename Eigen::aligned_deque<AccelData>::const_iterator AccelDataIter;
  typedef typename Eigen::aligned_deque<AprilgridCornersData>::const_iterator
      AprilgridCornersDataIter;
  typedef typename Eigen::aligned_deque<MocapPoseData>::const_iterator
      MocapPoseDataIter;

  typedef typename Eigen::aligned_deque<IntegratedImuMeasurement<double>::Ptr>::const_iterator ImuDataIter;

  // typedef typename LinearizeBase<Scalar>::PoseCalibH PoseCalibH;
  typedef typename LinearizeBase<Scalar>::CalibCommonData CalibCommonData;

  Scalar error;
  Scalar reprojection_error;
  int num_points;

  size_t opt_size;

  const SplineT* spline;

  const Eigen::aligned_map<int64_t, PoseVelState<double>>& timestam_to_frame_states;

  ComputeErrorSplineOpt(size_t opt_size, const SplineT* spl,
                        const Eigen::aligned_map<int64_t, PoseVelState<double>>& timestam_to_frame_states,
                        const CalibCommonData& common_data)
      : opt_size(opt_size), spline(spl), timestam_to_frame_states(timestam_to_frame_states) {
    this->common_data = common_data;

    error = 0;
    reprojection_error = 0;
    num_points = 0;

    BASALT_ASSERT(spline);
  }

  ComputeErrorSplineOpt(const ComputeErrorSplineOpt& other, tbb::split)
      : opt_size(other.opt_size), spline(other.spline), timestam_to_frame_states(other.timestam_to_frame_states) {
    this->common_data = other.common_data;
    error = 0;
    reprojection_error = 0;
    num_points = 0;
  }

  void operator()(const tbb::blocked_range<ImuDataIter>& r) {
    for (const IntegratedImuMeasurement<double>::Ptr imu_meas : r) {
      const int64_t start_t = imu_meas->get_start_t_ns();
      const int64_t end_t = imu_meas->imu_buff.back().t_ns;
      if (!timestam_to_frame_states.count(start_t))
      {
        printf("err: no start_t %f\n", start_t*1e-9);
        getchar();
        return;
      }
      if (!timestam_to_frame_states.count(end_t))
      {
        printf("err: no end_t %f\n", end_t*1e-9);
        getchar();
        return;
      }

      PoseVelState<Scalar> start_state = timestam_to_frame_states.at(start_t);
      PoseVelState<Scalar> end_state = timestam_to_frame_states.at(end_t);

      imu_meas->reinit(*(this->common_data.common_bg), *(this->common_data.common_ba));
      imu_meas->reintegrate(this->common_data.accel_var, this->common_data.gyro_var);
      
      const typename PoseVelState<Scalar>::VecN res = imu_meas->residual(
          start_state, *(this->common_data.g), end_state,
          *(this->common_data.common_bg), *(this->common_data.common_ba));

      Scalar imu_error =
          Scalar(0.5) * res.transpose() * imu_meas->get_cov_inv() * res;

      error += imu_error;
    }
  }

  void operator()(const tbb::blocked_range<AprilgridCornersDataIter>& r) {
    for (const AprilgridCornersData& acd : r) {
      std::visit(
          [&](const auto& cam) {
            // int64_t time_ns = acd.timestamp_ns +
            //                   this->common_data.calibration->cam_time_offset_ns;

            // if (time_ns < spline->minTimeNs() || time_ns >= spline->maxTimeNs())
            //   return;

            // SE3 T_w_i = spline->pose(time_ns);

            if (!timestam_to_frame_states.count(acd.timestamp_ns))
            {
              return;
            }

            SE3 T_w_i = timestam_to_frame_states.at(acd.timestamp_ns).T_w_i;

            SE3 T_w_c =
                T_w_i * this->common_data.calibration->T_i_c[acd.cam_id];
            SE3 T_c_w = T_w_c.inverse();
            Eigen::Matrix4d T_c_w_m = T_c_w.matrix();

            double err = 0;
            double reproj_err = 0;
            int num_inliers = 0;

            for (size_t i = 0; i < acd.corner_pos.size(); i++) {
              this->linearize_point(acd.corner_pos[i], acd.corner_id[i],
                                    T_c_w_m, cam, nullptr, err, num_inliers,
                                    reproj_err);
            }

            error += err;
            reprojection_error += reproj_err;
            num_points += num_inliers;
          },
          this->common_data.calibration->intrinsics[acd.cam_id].variant);
    }
  }

  void join(ComputeErrorSplineOpt& rhs) {
    error += rhs.error;
    reprojection_error += rhs.reprojection_error;
    num_points += rhs.num_points;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace basalt

#endif
