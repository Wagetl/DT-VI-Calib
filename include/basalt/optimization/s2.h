#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace basalt {

  inline void get_g_xyz(const Eigen::Matrix<double, 3, 1> &p_sphere, Eigen::Matrix<double, 3, 1> &p_1in0)
  {
    p_1in0 << (p_sphere(2)) * std::cos(p_sphere(0)) * std::sin(p_sphere(1)),
              (p_sphere(2)) * std::sin(p_sphere(0)) * std::sin(p_sphere(1)),
              (p_sphere(2)) * std::cos(p_sphere(1));
  }

  inline void get_g_sphere(const Eigen::Matrix<double, 3, 1> &p_1in0, Eigen::Matrix<double, 3, 1> &p_sphere)
  {
    double g_rho = p_1in0.norm();
    double g_phi = std::acos(p_1in0(2) / g_rho);
    double g_theta = std::atan2(p_1in0(1), p_1in0(0));

    p_sphere(0) = g_theta;
    p_sphere(1) = g_phi;
    p_sphere(2) = g_rho;
  }

  inline void get_g_jacobian_representation(const Eigen::Matrix<double, 3, 1> &p_1in0, Eigen::Matrix3d &H_f)
  {
    Eigen::Matrix<double, 3, 1> p_sphere;
    get_g_sphere(p_1in0, p_sphere);

    double sin_th = std::sin(p_sphere(0, 0));
    double cos_th = std::cos(p_sphere(0, 0));
    double sin_phi = std::sin(p_sphere(1, 0));
    double cos_phi = std::cos(p_sphere(1, 0));
    double rho = p_sphere(2, 0);

    // Construct the Jacobian
    H_f << -(rho) * sin_th * sin_phi, (rho) * cos_th * cos_phi, cos_th * sin_phi,
        (rho) * cos_th * sin_phi, (rho) * sin_th * cos_phi, sin_th * sin_phi, 0.0,
        -(rho) * sin_phi, cos_phi;
  }

}  // namespace basalt
