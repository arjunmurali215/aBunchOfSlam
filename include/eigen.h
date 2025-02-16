#ifndef EIGEN_H
#define EIGEN_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/SVD>
#include <vector>

using namespace Eigen;

const int N = 30;

// Type definitions
typedef Matrix<double, 6, 6> Hessian;
typedef Matrix<double, 6, 1> Pose;
typedef Matrix<double, 6, 1> Gradient; 
typedef Matrix<double, 3, 6> Jacobian; 
typedef Matrix<double, 3, 3> RotationMatrix;
typedef Matrix<double, 3, 3> CovarianceMatrix;
typedef Matrix<double, 3, 1> TranslationMatrix;
typedef Matrix<double, 3, N> Points;
typedef Matrix<double, 3, 1> Point;

// Function declarations
std::vector<Eigen::Vector3d> ConvertEigenToVector(const Points &mat);
RotationMatrix getRotation(double y, double p, double r);

#endif // EIGEN_H
