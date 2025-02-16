#include "../include/eigen.h"
#include <cmath>

std::vector<Eigen::Vector3d> ConvertEigenToVector(const Points &mat) {
    std::vector<Eigen::Vector3d> points;
    for (int i = 0; i < mat.cols(); ++i) {
        points.emplace_back(mat(0, i), mat(1, i), mat(2, i));
    }
    return points;
}

RotationMatrix getRotation(double y, double p, double r) {
    RotationMatrix R_x, R_y, R_z, R_True;
    R_x << cos(r), -sin(r), 0,
           sin(r), cos(r), 0,
           0, 0, 1;
    R_y << cos(p), 0, sin(p),
           0, 1, 0,
           -sin(p), 0, cos(p);
    R_z << 1, 0, 0, 
           0, cos(y), -sin(y),
           0, sin(y), cos(y);
    R_True = R_z * R_y * R_x;
    
    return R_True;
}
