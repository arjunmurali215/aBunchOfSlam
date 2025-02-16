#include "../include/eigen.h"
#include "../include/pangolin.h"

Point com(Points points) {
    Point com = points.rowwise().mean();
    return com;
}

std::vector<int> correspondences(Points P, Points Q) {
    int p_size = P.cols();
    int q_size = Q.cols();
    std::vector<int> indices;
    
    for (int i = 0; i < p_size; ++i) {
        Point point_in_p = P.col(i);
        double mindist = std::numeric_limits<double>::max();
        int index = -1;

        for (int j = 0; j < q_size; ++j) {
            Point point_in_q = Q.col(j);
            double dist = (point_in_p - point_in_q).squaredNorm();
            if (dist < mindist) {
                mindist = dist;
                index = j;
            }
        }

        indices.push_back(index);
    }

    return indices;
}

CovarianceMatrix getCovariance(Points P, Points Q) {

    return P * Q.transpose();
}

int main() {

    double r = M_PI / 4; 
    double p = M_PI / 4; 
    double y = M_PI / 4; 
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

    TranslationMatrix t_True;
    t_True << 1, 2, 3;

    Points true_data;
    for (int i = 0; i < N; ++i) {
        true_data(0, i) = 0.2*i*cos(i * 0.5);
        true_data(1, i) = 0.2*i*sin(i * 0.5);
        true_data(2, i) = i;
    }


    Points moved_data = Points::Zero();
    moved_data = R_True * true_data;
    moved_data.colwise() += t_True;


    //cout<<true_data<<endl<<moved_data<<endl;
    std::vector<int> indices;

    for(int i =0; i<10; i++){
        Point true_center, moved_center;
        true_center = com(true_data);
        moved_center = com(moved_data);
        true_data = true_data.colwise() - true_center;
        moved_data = moved_data.colwise() - moved_center;

        indices = correspondences(moved_data, true_data);

        CovarianceMatrix covariance = getCovariance(true_data, moved_data);

        JacobiSVD<CovarianceMatrix, ComputeFullU | ComputeFullV> svd(covariance);
        svd.compute(covariance, ComputeFullU | ComputeFullV);

        RotationMatrix R = svd.matrixU() * svd.matrixV().transpose();
        TranslationMatrix T = com(true_data) - R * com(moved_data);
        std::cout<<covariance<<std::endl<<std::endl<<R<<std::endl<<std::endl<<T<<std::endl;
        moved_data = (R * moved_data).colwise() + T;
    }

    Render(moved_data, true_data, indices);
}
