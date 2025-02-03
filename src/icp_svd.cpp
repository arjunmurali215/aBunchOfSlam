#include <iostream>
#include <iomanip>

using namespace std;

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/SVD>

using namespace Eigen;

#include <pangolin/pangolin.h>

#include <cmath>

const int N = 30;

typedef Matrix<double, 3, 3> CovarianceMatrix;
typedef Matrix<double, 3, 3> RotationMatrix;
typedef Matrix<double, 3, 1> TranslationMatrix;
typedef Matrix<double, 3, N> Points;
typedef Matrix<double, 3, 1> Point;

std::vector<Eigen::Vector3d> ConvertEigenToVector(const Points &mat) {
    std::vector<Eigen::Vector3d> points;
    for (int i = 0; i < mat.cols(); ++i) {
        points.emplace_back(mat(0, i), mat(1, i), mat(2, i));
    }
    return points;
}

void DrawPoints(const Points &pointsm, float r, float g, float b, float size) {
    std::vector<Eigen::Vector3d> points = ConvertEigenToVector(pointsm);
    glColor3f(r, g, b);
    glPointSize(size);
    glBegin(GL_POINTS);
    for (const auto &p : points) {
        glVertex3d(p.x(), p.y(), p.z());
    }
    glEnd();
}

void DrawCorrespondences(const Points &P, const Points &Q, const vector<int> &correspondences) {
    glColor3f(0.0f, 1.0f, 0.0f); 
    glBegin(GL_LINES);
    for (int i = 0; i < P.cols(); ++i) {
        int j = correspondences[i]; 
        if (j != -1) {
            glVertex3d(P(0, i), P(1, i), P(2, i));
            glVertex3d(Q(0, j), Q(1, j), Q(2, j));
        }
    }
    glEnd();
}

void Render(const Points &moved_data, const Points &true_data, const vector<int> &correspondences) {
    // Create a Pangolin window
    pangolin::CreateWindowAndBind("3D Eigen Points Visualization", 640, 480);
    glEnable(GL_DEPTH_TEST);
    
    // Camera settings
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 500, 500, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAt(0, 0, 20, 0, 0, 0, pangolin::AxisY)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    // Render Loop
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        DrawPoints(true_data, 1.0f, 0.0f, 0.0f, 3.0f);
        DrawPoints(moved_data, 0.0f, 0.0f, 1.0f, 5.0f);
        DrawCorrespondences(moved_data, true_data, correspondences);

        pangolin::FinishFrame();
    }
}

Point com(Points points) {
    Point com = points.rowwise().mean();
    return com;
}

std::vector<int> correspondences(Points P, Points Q) {
    int p_size = P.cols();
    int q_size = Q.cols();
    vector<int> indices;
    
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
    vector<int> indices;

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
        cout<<covariance<<endl<<endl<<R<<endl<<endl<<T<<endl;
        moved_data = (R * moved_data).colwise() + T;
    }

    Render(moved_data, true_data, indices);
}
