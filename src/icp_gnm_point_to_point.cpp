#include "../include/eigen.h"
#include "../include/pangolin.h"

std::vector<int> correspondences(Points P, Points Q) {
    int p_size = P.cols();
    int q_size = Q.cols();
    std::vector<int> indices(p_size, -1);

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

        indices[i] = index;
    }

    return indices;
}

Jacobian getJacob(Point p, double a, double b, double c) {
    Jacobian J = Jacobian::Zero();
    J.block<3,3>(0,0) = Eigen::Matrix3d::Identity(); // Identity for translation derivatives
    
    J(0,3) = (cos(a)*sin(b)*cos(c) + sin(a)*sin(c))*p(1) + (-sin(a)*sin(b)*cos(c) + cos(a)*sin(c))*p(2);
    J(0,4) = p(0) * (-sin(b)*cos(c)) + p(1)*sin(a)*cos(b)*cos(c) + p(2)*cos(a)*cos(b)*cos(c);
    J(0,5) = p(0) *(-cos(b)*sin(c)) + p(1)*(-sin(a)*sin(b)*sin(c) - cos(a)*cos(c)) + p(2)*(-cos(a)*sin(b)*sin(c) + sin(a)*cos(c));
    J(1,3) = (cos(a)*sin(b)*sin(c) - sin(a)*cos(c))*p(1) + (-sin(a)*sin(b)*sin(c) - cos(a)*cos(c))*p(2);
    J(1,4) = p(0)*(-sin(b)*sin(c)) + p(1)*sin(a)*cos(b)*sin(c) + p(2)*cos(a)*cos(b)*sin(c);
    J(1,5) = p(0)*cos(b)*cos(c) + p(1)*(-sin(a)*sin(b)*cos(c) + cos(a)*sin(c)) + p(2)*(-cos(a)*sin(b)*cos(c) - sin(a)*sin(c));
    J(2,3) = cos(a)*cos(b)*p(1) - sin(a)*cos(b)*p(2);
    J(2,4) = p(0)*(-cos(b)) - p(1)*sin(a)*sin(b) - p(2)*cos(a)*sin(b);
    J(2,5) = 0;


    return J;
}

int main() {
    double a = 0; 
    double b = 0; 
    double c = M_PI/4; 
    RotationMatrix R_True = getRotation(a, b, c);

    TranslationMatrix t_True;
    t_True << 1, 2, 0;

    Points true_data = Points::Zero();
    for (int i = 0; i < N; ++i) {
        true_data(0, i) = 3 * cos(2 * M_PI * i / N);
        true_data(1, i) = 9 *  sin(2 * M_PI * i / N);
        true_data(2, i) = 0;
    }

    Points moved_data = R_True * true_data;
    moved_data.colwise() += t_True;

    std::vector<int> indices;
    Pose x = Pose::Zero();

    for(int iter = 0; iter <500; iter++) {
        indices = correspondences(moved_data, true_data);
        Hessian H = Hessian::Zero();
        Gradient g = Gradient::Zero();

        RotationMatrix R;
        TranslationMatrix t;
        Point p;
        Point q;

        for (int j = 0; j < N; j++) {

            Point p = moved_data.col(j);
            Point q = true_data.col(indices[j]);

            double aa = x(3);
            double bb = x(4);
            double cc = x(5);
            R = getRotation(aa, bb, cc);
            
            t << x(0), x(1), x(2);

            Jacobian J = getJacob(p, aa, bb, cc);
            H += J.transpose() * J;
            g += J.transpose() * (R * p + t - q);

            // Pose dx = H.inverse()*(-g);  // Fix sign
            // x += dx;
        }

        Pose dx = H.inverse()*(-g);  // Fix sign
        x+=dx;
        auto e = R * p + t - q;
        std::cout<< e.transpose()*e << std::endl;

        if (H.determinant() < 1e-6) {
            std::cout << "Degenerate Hessian, stopping early.\n";
            // break;
        }

        //std::cout << "Iteration " << iter << " Pose: \n" << x << "\n\n";
        //cout<< (R * p + t - q).transpose()*(R * p + t - q) << endl;
        
        // if (dx.norm() < 1e-6) break;  // Convergence condition
    }

    moved_data = getRotation(x(3), x(4), x(5)) * true_data;
    moved_data.colwise() += x.head(3);
    indices = correspondences(moved_data, true_data);
    Render(moved_data, true_data, indices);
}
