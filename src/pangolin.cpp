#include "../include/pangolin.h"

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

void DrawCorrespondences(const Points &P, const Points &Q, const std::vector<int> &correspondences) {
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

void Render(const Points &moved_data, const Points &true_data, const std::vector<int> &correspondences) {
    pangolin::CreateWindowAndBind("3D Eigen Points Visualization", 640, 480);
    glEnable(GL_DEPTH_TEST);
    
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 500, 500, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAt(0, 0, 20, 0, 0, 0, pangolin::AxisY)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        DrawPoints(true_data, 1.0f, 0.0f, 0.0f, 3.0f);
        DrawPoints(moved_data, 0.0f, 0.0f, 1.0f, 5.0f);
        DrawCorrespondences(moved_data, true_data, correspondences);

        pangolin::FinishFrame();
    }
}
