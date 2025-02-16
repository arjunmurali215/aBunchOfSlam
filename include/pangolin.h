#ifndef PANGOLIN_H
#define PANGOLIN_H

#include <pangolin/pangolin.h>
#include "eigen.h"

void DrawPoints(const Points &pointsm, float r, float g, float b, float size);
void DrawCorrespondences(const Points &P, const Points &Q, const std::vector<int> &correspondences);
void Render(const Points &moved_data, const Points &true_data, const std::vector<int> &correspondences);

#endif // PANGOLIN_H
