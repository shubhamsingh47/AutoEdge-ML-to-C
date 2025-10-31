// Auto-generated multiclass logistic C inference
#pragma once

#include <math.h>

static const float W_0[] = { -0.3969916973f, 0.9604100304f, -2.3740166903f, -1.0030759207f };
static const float B_0 = 9.0320635500f;
static const float W_1[] = { 0.5127338595f, -0.2533963340f, -0.2152668653f, -0.7691621940f };
static const float B_1 = 1.8416241163f;
static const float W_2[] = { -0.1157421622f, -0.7070136964f, 2.5892835555f, 1.7722381147f };
static const float B_2 = -10.8736876664f;

int predict_model(const float *x, int n_features) {
    float scores[3];
    scores[0] = B_0;
    for (int i = 0; i < n_features; i++)
        scores[0] += W_0[i] * x[i];
    scores[1] = B_1;
    for (int i = 0; i < n_features; i++)
        scores[1] += W_1[i] * x[i];
    scores[2] = B_2;
    for (int i = 0; i < n_features; i++)
        scores[2] += W_2[i] * x[i];
    float max_s = scores[0];
    int max_i = 0;
    for (int c = 1; c < 3; c++) {
        if (scores[c] > max_s) { max_s = scores[c]; max_i = c; }
    }
    return max_i;
}