// Auto-generated linear-model C inference
#pragma once

static const float WEIGHTS[] = { 5000.0000000000f };
static const float BIAS = 25000.0000000000f;

float predict_model(const float *x, int n_features) {
    float s = BIAS;
    for (int i = 0; i < n_features; i++) {
        s += WEIGHTS[i] * x[i];
    }
    return s;
}