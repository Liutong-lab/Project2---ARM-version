//
// Created by SeeE on 2021/1/4.
//

#ifndef FINAL_PROJECT_FACE_WEIGHTS_H
#define FINAL_PROJECT_FACE_WEIGHTS_H

struct conv_param {
    int pad;
    int stride;
    int kernel_size;
    int in_channels;
    int out_channels;
    float* p_weight;
    float* p_bias;
} ;

struct fc_param {
    int in_features;
    int out_features;
    float* p_weight;
    float* p_bias;
} ;

#endif //FINAL_PROJECT_FACE_WEIGHTS_H
