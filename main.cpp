#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
//
// Created by Liu on 2021/1/4.
//
#include "face_weights.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <utility>
#include <cmath>
#include <omp.h>
#include <vector>
#include <chrono>
#include <string.h>
#include <dirent.h>

using namespace std;
using namespace cv;



#define TIME_START start = std::chrono::steady_clock::now(); \
duration = 0L;
#define TIME_END end = std::chrono::steady_clock::now();\
duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();\
std::cout << " Duration: " << duration << "ms" << std::endl;



/**
 * 传入OpenCV读入的img，返回0f-1f按RGB顺序的float数组
 */
float * GetRGB(const Mat& img){
    //通道分离
    Mat *bgr = new Mat[3];
    split(img, bgr);

    //bgr->rgb
    const int img_size = bgr->rows * bgr->cols;
    auto *rgb = new float[3 * img_size];
    auto *array = new unsigned char[3 * img_size];
    for (int i = 0; i < 3; ++i) {
        if (bgr[i].isContinuous()) {
            array = bgr[i].data;
        }
        for (int j = 0; j < img_size; ++j) {
            rgb[(2 - i) * img_size + j] = (float)array[j] / 255.0f;
        }
    }
    return rgb;
}

float calculate(const float *inMat, const conv_param &convParam, int kernel, int row, int col, int row_size){
    float sum = 0;
    row = row * convParam.stride - convParam.pad;
    col = col * convParam.stride - convParam.pad;
    for (int i = 0; i < convParam.in_channels; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                int temp_row = row + j;
                int temp_col = col + k;
                float temp_val = 0;
                if (temp_col >= 0 && temp_row >= 0 && temp_col < row_size && temp_row < row_size){
                    temp_val = inMat[i*row_size*row_size + temp_row*row_size + temp_col];
                }
                sum += temp_val * convParam.p_weight[kernel*9*convParam.in_channels + i*9 + j*3 + k];
            }
        }
    }
    sum += convParam.p_bias[kernel];
    return sum;
}

/**
 * Do the convolution and return the output array
 * @param inMat input matrix
 * @param convParam param of convolution
 * @param mat_row rows of input
 */
float *Convolution(const float *inMat, const conv_param &convParam, int row_size){
    const int out_row = (row_size + convParam.pad * 2 - convParam.kernel_size) / convParam.stride + 1;
    const int out_size = out_row * out_row;
    auto *out = new float[out_size * convParam.out_channels];
#pragma omp parallel for
    for (int i = 0; i < convParam.out_channels; ++i) {
        for (int j = 0; j < out_row; ++j) {
            for (int k = 0; k < out_row; ++k) {
                out[i*out_size + j*out_row + k] = calculate(inMat,convParam,i,j,k,row_size);
            }
        }
    }
    return out;
}

float *Relu(const float *inMat, int size){
    auto *out = new float[size];
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        if (inMat[i] < 0){
            out[i] = 0;
        } else {
            out[i] = inMat[i];
        }
    }
    return out;
}

inline float max_x(float a, float b){
    if (a > b)
        return a;
    return b;
}

float *MaxPool(float * inMat, int channel, int row_size){
    const int out_row = row_size / 2;
    auto *out = new float[channel * out_row * out_row];
#pragma omp parallel for
    for (int i = 0; i < channel; ++i) {
        for (int j = 0; j < out_row; ++j) {
            for (int k = 0; k < out_row; ++k) {
                out[i*out_row*out_row + j*out_row + k] = max_x(max_x(inMat[i*row_size*row_size + j*2*row_size + k*2], inMat[i*row_size*row_size + j*2*row_size + k*2+1]), max_x(inMat[i*row_size*row_size + (j*2+1)*row_size + k*2],inMat[i*row_size*row_size + (j*2+1)*row_size + k*2+1]));
            }
        }
    }
    return out;
}

float *full_connect(const float *inMat, const fc_param &fcParam){
    auto *out = new float[2]();
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2048; ++j) {
            out[i] += inMat[j] * fcParam.p_weight[i*2048 + j];
        }
        out[i] += fcParam.p_bias[i];
    }
    return out;
}

float *Softmax(float *inMat){
    auto *out = new float [2];
    for (int i = 0; i < 2; ++i) {
        out[i] = exp(inMat[i]) / (exp(inMat[0]) + exp(inMat[1]));
    }
    return out;
}

float *GetFaceScore(const Mat& img){
    //得到img的rgb数组
    float *rgb = GetRGB(img);

    //get trained conv_params
    extern conv_param conv_params[3];
    //conv_0 from 3*128*128 to 16*64*64
    float *out_0 = Convolution(rgb,conv_params[0],128);
    //Relu_0 keep
    float *out_1 = Relu(out_0, 16*64*64);
    //MaxPool_0 from 16*64*64 to 16*32*32
    float *out_2 = MaxPool(out_1, 16,64);
    delete[] out_0;
    delete[] out_1;

    //conv_1 from 16*32*32 to 32*30*30
    out_0 = Convolution(out_2, conv_params[1],32);
    //Relu_1 keep
    out_1 = Relu(out_0, 32*30*30);
    //MaxPool_1 from 32*30*30 to 32*15*15
    float *out_3 = MaxPool(out_1, 32,30);
    delete[] out_0;
    delete[] out_1;
    delete[] out_2;

    //conv_2 from 32*15*15 to 32*8*8
    out_0 = Convolution(out_3, conv_params[2],15);
    //Relu_2 keep
    out_1 = Relu(out_0, 32*8*8);

    //get fc_params
    extern fc_param fc_params[1];
    //full_connect from 2048*1 to 2*1
    out_2 = full_connect(out_1, fc_params[0]);
    //Softmax
    float *out_final = Softmax(out_2);
    delete[] out_0;
    delete[] out_1;
    delete[] out_2;
    return out_final;
}

void GetFileNames(string path,vector<string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        cout<<"Folder doesn't Exist!"<<endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
}

int main() {
    //图像读取
    const string imgPath = R"(/home/see/下载/CS205_Final_Project-master/picture)";
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto duration = 0L;
    vector<string> files;
    vector<string> file_name;
    GetFileNames( imgPath, files);
    //程序预热
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < files.size(); ++i) {
            Mat img = imread(files[i]);
            float *out = GetFaceScore(img);
            delete[] out;
        }
    }

    TIME_START
    for (int i = 0; i < files.size(); ++i) {
        Mat img = imread(files[i]);
        float *out = GetFaceScore(img);
        cout << "--------------------------" << endl;
        cout << "File name: " << files[i] << endl;
        cout << "bg score: " << out[0] << endl;
        cout << "face score: " << out[1] << endl;
        cout << "--------------------------" << endl;
        delete[] out;
    }
    cout << "Total " << files.size() << " files,";
    TIME_END
}



#pragma clang diagnostic pop