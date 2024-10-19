/////////////////////////////////////////////////////////////////////////////////////////////
/*
 *  ann-cnn.h
 *  Home Page: http://www.1234998.top
 *  Created on: March 29, 2023
 *  Author: M
 */
/////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _INC_ANN_CNN_H_ // Prevent multiple inclusions of header file
#define _INC_ANN_CNN_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

#define DEPTHWISE_POINTWISE_CONVOLUTION // Define depthwise separable convolution
#define NULL ((void *)0) // Define null pointer
#define MINFLOAT_POSITIVE_NUMBER (1.175494351E-38) // Minimum normalized positive number
#define MAXFLOAT_POSITIVE_NUMBER (3.402823466E+38) // Maximum normalized positive number
#define MAXFLOAT_NEGATIVE_NUMBER (-1.175494351E-38) // Maximum normalized negative number
#define MINFLOAT_NEGATIVE_NUMBER (-3.402823466E+38) // Minimum normalized negative number

// Check for float overflow or underflow
#define IsFloatOverflow(x) (x > MAXFLOAT_POSITIVE_NUMBER || x < MINFLOAT_NEGATIVE_NUMBER)
// #define IsFloatUnderflow(x)(x == 0)

#define PRINTFLAG_WEIGHT 0 // Flag for printing weights
#define PRINTFLAG_GRADS 1 // Flag for printing gradients
#define PRINTFLAG_FORMAT "%9.6f" // Print format

#define NEURALNET_CNN_WEIGHT_FILE_NAME "_cnn.w" // Neural network weight file name
#define NEURALNET_CNN_FILE_NAME "_cnn.csv" // Neural network file name
#define NEURALNET_ERROR_BASE 10001 // Base error number for neural network

#define PLATFORM_WINDOWS // Define platform as Windows
//#define PLATFORM_WINDOWS_DLL // Define as Windows DLL

// #define PLATFORM_STM32 // Define as STM32 platform

#define __DEBUG__LOG__ // Enable debug logging

/////////////////////////////////////////////////////////////////////////////////////////////
#ifdef PLATFORM_STM32 // Includes and definitions for STM32 platform
#include "usart.h"
#include "octopus.h"
// #include "arm_math.h"
// #define DSP_SQRT_FUNCTION arm_sqrt_f32
#define CNNLOG debug_printf // Use debug print function
#define LOG debug_printf // Use debug print function
#define FUNCTIONNAME __func__ // Get current function name
#define DLLEXPORT // DLL export definition

#else // Definitions for other platforms (e.g., Windows)

#include <windows.h>
typedef double float32_t; // Define 32-bit float type
typedef int uint32_t; // Define 32-bit unsigned integer type
typedef short uint16_t; // Define 16-bit unsigned integer type
typedef bool bool_t; // Define boolean type
typedef unsigned char uint8_t; // Define 8-bit unsigned integer type

#define CNNLOG printf // Use standard print function
#define LOG printf // Use standard print function
#define LOGLOG printf // Use standard print function
#define FUNCTIONNAME __func__ // Get current function name

#ifdef PLATFORM_WINDOWS_DLL // If Windows DLL is defined
#define DLLEXPORT _declspec(dllexport) // DLL export
#else
#define DLLEXPORT // No export
#endif

#endif

/////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __DEBUG__LOG__ // If debug logging is enabled
#define LOGINFO(format, ...)   LOGLOG("[Info][%-9.9s][%s]:" format "\n", __FILE__, __func__, ##__VA_ARGS__) // Info log
#define LOGINFOR(format, ...)  LOGLOG("[Info][%-9.9s][Line:%04d][%s]:" format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__) // Line info log
#define LOGERROR(format, ...)  LOGLOG("[Error][%-9.9s][Line:%04d][%s]:" format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__) // Error log
#else
#define LOGINFO(format, ...) // Disable info log
#define LOGINFOR(format, ...) // Disable line info log
#define LOGERROR(format, ...) // Disable error log
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
//定义网络各层类型
typedef enum LayerType
{
    Layer_Type_Input,
    Layer_Type_Convolution,
    Layer_Type_ReLu,
    Layer_Type_Pool,
    Layer_Type_FullyConnection,
    Layer_Type_SoftMax,
    Layer_Type_None
} TLayerType;


/////////////////////////////////////////////////////////////////////////////////////////////
/*深度学习优化算法：
 00. 梯度下降法（Gradient Descent）
 01. 随机梯度下降法（Stochastic Gradient Descent）
 02. 批量梯度下降法（Batch Gradient Descent）
 03. 动量法（Momentum）
 04. Nesterov加速梯度（Nesterov Accelerated Gradient）
 05. 自适应梯度算法（Adagrad）
 06. 自适应矩估计算法（Adadelta）
 07. 自适应矩估计算法（RMSprop）
 08. 自适应矩估计算法（Adam）
 09. 自适应矩估计算法（Adamax）
 10. 自适应矩估计算法（Nadam）
 11. 自适应学习率优化算法（AdaBound）
 12. 自适应学习率优化算法（AdaBelief）
 13. 自适应学习率优化算法（AdaMod）
 14. 自适应学习率优化算法（AdaShift）
 15. 自适应学习率优化算法（AdaSOM）
 16. 自适应学习率优化算法（AdaHessian）
 17. 自适应学习率优化算法（AdaLAMB）
 18. 自适应学习率优化算法（AdaLIPS）
 19. 自适应学习率优化算法（AdaRAdam）
 */
typedef enum OptimizeMethod
{
    Optm_Gradient, // 梯度下降法
    Optm_Stochastic, // 随机梯度下降法
    Optm_BatchGradient, // 批量梯度下降法
    Optm_Momentum, // 动量法
    Optm_Nesterov, // Nesterov加速梯度
    Optm_Adagrad, // 自适应梯度算法
    Optm_Adadelta, // 自适应矩估计算法
    Optm_RMSprop, // 自适应矩估计算法
    Optm_Adam, // 自适应矩估计算法
    Optm_Adamax, // 自适应矩估计算法
    Optm_Nadam, // 自适应矩估计算法
    Optm_AdaBound, // 自适应学习率优化算法
    Optm_AdaBelief, // 自适应学习率优化算法
    Optm_AdaMod, // 自适应学习率优化算法
    Optm_AdaShift, // 自适应学习率优化算法
    Optm_AdaSOM, // 自适应学习率优化算法
    Optm_AdaHessian, // 自适应学习率优化算法
    Optm_AdaLAMB, // 自适应学习率优化算法
    Optm_AdaLIPS, // 自适应学习率优化算法
    Optm_AdaRAdam // 自适应学习率优化算法
} TOptimizeMethod;

/////////////////////////////////////////////////////////////////////////////////////////////

typedef void (*CnnFunction)(int x, int y, int depth, float32_t v);
typedef void (*VolumeFunction)(int x, int y, int depth, float32_t v);
typedef double (*VolumeFunction_get)(int x, int y, int depth, float32_t v);

typedef struct ANN_CNN_MaxMin
{
    float32_t max;
    float32_t min;
} TMaxMin, * TPMaxMin;

typedef struct ANN_CNN_Tensor
{
    float32_t* buffer;
    uint32_t length;
} TTensor, * TPTensor;

typedef struct ANN_CNN_Parameter
{
    TPTensor filterWeight;
    TPTensor filterGrads;
    float32_t l2_decay_rate;
    float32_t l1_decay_rate;
    void (*fillZero)(TPTensor PTensor);
    void (*free)(TPTensor PTensor);
} TParameters, * TPParameters;

typedef struct ANN_CNN_Prediction
{
    uint16_t labelIndex;
    float32_t likeliHood;
} TPrediction, * TPPrediction;

typedef struct ANN_CNN_Volume
{
    uint16_t _w, _h, _depth;
    TPTensor weight;
    TPTensor grads;
    void (*init)(struct ANN_CNN_Volume* PVolume, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias);
    void (*setValue)(struct ANN_CNN_Volume* PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
    void (*addValue)(struct ANN_CNN_Volume* PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
    float32_t(*getValue)(struct ANN_CNN_Volume* PVolume, uint16_t X, uint16_t Y, uint16_t Depth);
    void (*setGradValue)(struct ANN_CNN_Volume* PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
    void (*addGradValue)(struct ANN_CNN_Volume* PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
    float32_t(*getGradValue)(struct ANN_CNN_Volume* PVolume, uint16_t X, uint16_t Y, uint16_t Depth);
    void (*fillZero)(TPTensor PTensor);
    void (*fillGauss)(TPTensor PTensor);
    void (*flip)(struct ANN_CNN_Volume* PVolume);
    void (*print)(struct ANN_CNN_Volume* PVolume, uint8_t wg);
    void (*free)(TPTensor PTensor);
} TVolume, * TPVolume;

typedef struct ANN_CNN_Filters
{
    uint16_t _w, _h, _depth;
    uint16_t filterNumber;
    TPVolume* volumes;
    void (*init)(TPVolume PVolume, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias);
    void (*free)(struct ANN_CNN_Filters* PFilters);
} TFilters, * TPFilters;

typedef struct ANN_CNN_LayerOption
{
    uint16_t LayerType;
    uint16_t in_w;
    uint16_t in_h;
    uint16_t in_depth;
    uint16_t out_w;
    uint16_t out_h;
    uint16_t out_depth;
    uint16_t filter_w;
    uint16_t filter_h;
    uint16_t filter_depth;
    uint16_t filter_number;
    uint16_t stride;
    uint16_t padding;
    uint16_t neurons;
    uint16_t group_size;
    float32_t l1_decay_rate;
    float32_t l2_decay_rate;
    float32_t drop_prob;
    float32_t bias;
} TLayerOption, * TPLayerOption;

typedef struct ANN_CNN_Layer
{
    uint16_t LayerType;
    uint16_t in_w;
    uint16_t in_h;
    uint16_t in_depth;
    TPVolume in_v;
    uint16_t out_w;
    uint16_t out_h;
    uint16_t out_depth;
    TPVolume out_v;
} TLayer, * TPLayer;

typedef struct ANN_CNN_InputLayer
{
    TLayer layer;
    void (*init)(struct ANN_CNN_InputLayer* PLayer, TPLayerOption PLayerOption);
    void (*free)(struct ANN_CNN_InputLayer* PLayer);
    void (*forward)(struct ANN_CNN_InputLayer* PLayer, TVolume* Volume);
    void (*backward)(struct ANN_CNN_InputLayer* PLayer);
    float32_t(*computeLoss)(struct ANN_CNN_InputLayer* PLayer, int Y);
} TInputLayer, * TPInputLayer;

typedef struct ANN_CNN_ConvolutionLayer
{
    TLayer layer;
    TPFilters filters;
    uint16_t stride;
    uint16_t padding;
    float32_t l1_decay_rate;
    float32_t l2_decay_rate;
    float32_t bias;
    TPVolume biases;
    void (*init)(struct ANN_CNN_ConvolutionLayer* PLayer, TPLayerOption PLayerOption);
    void (*free)(struct ANN_CNN_ConvolutionLayer* PLayer);
    void (*forward)(struct ANN_CNN_ConvolutionLayer* PLayer);
    void (*backward)(struct ANN_CNN_ConvolutionLayer* PLayer);
    float32_t(*computeLoss)(struct ANN_CNN_ConvolutionLayer* PLayer, int Y);
    TPParameters* (*getWeightsAndGrads)(struct ANN_CNN_ConvolutionLayer* PConvLayer);
} TConvLayer, * TPConvLayer;

typedef struct ANN_CNN_PoolLayer
{
    TLayer layer;
    uint16_t stride;
    uint16_t padding;
    TPVolume filter;
    TPVolume switchxy;
    void (*init)(struct ANN_CNN_PoolLayer* PLayer, TPLayerOption PLayerOption);
    void (*free)(struct ANN_CNN_PoolLayer* PLayer);
    void (*forward)(struct ANN_CNN_PoolLayer* PLayer);
    void (*backward)(struct ANN_CNN_PoolLayer* PLayer);
    float32_t(*computeLoss)(struct ANN_CNN_PoolLayer* PLayer, int Y);
    TPParameters* (*getWeightsAndGrads)(struct ANN_CNN_PoolLayer* PPoolLayer);
} TPoolLayer, * TPPoolLayer;

typedef struct ANN_CNN_ReluLayer
{
    TLayer layer;
    void (*init)(struct ANN_CNN_ReluLayer* PLayer, TPLayerOption PLayerOption);
    void (*free)(struct ANN_CNN_ReluLayer* PLayer);
    void (*forward)(struct ANN_CNN_ReluLayer* PLayer);
    void (*backward)(struct ANN_CNN_ReluLayer* PLayer);
    float32_t(*computeLoss)(struct ANN_CNN_ReluLayer* PLayer, int Y);
    TPParameters* (*getWeightsAndGrads)(struct ANN_CNN_ReluLayer* PReluLayer);
} TReluLayer, * TPReluLayer;

typedef struct ANN_CNN_FullyConnectedLayer //FullyConnectedLayer
{
    TLayer layer;
    TPFilters filters;
    TPVolume biases;
    float32_t l1_decay_rate;
    float32_t l2_decay_rate;
    float32_t bias;
    void (*init)(struct ANN_CNN_FullyConnectedLayer* PLayer, TPLayerOption PLayerOption);
    void (*free)(struct ANN_CNN_FullyConnectedLayer* PLayer);
    void (*forward)(struct ANN_CNN_FullyConnectedLayer* PLayer);
    void (*backward)(struct ANN_CNN_FullyConnectedLayer* PLayer);
    float32_t(*computeLoss)(struct ANN_CNN_FullyConnectedLayer* PLayer, int Y);
    TPParameters* (*getWeightsAndGrads)(struct ANN_CNN_FullyConnectedLayer* PFullyConnLayer);
} TFullyConnLayer, * TPFullyConnLayer;

typedef struct ANN_CNN_SoftmaxLayer
{
    TLayer layer;
    TPTensor exp;
    uint16_t expected_value;
    void (*init)(struct ANN_CNN_SoftmaxLayer* PLayer, TPLayerOption PLayerOption);
    void (*free)(struct ANN_CNN_SoftmaxLayer* PLayer);
    void (*forward)(struct ANN_CNN_SoftmaxLayer* PLayer);
    void (*backward)(struct ANN_CNN_SoftmaxLayer* PLayer);
    float32_t(*computeLoss)(struct ANN_CNN_SoftmaxLayer* PLayer);
    TPParameters* (*getWeightsAndGrads)(struct ANN_CNN_SoftmaxLayer* PSoftmaxLayer);
} TSoftmaxLayer, * TPSoftmaxLayer;

typedef struct ANN_CNN_LearningParameter
{
    uint16_t optimize_method;
    uint16_t batch_size;
    float32_t learning_rate;
    float32_t l1_decay_rate;
    float32_t l2_decay_rate;
    float32_t eps;
    float32_t beta1; // for adam
    float32_t beta2; // for adam
    float32_t momentum;
    float32_t bias;
} TLearningParameter, * TPLearningParameter;

typedef struct ANN_CNN_Learning
{
    bool_t trainningGoing;
    uint16_t data_type;
    float32_t sum_cost_loss;
    float32_t sum_l1_decay_loss;
    float32_t sum_l2_decay_loss;
    float32_t trainingAccuracy;
    float32_t testingAccuracy;

    uint32_t labelIndex;
    uint32_t sampleCount;
    uint32_t epochCount;
    uint32_t batchCount;
    uint32_t iterations;
    uint32_t trinning_dataset_index;
    uint32_t datasetTotal;
    uint32_t testing_dataset_index;

    uint16_t responseCount;
    TPParameters* pResponseResults;
    uint16_t predictionCount;
    TPPrediction* pPredictions;
    TPTensor* grads_sum1;
    TPTensor* grads_sum2;
    uint16_t grads_sum_count;
    bool_t underflow;
    bool_t overflow;
    bool_t one_by_one;
    bool_t batch_by_batch;
    bool_t trainingSaving;
    bool_t randomFlip;
} TLearningResult, * TPLearningResult;

typedef struct ANN_CNN_NeuralNet
{
    char* name;
    TPLayer* layers;
    uint16_t depth;
    time_t fwTime;
    time_t bwTime;
    time_t optimTime;
    time_t totalTime;
    TLearningParameter trainningParam;
    TLearningResult trainning;

    void (*init)(struct ANN_CNN_NeuralNet* PNeuralNet, TPLayerOption PLayerOption);
    void (*free)(struct ANN_CNN_NeuralNet* PNeuralNet);
    void (*forward)(struct ANN_CNN_NeuralNet* PNeuralNet, TPVolume PVolume);
    void (*backward)(struct ANN_CNN_NeuralNet* PNeuralNet);
    void (*getCostLoss)(struct ANN_CNN_NeuralNet* PNeuralNet, float32_t* CostLoss);
    void (*getWeightsAndGrads)(struct ANN_CNN_NeuralNet* PNeuralNet);
    void (*getPredictions)(struct ANN_CNN_NeuralNet* PNeuralNet);
    void (*getMaxPrediction)(struct ANN_CNN_NeuralNet* PNeuralNet, TPPrediction PPrediction);
    void (*train)(struct ANN_CNN_NeuralNet* PNeuralNet, TPVolume PVolume);
    void (*predict)(struct ANN_CNN_NeuralNet* PNeuralNet, TPVolume PVolume);
    void (*printWeights)(struct ANN_CNN_NeuralNet* PNeuralNet, uint16_t LayerIndex, uint8_t InOut);
    void (*printTensor)(char* Name, TPTensor PTensor);
    void (*printGradients)(struct ANN_CNN_NeuralNet* PNeuralNet, uint16_t LayerIndex, uint8_t InOut);
    void (*printTrainningInfor)(struct ANN_CNN_NeuralNet* PNeuralNet);
    void (*printNetLayersInfor)(struct ANN_CNN_NeuralNet* PNeuralNet);

    void (*saveWeights)(struct ANN_CNN_NeuralNet* PNeuralNet);
    void (*loadWeights)(struct ANN_CNN_NeuralNet* PNeuralNet);
    void (*saveNet)(struct ANN_CNN_NeuralNet* PNeuralNet);
    void (*loadNet)(struct ANN_CNN_NeuralNet* PNeuralNet);
    char* (*getName)(TLayerType LayerType);
} TNeuralNet, * TPNeuralNet;

////////////////////////////////////////////////////////////////////////////////////////////////////////
float32_t GenerateRandomNumber();
////////////////////////////////////////////////////////////////////////////////////////////////////////
void TensorFillZero(TPTensor PTensor);
void TensorFillGauss(TPTensor PTensor);
TPMaxMin TensorMaxMin(TPTensor PTensor);
void TensorSave(FILE* pFile, TPTensor PTensor);
TPVolume MakeVolume(uint16_t W, uint16_t H, uint16_t Depth);
void VolumeInit(TPVolume PVolume, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias);
void VolumeFree(TPVolume PVolume);
void VolumeSetValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
void VolumeAddValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
float32_t VolumeGetValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth);
void VolumeSetGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
void VolumeAddGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
float32_t VolumeGetGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth);
void  VolumeFlip(TPVolume PVolume);
void VolumePrint(TPVolume PVolume, uint8_t wg);
TPFilters MakeFilters(uint16_t W, uint16_t H, uint16_t Depth, uint16_t FilterNumber);
void FilterVolumesFree(TPFilters PFilters);
void NeuralNetPrintLayersInfor(TPNeuralNet PNeuralNet);


DLLEXPORT TPNeuralNet NeuralNetCNNCreate(char* name);
DLLEXPORT int NeuralNetAddLayer(TPNeuralNet PNeuralNet, TLayerOption LayerOption);
DLLEXPORT void NeuralNetInit(TPNeuralNet PNeuralNet, TPLayerOption PLayerOption);
DLLEXPORT void NeuralNetFree(TPNeuralNet PNeuralNet);
DLLEXPORT void NeuralNetForward(TPNeuralNet PNeuralNet, TPVolume PVolume);
DLLEXPORT void NeuralNetBackward(TPNeuralNet PNeuralNet);
DLLEXPORT void NeuralNetGetWeightsAndGrads(TPNeuralNet PNeuralNet);
DLLEXPORT void NeuralNetComputeCostLoss(TPNeuralNet PNeuralNet, float32_t* CostLoss);
DLLEXPORT void NeuralNetUpdatePrediction(TPNeuralNet PNeuralNet);
DLLEXPORT void NeuralNetSaveWeights(TPNeuralNet PNeuralNet);
DLLEXPORT void NeuralNetLoadWeights(TPNeuralNet PNeuralNet);
DLLEXPORT void NeuralNetGetMaxPrediction(TPNeuralNet PNeuralNet, TPPrediction PPrediction);
DLLEXPORT void NeuralNetTrain(TPNeuralNet PNeuralNet, TPVolume PVolume);
DLLEXPORT char* NeuralNetGetLayerName(TLayerType LayerType);
/// @brief ////////////////////////////////////////////////////////////////////////////////////////////////
/// @return

time_t GetTimestamp(void);
double GenerateGaussRandom(void);
double GenerateGaussRandom1(void);
double GenerateGaussRandom2(void);
#endif /* _INC_ANN_CNN_H_ */
