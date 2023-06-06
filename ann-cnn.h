/////////////////////////////////////////////////////////////////////////////////////////////
/*
 *  ann-cnn.h
 *  Home Page :http://www.1234998.top
 *  Created on: Mar 29, 2023
 *  Author: M
 */
/////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _INC_ANN_CNN_H_
#define _INC_ANN_CNN_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h >
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

#define DEPTHWISE_POINTWISE_CONVOLUTION
#define NULL ((void *)0)
#define MINFLOAT_POSITIVE_NUMBER (1.175494351E-38)	// 最小正规化数
#define MAXFLOAT_POSITIVE_NUMBER (3.402823466E+38)	// 最大正规化数
#define MAXFLOAT_NEGATIVE_NUMBER (-1.175494351E-38) // 负数的最大值
#define MINFLOAT_NEGATIVE_NUMBER (-3.402823466E+38) // 负数的最小值

//上溢或下溢
#define IsFloatOverflow(x) (x > MAXFLOAT_POSITIVE_NUMBER || x < MINFLOAT_NEGATIVE_NUMBER)
// #define IsFloatUnderflow(x)(x == 0)

#define PRINTFLAG_WEIGHT 0
#define PRINTFLAG_GRADS 1
#define PRINTFLAG_FORMAT "%9.6f"

#define NEURALNET_CNN_WEIGHT_FILE_NAME "_cnn.w"
#define NEURALNET_CNN_FILE_NAME "_cnn.csv"
#define NEURALNET_ERROR_BASE 10001

#define PLATFORM_WINDOWS
// #define PLATFORM_STM32
#define __DEBUG__LOG__

/////////////////////////////////////////////////////////////////////////////////////////////
#ifdef PLATFORM_STM32
#include "usart.h"
#include "octopus.h"
// #include "arm_math.h"
// #define DSP_SQRT_FUNCTION arm_sqrt_f32
#define CNNLOG debug_printf
#define LOG debug_printf
#define FUNCTIONNAME __func__
#define DLLEXPORT
#else
#include <windows.h>
typedef double float32_t;
typedef int uint32_t;
typedef short uint16_t;
typedef bool bool_t;
typedef unsigned char uint8_t;

#define CNNLOG printf
#define LOG printf
#define LOGLOG printf
#define FUNCTIONNAME __func__
#define DLLEXPORT _declspec(dllexport)
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __DEBUG__LOG__
#define LOGINFO(format, ...)   LOGLOG("[Infor][%-9.9s][%s]:" format "\n", __FILE__, __func__, ##__VA_ARGS__)
#define LOGINFOR(format, ...)  LOGLOG("[Infor][%-9.9s][Line:%04d][%s]:" format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#define LOGERROR(format, ...)  LOGLOG("[Error][%-9.9s][Line:%04d][%s]:" format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
#define LOGINFO(format, ...)  
#define LOGINFOR(format, ...)
#define LOGERROR(format, ...)
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
	Optm_Gradient,
	Optm_Stochastic,
	Optm_BatchGradient,
	Optm_Momentum,
	Optm_Nesterov,
	Optm_Adagrad,
	Optm_Adadelta,
	Optm_RMSprop,
	Optm_Adam,
	Optm_Adamax,
	Optm_Nadam,
	Optm_AdaBound,
	Optm_AdaBelief,
	Optm_AdaMod,
	Optm_AdaShift,
	Optm_AdaSOM,
	Optm_AdaHessian,
	Optm_AdaLAMB,
	Optm_AdaLIPS,
	Optm_AdaRAdam
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

typedef struct ANN_CNN_Response
{
	TPTensor filterWeight;
	TPTensor filterGrads;
	float32_t l2_decay_rate;
	float32_t l1_decay_rate;
	void (*fillZero)(TPTensor PTensor);
	void (*free)(TPTensor PTensor);
} TResponse, * TPResponse;

typedef struct ANN_CNN_Prediction
{
	uint16_t labelIndex;
	float32_t likeliHood;
} TPrediction, * TPPrediction;

typedef struct ANN_CNN_Volume
{
	uint16_t _w, _h, _depth;
	TPTensor weight;
	TPTensor weight_d;
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
	TPResponse* (*getWeightsAndGrads)(struct ANN_CNN_ConvolutionLayer* PConvLayer);
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
	TPResponse* (*getWeightsAndGrads)(struct ANN_CNN_PoolLayer* PPoolLayer);
} TPoolLayer, * TPPoolLayer;

typedef struct ANN_CNN_ReluLayer
{
	TLayer layer;
	void (*init)(struct ANN_CNN_ReluLayer* PLayer, TPLayerOption PLayerOption);
	void (*free)(struct ANN_CNN_ReluLayer* PLayer);
	void (*forward)(struct ANN_CNN_ReluLayer* PLayer);
	void (*backward)(struct ANN_CNN_ReluLayer* PLayer);
	float32_t(*computeLoss)(struct ANN_CNN_ReluLayer* PLayer, int Y);
	TPResponse* (*getWeightsAndGrads)(struct ANN_CNN_ReluLayer* PReluLayer);
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
	TPResponse* (*getWeightsAndGrads)(struct ANN_CNN_FullyConnectedLayer* PFullyConnLayer);
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
	TPResponse* (*getWeightsAndGrads)(struct ANN_CNN_SoftmaxLayer* PSoftmaxLayer);
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
	TPResponse* pResponseResults;
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
double GenerateRandomNumber();
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
DLLEXPORT void NeuralNetInit_Cifar10_11(void);

time_t GetTimestamp(void);

#endif /* _INC_ANN_CNN_H_ */
