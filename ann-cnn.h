///*
// * ann-cnn.h
// *
// *  Created on: Mar 29, 2023
// *      Author: M
// */

#ifndef _INC_ANN_CNN_H_
#define _INC_ANN_CNN_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
// cifar-10
#define CIFAR10_IMAGE_NUM_TOTAL 60000
#define CIFAR10_IMAGE_NUM_TEST 10000
#define CIFAR10_IMAGE_LABEL_NUM 20
#define CIFAR10_IMAGE_WIDTH 32
#define CIFAR10_IMAGE_HEIGHT 32
#define CIFAR10_IMAGE_SIZE (3072 + 1) // 32X32X3+1
// Minimum negative number
#define MINI_NEGATIVE_NUMBER (-2147483648.00)
#define NULL ((void *)0)

#define PLATFORM_WINDOWS
// #define PLATFORM_STM32
#define __DEBUG__LOG__

#ifdef PLATFORM_STM32
#include "usart.h"
#include "octopus.h"
// #include "arm_math.h"
// #define DSP_SQRT_FUNCTION arm_sqrt_f32
#define CNNLOG debug_printf
#define LOG debug_printf
#define FUNCTIONNAME __func__
#else
typedef double float32_t;
typedef int uint32_t;
typedef short uint16_t;
typedef unsigned char uint8_t;
// #define DSP_SQRT_FUNCTION sqrt
#define CNNLOG printf
#define LOG printf
#define FUNCTIONNAME __func__
#endif

#ifdef __DEBUG__LOG__
#define LOGINFO(format, ...) LOG("[Infor][%-9.9s][Line:%04d][%s]:" format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#define LOGERROR(format, ...) LOG("[Error][%-9.9s][Line:%04d][%s]:" format "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
#define LOGINFO(format, ...)
#define LOGERROR(format, ...)
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

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

typedef enum DataSetType
{
	Cifar10,
	Cifar100
} TDataSetType;
/////////////////////////////////////////////////////////////////////////////////////////////
/*深度学习优化算法：
 0. 梯度下降法（Gradient Descent）
 1. 随机梯度下降法（Stochastic Gradient Descent）
 2. 批量梯度下降法（Batch Gradient Descent）
 3. 动量法（Momentum）
 4. Nesterov加速梯度（Nesterov Accelerated Gradient）
 5. 自适应梯度算法（Adagrad）
 6. 自适应矩估计算法（Adadelta）
 7. 自适应矩估计算法（RMSprop）
 8. 自适应矩估计算法（Adam）
 9. 自适应矩估计算法（Adamax）
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

typedef struct ANN_CNN_Tensor
{
	float32_t *buffer;
	uint32_t length;
} TTensor, *TPTensor;

typedef struct ANN_CNN_Response
{
	TPTensor filter;
	TPTensor filterGrads;
	float32_t l2_decay_mul;
	float32_t l1_decay_mul;
	void (*fillZero)(TPTensor PTensor);
	void (*free)(TPTensor PTensor);
} TResponse, *TPResponse;

typedef struct ANN_CNN_Prediction
{
	// char *lableName;
	uint16_t labelIndex;
	float32_t likeliHood;
} TPrediction, *TPPrediction;

typedef struct ANN_CNN_Volume
{
	uint16_t _w, _h, _depth;
	TPTensor weight;
	TPTensor weight_d;
	void (*init)(struct ANN_CNN_Volume *PVolume, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias);
	void (*setValue)(struct ANN_CNN_Volume *PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
	void (*setGradValue)(struct ANN_CNN_Volume *PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
	void (*addGradValue)(struct ANN_CNN_Volume *PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
	float32_t (*getValue)(struct ANN_CNN_Volume *PVolume, uint16_t X, uint16_t Y, uint16_t Depth);
	float32_t (*getGradValue)(struct ANN_CNN_Volume *PVolume, uint16_t X, uint16_t Y, uint16_t Depth);
	void (*fillZero)(TPTensor PTensor);
	void (*fillGauss)(TPTensor PTensor);
	void (*print)(struct ANN_CNN_Volume *PVolume, uint8_t wg);
	void (*free)(TPTensor PTensor);
} TVolume, *TPVolume;

typedef struct ANN_CNN_Filters
{
	uint16_t _w, _h, _depth;
	uint16_t filterNumber;
	TPVolume *volumes;
	void (*init)(TPVolume PVolume, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias);
	void (*free)(struct ANN_CNN_Filters *PFilters);
} TFilters, *TPFilters;

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
	float32_t l1_decay_mul;
	float32_t l2_decay_mul;
	float32_t drop_prob;
	float32_t bias;
} TLayerOption, *TPLayerOption;

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
} TLayer, *TPLayer;

typedef struct ANN_CNN_InputLayer
{
	TLayer layer;
	void (*init)(struct ANN_CNN_InputLayer *PLayer, TPLayerOption PLayerOption);
	void (*free)(struct ANN_CNN_InputLayer *PLayer);
	void (*forward)(struct ANN_CNN_InputLayer *PLayer, TVolume *Volume);
	void (*backward)(struct ANN_CNN_InputLayer *PLayer);
	float32_t (*computeLoss)(struct ANN_CNN_InputLayer *PLayer, int Y);
	float32_t (*backwardOutput)(struct ANN_CNN_InputLayer *PLayer, TTensor Tensor);
} TInputLayer, *TPInputLayer;

typedef struct ANN_CNN_ConvolutionLayer
{
	TLayer layer;
	TPFilters filters;
	uint16_t stride;
	uint16_t padding;
	float32_t l1_decay_mul;
	float32_t l2_decay_mul;
	float32_t bias;
	TPVolume biases;
	void (*init)(struct ANN_CNN_ConvolutionLayer *PLayer, TPLayerOption PLayerOption);
	void (*free)(struct ANN_CNN_ConvolutionLayer *PLayer);
	void (*forward)(struct ANN_CNN_ConvolutionLayer *PLayer);
	void (*backward)(struct ANN_CNN_ConvolutionLayer *PLayer);
	float32_t (*computeLoss)(struct ANN_CNN_ConvolutionLayer *PLayer, int Y);
	TPResponse *(*getParamsAndGrads)(struct ANN_CNN_ConvolutionLayer *PConvLayer);
	float32_t (*backwardOutput)(struct ANN_CNN_ConvolutionLayer *PLayer, TTensor Tensor);

} TConvLayer, *TPConvLayer;

typedef struct ANN_CNN_PoolLayer
{
	TLayer layer;
	uint16_t stride;
	uint16_t padding;
	// float32_t l1_decay_mul;
	// float32_t l2_decay_mul;
	// float32_t bias;
	TVolume filter;
	TPVolume switchxy;
	// TPVolume switchy;
	void (*init)(struct ANN_CNN_PoolLayer *PLayer, TPLayerOption PLayerOption);
	void (*free)(struct ANN_CNN_PoolLayer *PLayer);
	void (*forward)(struct ANN_CNN_PoolLayer *PLayer);
	void (*backward)(struct ANN_CNN_PoolLayer *PLayer);
	float32_t (*computeLoss)(struct ANN_CNN_PoolLayer *PLayer, int Y);
	TPResponse *(*getParamsAndGrads)(struct ANN_CNN_PoolLayer *PPoolLayer);
	float32_t (*backwardOutput)(struct ANN_CNN_PoolLayer *PLayer, TTensor Tensor);
} TPoolLayer, *TPPoolLayer;

typedef struct ANN_CNN_ReluLayer
{
	TLayer layer;
	// TTensor tensor;
	void (*init)(struct ANN_CNN_ReluLayer *PLayer, TPLayerOption PLayerOption);
	void (*free)(struct ANN_CNN_ReluLayer *PLayer);
	void (*forward)(struct ANN_CNN_ReluLayer *PLayer);
	void (*backward)(struct ANN_CNN_ReluLayer *PLayer);
	float32_t (*computeLoss)(struct ANN_CNN_ReluLayer *PLayer, int Y);
	float32_t (*backwardOutput)(struct ANN_CNN_ReluLayer *PLayer, TTensor Tensor);
	TPResponse *(*getParamsAndGrads)(struct ANN_CNN_ReluLayer *PReluLayer);
} TReluLayer, *TPReluLayer;

typedef struct ANN_CNN_FullyConnLayer
{
	TLayer layer;
	TPFilters filters;
	TPVolume biases;
	float32_t l1_decay_mul;
	float32_t l2_decay_mul;
	float32_t bias;
	void (*init)(struct ANN_CNN_FullyConnLayer *PLayer, TPLayerOption PLayerOption);
	void (*free)(struct ANN_CNN_FullyConnLayer *PLayer);
	void (*forward)(struct ANN_CNN_FullyConnLayer *PLayer);
	void (*backward)(struct ANN_CNN_FullyConnLayer *PLayer);
	float32_t (*computeLoss)(struct ANN_CNN_FullyConnLayer *PLayer, int Y);
	float32_t (*backwardOutput)(struct ANN_CNN_FullyConnLayer *PLayer, TTensor Tensor);
	TPResponse *(*getParamsAndGrads)(struct ANN_CNN_FullyConnLayer *PFullyConnLayer);
} TFullyConnLayer, *TPFullyConnLayer;

typedef struct ANN_CNN_SoftmaxLayer
{
	TLayer layer;
	TPTensor exp;
	uint16_t expected_value;
	void (*init)(struct ANN_CNN_SoftmaxLayer *PLayer, TPLayerOption PLayerOption);
	void (*free)(struct ANN_CNN_SoftmaxLayer *PLayer);
	void (*forward)(struct ANN_CNN_SoftmaxLayer *PLayer);
	void (*backward)(struct ANN_CNN_SoftmaxLayer *PLayer);
	float32_t (*computeLoss)(struct ANN_CNN_SoftmaxLayer *PLayer);
	// float32_t (*backwardOutput)(struct ANN_CNN_SoftmaxLayer *PLayer, TTensor Tensor);
	TPResponse *(*getParamsAndGrads)(struct ANN_CNN_SoftmaxLayer *PSoftmaxLayer);
} TSoftmaxLayer, *TPSoftmaxLayer;

typedef struct ANN_CNN_LearningParameter
{
	TDataSetType data_type;
	uint16_t optimize_method;
	uint16_t batch_size;
	float32_t learning_rate;
	float32_t l1_decay;
	float32_t l2_decay;
	float32_t eps;
	float32_t beta1; // for adam
	float32_t beta2; // for adam
	float32_t momentum;
	float32_t bias;
} TLearningParameter, *TPLearningParameter;

typedef struct ANN_CNN_Learning
{
	bool trainningGoing;
	float32_t cost_loss;
	float32_t sum_cost_loss;
	float32_t l1_decay_loss;
	float32_t l2_decay_loss;
	float32_t trainingAccuracy;
	float32_t testingAccuracy;

	uint32_t labelIndex;
	uint32_t sampleCount;
	uint32_t epochCount;
	uint32_t batchCount;
	uint32_t datasetIndex;
	uint32_t datasetTotal;

	uint16_t responseCount;
	TPResponse *pResponseResults;
	uint16_t predictionCount;
	TPPrediction *pPredictions;
	TPTensor *gsum1;
	TPTensor *gsum2;
	uint16_t gsumCount;
} TLearningResult, *TPLearningResult;

typedef struct ANN_CNN_NeuralNet
{
	TPLayer *layers;
	uint16_t depth;
	uint32_t fwTime;
	uint32_t bwTime;
	uint32_t optimTime;
	TLearningParameter trainningParam;
	TLearningResult trainning;

	void (*init)(struct ANN_CNN_NeuralNet *PNeuralNet, TPLayerOption PLayerOption);
	void (*free)(struct ANN_CNN_NeuralNet *PNeuralNet);
	void (*forward)(struct ANN_CNN_NeuralNet *PNeuralNet, TPVolume PVolume);
	void (*backward)(struct ANN_CNN_NeuralNet *PNeuralNet);
	void (*getParamsAndGrads)(struct ANN_CNN_NeuralNet *PNeuralNet);
	void (*getCostLoss)(struct ANN_CNN_NeuralNet *PNeuralNet);
	void (*getPredictions)(struct ANN_CNN_NeuralNet *PNeuralNet);
	void (*getMaxPrediction)(struct ANN_CNN_NeuralNet *PNeuralNet, TPPrediction PPrediction);
	void (*train)(struct ANN_CNN_NeuralNet *PNeuralNet, TPVolume PVolume);
	void (*printWeights)(struct ANN_CNN_NeuralNet *PNeuralNet, uint16_t LayerIndex, uint8_t InOut);
	// void (*printFilters)(struct ANN_CNN_NeuralNet *PNeuralNet,uint16_t LayerIndex,uint8_t InOut);
	void (*printGradients)(struct ANN_CNN_NeuralNet *PNeuralNet, uint16_t LayerIndex, uint8_t InOut);
	void (*printTrainningInfo)(struct ANN_CNN_NeuralNet *PNeuralNet);
	void (*print)(char *Name, TPTensor PTensor);
	void (*save)(struct ANN_CNN_NeuralNet *PNeuralNet);
	void (*load)(struct ANN_CNN_NeuralNet *PNeuralNet);
	char *(*getName)(TLayerType LayerType);
} TNeuralNet, *TPNeuralNet;

typedef struct ANN_CNN_DataSet_Image
{
	TDataSetType data_type;
	// char *lableName;
	uint16_t labelIndex;
	uint16_t detailIndex;
	TPVolume volume;
} TDSImage, *TPPicture;

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
void FillZero(TPTensor PTensor);
void FillGauss(TPTensor PTensor);
TPFilters MakeFilters(uint16_t W, uint16_t H, uint16_t Depth, uint16_t FilterNumber);
TPVolume MakeVolume(uint16_t W, uint16_t H, uint16_t Depth);
void FreeFilterVolumes(TPFilters PFilters);
void InitVolume(TPVolume PVolume, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias);
void FreeVolume(TPVolume PVolume);
void VolumeSetValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
void VolumeAddValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
float32_t VolumeGetValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth);
void VolumeSetGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
void VolumeAddGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value);
float32_t VolumeGetGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth);
void VolumePrint(TPVolume PVolume, uint8_t wg);

void NeuralNetInit(TPNeuralNet PNeuralNet, TPLayerOption PLayerOption);
void NeuralNetFree(TPNeuralNet PNeuralNet);
void NeuralNetForward(TPNeuralNet PNeuralNet, TPVolume PVolume);
void NeuralNetBackward(TPNeuralNet PNeuralNet);
void NeuralNetGetParamsAndGrads(TPNeuralNet PNeuralNet);
void NeuralNetComputeCostLoss(TPNeuralNet PNeuralNet);
void NeuralNetUpdatePrediction(TPNeuralNet PNeuralNet);
void NeuralNetSave(TPNeuralNet PNeuralNet);
void NeuralNetLoad(TPNeuralNet PNeuralNet);
void NeuralNetGetMaxPrediction(TPNeuralNet PNeuralNet, TPPrediction PPrediction);
void NeuralNetTrain(TPNeuralNet PNeuralNet, TPVolume PVolume);

/// @brief ////////////////////////////////////////////////////////////////////////////////////////////////
/// @return
TPNeuralNet CreateNeuralNetCNN();
void InitNeuralNet_CNN(void);
void InitLeaningParameter(void);
void NeuralNet_Start(void);
void PrintNetInformation(void);

void testDSPFloatProcess(void);
time_t GetTimestamp(void);
#endif /* _INC_ANN_CNN_H_ */
