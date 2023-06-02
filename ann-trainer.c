/////////////////////////////////////////////////////////////////////////////////////////////
/*
 *  trainer-cnn.c
 *  Home Page :http://www.1234998.top
 *  Created on: May 20, 2023
 *  Author: M
 */
/////////////////////////////////////////////////////////////////////////////////////////////

#ifdef PLATFORM_STM32
#include "usart.h"
#include "octopus.h"
#endif

#include "string.h"
#include "ann-cnn.h"
#include "ann-dataset.h"

/////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetInitLeaningParameter(TPNeuralNet PNeuralNetCNN);
void NeuralNetPrintNetInformation(TPNeuralNet PNeuralNetCNN);
void NeuralNetStartPrediction(TPNeuralNet PNeuralNetCNN);
void PrintTrainningInfor(TPNeuralNet PNeuralNetCNN);

/////////////////////////////////////////////////////////////////////////////////////////////
// 定义三个不同深度的神经网络
#define NET_CIFAR10_NAME "Cifar10"
#define NET_CIFAR100_NAME "Cifar100"
#define NET_CIFAR_16_NAME "Cifar100"

// 预定义三个不同深度的网络模型用来学习Cifar10和Cifar100
TPNeuralNet PNeuralNetCNN_Cifar10 = NULL;
TPNeuralNet PNeuralNetCNN_Cifar100 = NULL;
TPNeuralNet PNeuralNetCNN_16 = NULL;

// TLayerOption InputOption = {Layer_Type_Input, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption ConvOption = {Layer_Type_Convolution, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption PoolOption = {Layer_Type_Pool, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption ReluOption = {Layer_Type_ReLu, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption FullyConnOption = {Layer_Type_FullyConnection, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption SoftMaxOption = {Layer_Type_SoftMax, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
TLayerOption LayerOption = {Layer_Type_None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
/////////////////////////////////////////////////////////////////////////////////////////////
/// @brief ///////////////////////////////////////////////////////
/// @param
void NeuralNetInit_Cifar10_11(void)
{
	TPLayer pNetLayer;
	if (PNeuralNetCNN_Cifar10 != NULL)
		PNeuralNetCNN_Cifar10->free(PNeuralNetCNN_Cifar10);
	PNeuralNetCNN_Cifar10 = NeuralNetCNNCreate(NET_CIFAR10_NAME);

	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Input;
	LayerOption.in_w = 32;
	LayerOption.in_h = 32;
	LayerOption.in_depth = 3;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);
	///////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 5;
	LayerOption.filter_h = 5;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 16;
	LayerOption.stride = 1;
	LayerOption.padding = 0;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;

	LayerOption.stride = 2;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);
	///////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 5;
	LayerOption.filter_h = 5;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 20;
	LayerOption.stride = 1;
	LayerOption.padding = 2;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;

	LayerOption.stride = 2;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);

	/////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 5;
	LayerOption.filter_h = 5;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 20;
	LayerOption.stride = 1;
	LayerOption.padding = 2;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;

	LayerOption.stride = 2;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);
	//////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_FullyConnection;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;

	// LayerOption.filter_w = 1;
	// LayerOption.filter_h = 1;
	LayerOption.filter_depth = LayerOption.in_w * LayerOption.in_h * LayerOption.in_depth;
	LayerOption.filter_number = 10;

	LayerOption.out_depth = LayerOption.filter_number;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;

	LayerOption.bias = 0;
	LayerOption.l1_decay_rate = 0;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_SoftMax;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;

	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.out_depth = LayerOption.in_depth * LayerOption.in_w * LayerOption.in_h; // 10;

	PNeuralNetCNN_Cifar10->init(PNeuralNetCNN_Cifar10, &LayerOption);
	pNetLayer = PNeuralNetCNN_Cifar10->layers[PNeuralNetCNN_Cifar10->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	// pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	// LOG("\n////////////////////////////////////////////////////////////////////////////////////\n");
	PNeuralNetCNN_Cifar10->printNetLayersInfor(PNeuralNetCNN_Cifar10);
}

/// @brief ///////////////////////////////////////////////////////
/// @param
void NeuralNetInit_Cifar100(void)
{
	TPLayer pNetLayer;
	if (PNeuralNetCNN_Cifar100 != NULL)
		PNeuralNetCNN_Cifar100->free(PNeuralNetCNN_Cifar100);
	PNeuralNetCNN_Cifar100 = NeuralNetCNNCreate(NET_CIFAR100_NAME);

	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Input;
	LayerOption.in_w = 32;
	LayerOption.in_h = 32;
	LayerOption.in_depth = 3;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);
	///////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 3;
	LayerOption.filter_h = 3;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 20;
	LayerOption.stride = 1;
	LayerOption.padding = 0;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;

	LayerOption.stride = 2;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);
	///////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 3;
	LayerOption.filter_h = 3;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 20;
	LayerOption.stride = 1;
	LayerOption.padding = 2;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;

	LayerOption.stride = 2;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);

	/////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 3;
	LayerOption.filter_h = 3;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 20;
	LayerOption.stride = 1;
	LayerOption.padding = 2;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;

	LayerOption.stride = 2;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);
	//////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_FullyConnection;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_depth = LayerOption.in_w * LayerOption.in_h * LayerOption.in_depth;
	LayerOption.filter_number = 100;
	LayerOption.out_depth = LayerOption.filter_number;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.bias = 0;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);

	pNetLayer = PNeuralNetCNN_Cifar100->layers[PNeuralNetCNN_Cifar100->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_SoftMax;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.out_depth = LayerOption.in_depth * LayerOption.in_w * LayerOption.in_h; // 10;
	PNeuralNetCNN_Cifar100->init(PNeuralNetCNN_Cifar100, &LayerOption);

	PNeuralNetCNN_Cifar100->printNetLayersInfor(PNeuralNetCNN_Cifar100);
}
////////////////////////////////////////////////////////////////////////////////////////////
///更深的网络更多的滤波器
///加入随机翻转
TPNeuralNet NeuralNetInit_C_CNN_16(char* NetName)
{
	TPNeuralNet PNeuralNetCNN_16 = NeuralNetCNNCreate(NetName);
	TPLayer pNetLayer;
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Input;
	LayerOption.in_w = 32;
	LayerOption.in_h = 32;
	LayerOption.in_depth = 3;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///1////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 3;
	LayerOption.filter_h = 3;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 32;
	LayerOption.stride = 1;
	LayerOption.padding = 0;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///2////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///3////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.stride = 2;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///4////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 3;
	LayerOption.filter_h = 3;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 32;
	LayerOption.stride = 1;
	LayerOption.padding = 2;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///5////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///6////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.stride = 2;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);

	///7//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 3;
	LayerOption.filter_h = 3;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 32;
	LayerOption.stride = 1;
	LayerOption.padding = 2;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///8//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///9//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 3;
	LayerOption.filter_h = 3;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 32;
	LayerOption.stride = 1;
	LayerOption.padding = 2;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///10//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///11//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 3;
	LayerOption.filter_h = 3;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 64;
	LayerOption.stride = 1;
	LayerOption.padding = 2;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///12//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);

	///13//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 3;
	LayerOption.filter_h = 3;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 64;
	LayerOption.stride = 1;
	LayerOption.padding = 2;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///14//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);

	///15//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Convolution;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 3;
	LayerOption.filter_h = 3;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.filter_number = 64;
	LayerOption.stride = 1;
	LayerOption.padding = 2;
	LayerOption.bias = 0.1;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///16//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///17//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.stride = 2;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);

	///18//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_FullyConnection;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_depth = LayerOption.in_w * LayerOption.in_h * LayerOption.in_depth;
	LayerOption.filter_number = 64;
	LayerOption.out_depth = LayerOption.filter_number;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.bias = 0;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///19//////////////////////////////////////////////////////////////
	#if 0
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	#endif
	///20//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_FullyConnection;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_depth = LayerOption.in_w * LayerOption.in_h * LayerOption.in_depth;
	LayerOption.filter_number = 64;
	LayerOption.out_depth = LayerOption.filter_number;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.bias = 0;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);
	///21//////////////////////////////////////////////////////////////

	///22//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_FullyConnection;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_depth = LayerOption.in_w * LayerOption.in_h * LayerOption.in_depth;
	LayerOption.filter_number = 10;
	LayerOption.out_depth = LayerOption.filter_number;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.bias = 0;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);

	///23//////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN_16->layers[PNeuralNetCNN_16->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_SoftMax;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.out_depth = LayerOption.in_depth * LayerOption.in_w * LayerOption.in_h; // 10;
	PNeuralNetCNN_16->init(PNeuralNetCNN_16, &LayerOption);

	LOG("\n");
	PNeuralNetCNN_16->printNetLayersInfor(PNeuralNetCNN_16);
	LOG("\n");
	NeuralNetInitLeaningParameter(PNeuralNetCNN_16);
	return PNeuralNetCNN_16;
}
////////////////////////////////////////////////////////////////////
/// @brief /////////////////////////////////////////////////////////
void NeuralNetInitLeaningParameter(TPNeuralNet PNeuralNetCNN)
{
	if (PNeuralNetCNN == NULL)
	{
		LOG("PNeuralNetCNN is null,please create a neural net cnn first!");
		return;
	}
	PNeuralNetCNN->trainningParam.optimize_method = Optm_Adadelta;
	PNeuralNetCNN->trainningParam.batch_size = 16;
	PNeuralNetCNN->trainningParam.l1_decay_rate = 0;
	PNeuralNetCNN->trainningParam.l2_decay_rate = 0.0001;
	PNeuralNetCNN->trainningParam.beta1 = 0.9;
	PNeuralNetCNN->trainningParam.beta2 = 0.999;
	PNeuralNetCNN->trainningParam.eps = 0.0000001;
	PNeuralNetCNN->trainningParam.learning_rate = 0.005;
	PNeuralNetCNN->trainningParam.momentum = 0.90;
	PNeuralNetCNN->trainningParam.bias = 0.1;

	////////////////////////////////////////////////////
	if (strcmp(PNeuralNetCNN->name, NET_CIFAR10_NAME) == 0)
		PNeuralNetCNN->trainning.data_type = Cifar10;
	else if (strcmp(PNeuralNetCNN->name, NET_CIFAR100_NAME) == 0)
		PNeuralNetCNN->trainning.data_type = Cifar100;
	else
		PNeuralNetCNN->trainning.data_type = Cifar10;

	PNeuralNetCNN->trainning.datasetTotal = CIFAR_TRAINNING_IMAGE_COUNT;
	PNeuralNetCNN->trainning.trinning_dataset_index = 0;
	PNeuralNetCNN->trainning.testing_dataset_index = 0;
	PNeuralNetCNN->trainning.sampleCount = 0;
	PNeuralNetCNN->trainning.epochCount = 0;
	PNeuralNetCNN->trainning.batchCount = 0;
	PNeuralNetCNN->trainning.iterations = 0;
	PNeuralNetCNN->trainning.sum_cost_loss = 0;
	PNeuralNetCNN->trainning.sum_l1_decay_loss = 0;
	PNeuralNetCNN->trainning.sum_l2_decay_loss = 0;
	PNeuralNetCNN->trainning.pResponseResults = NULL;
	PNeuralNetCNN->trainning.responseCount = 0;
	PNeuralNetCNN->trainning.pPredictions = NULL;
	PNeuralNetCNN->trainning.predictionCount = 0;
	PNeuralNetCNN->trainning.grads_sum_count = 0;
	PNeuralNetCNN->trainning.grads_sum1 = NULL;
	PNeuralNetCNN->trainning.grads_sum2 = NULL;
	PNeuralNetCNN->trainning.trainingAccuracy = 0;
	PNeuralNetCNN->trainning.testingAccuracy = 0;
	PNeuralNetCNN->trainning.underflow = false;
	PNeuralNetCNN->trainning.overflow = false;
	PNeuralNetCNN->trainning.batch_by_batch = false;
	PNeuralNetCNN->trainning.one_by_one = false;
	PNeuralNetCNN->trainning.randomFlip = false;
	PNeuralNetCNN->totalTime = 0;
	LOG("\n");
	LOG("[LeaningParameters]:data_type:%s optimize_method:%d batch_size:%d l1_decay_rate:%f l2_decay_rate:%f\n",
		GetDataSetName(PNeuralNetCNN->trainning.data_type),
		PNeuralNetCNN->trainningParam.optimize_method,
		PNeuralNetCNN->trainningParam.batch_size,
		PNeuralNetCNN->trainningParam.l1_decay_rate,
		PNeuralNetCNN->trainningParam.l2_decay_rate);
	LOG("[LeaningParameters]:beta1:%f beta2:%f eps:%f learning_rate:%f,momentum:%f bias:%f\n",
		PNeuralNetCNN->trainningParam.beta1,
		PNeuralNetCNN->trainningParam.beta2,
		PNeuralNetCNN->trainningParam.eps,
		PNeuralNetCNN->trainningParam.learning_rate,
		PNeuralNetCNN->trainningParam.momentum,
		PNeuralNetCNN->trainningParam.bias);
}

void NeuralNetPrintNetInformation(TPNeuralNet PNeuralNetCNN)
{
	uint16_t inputVolCount = 0;
	uint16_t outputVolCount = 0;
	uint16_t filterCount = 0;
	uint32_t filterLength = 0;
	uint16_t biasCount = 0;
	uint16_t others = 0;
	uint32_t outLength = 0;
	float32_t totalSize = 0;
	float32_t out_size = 0;
	float32_t filter_size = 0;

	if (PNeuralNetCNN == NULL || PNeuralNetCNN->backward == NULL || PNeuralNetCNN->forward == NULL || PNeuralNetCNN->init == NULL || PNeuralNetCNN->train == NULL || PNeuralNetCNN->depth < 5)
	{
		LOGINFOR("Neural Net CNN is not init!!!\n");
		return;
	}
	for (uint16_t layerIndex = 0; layerIndex < PNeuralNetCNN->depth; layerIndex++)
	{
		TPLayer pNetLayer = PNeuralNetCNN->layers[layerIndex];
		switch (pNetLayer->LayerType)
		{
		case Layer_Type_Input:
		{
			inputVolCount++;
			outputVolCount++;
			outLength = outLength + pNetLayer->out_w * pNetLayer->out_h * pNetLayer->out_depth * 2;
			break;
		}
		case Layer_Type_Convolution:
		{
			outputVolCount++;
			biasCount++;
			filterCount = filterCount + ((TPConvLayer)pNetLayer)->filters->filterNumber;
			filterLength = filterLength + ((TPConvLayer)pNetLayer)->filters->filterNumber * ((TPConvLayer)pNetLayer)->filters->_w * ((TPConvLayer)pNetLayer)->filters->_h * ((TPConvLayer)pNetLayer)->filters->_depth * 2;
			outLength = outLength + pNetLayer->out_w * pNetLayer->out_h * pNetLayer->out_depth * 4;
			break;
		}

		case Layer_Type_ReLu:
		{
			outputVolCount++;
			outLength = outLength + pNetLayer->out_w * pNetLayer->out_h * pNetLayer->out_depth * 2;
			break;
		}
		case Layer_Type_Pool:
		{
			outputVolCount++;
			others = 2;
			outLength = outLength + pNetLayer->out_w * pNetLayer->out_h * pNetLayer->out_depth * 6;
			break;
		}
		case Layer_Type_FullyConnection:
		{
			outputVolCount++;
			biasCount++;
			filterCount = filterCount + ((TPFullyConnLayer)pNetLayer)->filters->filterNumber;
			filterLength = filterLength + ((TPFullyConnLayer)pNetLayer)->filters->filterNumber * ((TPFullyConnLayer)pNetLayer)->filters->_w * ((TPFullyConnLayer)pNetLayer)->filters->_h * ((TPFullyConnLayer)pNetLayer)->filters->_depth * 2;
			outLength = outLength + pNetLayer->out_w * pNetLayer->out_h * pNetLayer->out_depth * 4; // 4个张量空间
			break;
		}
		case Layer_Type_SoftMax:
		{
			//((TPSoftmaxLayer)
			outputVolCount++;
			others = 1;
			outLength = outLength + pNetLayer->out_w * pNetLayer->out_h * pNetLayer->out_depth * 3; // 三个张量空间
			break;
		}
		default:
			break;
		}
	}

	totalSize = (outLength * sizeof(float32_t) + filterLength * sizeof(float32_t)) / 1024;
	out_size = outLength * sizeof(float32_t) / 1024;
	filter_size = filterLength * sizeof(float32_t) / 1024;

	LOG("[NeuralNetCNNInfor]:in_v_count:%d out_v_count:%d bias_count:%d others:%d filter_count:%d\n",
		inputVolCount, outputVolCount, biasCount, others, filterCount);

	LOG("[NeuralNetCNNInfor]:filter_length:%d out_length:%d filter_size:%.2fk out_size:%.2fk total_size:%.2fk",
		filterLength, outLength, filter_size, out_size, totalSize);
}

void NeuralNetStartTrainning(TPNeuralNet PNeuralNetCNN)
{
	TPPicture pTrainningImage = NULL;
	TPrediction prediction;
	bool_t hide_cursor = false;
	time_t elapsed_time_ms = 0;
	
	if (PNeuralNetCNN == NULL || PNeuralNetCNN->backward == NULL || PNeuralNetCNN->forward == NULL || PNeuralNetCNN->init == NULL || PNeuralNetCNN->train == NULL || PNeuralNetCNN->depth < 5)
	{
		LOGINFOR("Neural Net CNN is not init!!!\n");
		return;
	}
	if (!PNeuralNetCNN->trainning.one_by_one)
	{
		if (!hide_cursor)
		{
			CONSOLE_CURSOR_INFO CursorInfo = {1, 0};
			SetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &CursorInfo);
			hide_cursor = true;
		}
	}
	elapsed_time_ms = GetTimestamp();
	PNeuralNetCNN->trainning.trainningGoing = true;
	srand(time(NULL));
	while (PNeuralNetCNN->trainning.trainningGoing)
	{
		pTrainningImage = (TPPicture)Dataset_GetTrainningPic(PNeuralNetCNN->trainning.trinning_dataset_index, PNeuralNetCNN->trainning.data_type);
		if (GenerateRandomNumber() < 0.5 && PNeuralNetCNN->trainning.randomFlip)
		{
			pTrainningImage->volume->flip(pTrainningImage->volume);
			//LOGINFOR("random flip");
		}
		if (pTrainningImage != NULL)
		{
			PNeuralNetCNN->trainning.labelIndex = pTrainningImage->labelIndex;
			PNeuralNetCNN->trainning.sampleCount++;
			PNeuralNetCNN->train(PNeuralNetCNN, pTrainningImage->volume);
			PNeuralNetCNN->getMaxPrediction(PNeuralNetCNN, &prediction);
			if (prediction.labelIndex == pTrainningImage->labelIndex)
			{
				PNeuralNetCNN->trainning.trainingAccuracy++;
			}
			PNeuralNetCNN->totalTime = GetTimestamp() - elapsed_time_ms;
			NeuralNetStartPrediction(PNeuralNetCNN);
			#ifdef PLATFORM_STM32
			PNeuralNetCNN->printTrainningInfor(PNeuralNetCNN);
			#else PLATFORM_WINDOWS
			PrintTrainningInfor(PNeuralNetCNN);
			#endif
		}
		else
		{
			LOGINFOR("TrainningImage NULL dataset is missing index=%d", PNeuralNetCNN->trainning.trinning_dataset_index);
		}

		PNeuralNetCNN->trainning.trinning_dataset_index++;
		if (PNeuralNetCNN->trainning.trinning_dataset_index >= PNeuralNetCNN->trainning.datasetTotal)
		{
			PNeuralNetCNN->trainning.trinning_dataset_index = 0;
			PNeuralNetCNN->trainning.epochCount++;
		}

		if (PNeuralNetCNN->trainning.sampleCount % CIFAR_TRAINNING_IMAGE_SAVINT_COUNT == 0)
		{
			if (PNeuralNetCNN->trainning.trainingSaving)
				PNeuralNetCNN->saveWeights(PNeuralNetCNN);
		}
		if (PNeuralNetCNN->trainning.one_by_one)
		{
			PNeuralNetCNN->trainning.trainningGoing = false;
		}
		else if (PNeuralNetCNN->trainning.batch_by_batch && (PNeuralNetCNN->trainning.batchCount % PNeuralNetCNN->trainningParam.batch_size == 0))
		{
			PNeuralNetCNN->trainning.trainningGoing = false;
		}
	}
}

void NeuralNetStartPrediction(TPNeuralNet PNeuralNetCNN)
{
	TPPicture pTestImage = NULL;
	TPrediction prediction;
	if (PNeuralNetCNN == NULL || PNeuralNetCNN->backward == NULL || PNeuralNetCNN->forward == NULL || PNeuralNetCNN->init == NULL || PNeuralNetCNN->train == NULL || PNeuralNetCNN->depth < 5)
	{
		LOGINFOR("Neural Net CNN is not init!!!\n");
		return;
	}
	pTestImage = (TPPicture)Dataset_GetTestingPic(PNeuralNetCNN->trainning.testing_dataset_index, PNeuralNetCNN->trainning.data_type);
	if (pTestImage != NULL)
	{
		PNeuralNetCNN->predict(PNeuralNetCNN, pTestImage->volume);
		PNeuralNetCNN->getMaxPrediction(PNeuralNetCNN, &prediction);
		if (prediction.labelIndex == pTestImage->labelIndex)
		{
			PNeuralNetCNN->trainning.testingAccuracy++;
		}
	}
	PNeuralNetCNN->trainning.testing_dataset_index++;
	if (PNeuralNetCNN->trainning.testing_dataset_index >= CIFAR_TESTING_IMAGE_COUNT)
	{
		PNeuralNetCNN->trainning.testing_dataset_index = 0;
	}
}

void PrintTrainningInfor(TPNeuralNet PNeuralNetCNN)
{
	time_t avg_iterations_time = 0;
    #define FORMATD = "%06d\n"
    #define FORMATF = "%9.6f\n"

	#if 1
	if (strcmp(PNeuralNetCNN->name, "Cifar10") == 0 && PNeuralNetCNN->trainning.sampleCount > 0)
	{
		LOGINFO("NeuralNetName   :%s  ", PNeuralNetCNN->name);
		LOGINFO("DatasetTotal    :%09d  ", PNeuralNetCNN->trainning.datasetTotal);
		LOGINFO("DatasetIndex    :%09d  ", PNeuralNetCNN->trainning.trinning_dataset_index);
		LOGINFO("EpochCount      :%09d  ", PNeuralNetCNN->trainning.epochCount);
		LOGINFO("SampleCount     :%09d  ", PNeuralNetCNN->trainning.sampleCount);
		LOGINFO("LabelIndex      :%09d  ", PNeuralNetCNN->trainning.labelIndex);
		LOGINFO("BatchCount      :%09d  ", PNeuralNetCNN->trainning.batchCount);
		LOGINFO("Iterations      :%09d  ", PNeuralNetCNN->trainning.iterations);

		LOGINFO("AverageCostLoss :%9.6f", PNeuralNetCNN->trainning.sum_cost_loss / PNeuralNetCNN->trainning.sampleCount);
		LOGINFO("L1_decay_loss   :%9.6f", PNeuralNetCNN->trainning.sum_l1_decay_loss / PNeuralNetCNN->trainning.sampleCount);
		LOGINFO("L2_decay_loss   :%9.6f", PNeuralNetCNN->trainning.sum_l2_decay_loss / PNeuralNetCNN->trainning.sampleCount);
		LOGINFO("TrainingAccuracy:%9.6f", PNeuralNetCNN->trainning.trainingAccuracy / PNeuralNetCNN->trainning.sampleCount);
		LOGINFO("TestingAccuracy :%9.6f", PNeuralNetCNN->trainning.testingAccuracy / PNeuralNetCNN->trainning.sampleCount);

		if (PNeuralNetCNN->trainning.iterations > 0)
			avg_iterations_time = PNeuralNetCNN->totalTime / PNeuralNetCNN->trainning.iterations;

		LOGINFO("TotalElapsedTime:%09lld", PNeuralNetCNN->totalTime);
		LOGINFO("ForwardTime     :%09lld", PNeuralNetCNN->fwTime);
		LOGINFO("BackwardTime    :%09lld", PNeuralNetCNN->bwTime);
		LOGINFO("OptimTime       :%09lld", PNeuralNetCNN->optimTime);
		LOGINFO("AvgBatchTime    :%09lld", avg_iterations_time);
		LOGINFO("AvgSampleTime   :%09lld", PNeuralNetCNN->totalTime / PNeuralNetCNN->trainning.sampleCount);
		// if (PNeuralNetCNN_Cifar10->trainning.batchCount > 1 || PNeuralNetCNN_Cifar100->trainning.batchCount > 1)
		printf("\033[19A");
	}
	if (strcmp(PNeuralNetCNN->name, "Cifar100") == 0 && PNeuralNetCNN->trainning.sampleCount > 0)
	{
		LOGINFO("NeuralNetName   :\t\t%06s", PNeuralNetCNN->name);
		LOGINFO("DatasetTotal    :\t\t%06d", PNeuralNetCNN->trainning.datasetTotal);
		LOGINFO("DatasetIndex    :\t\t%06d", PNeuralNetCNN->trainning.trinning_dataset_index);
		LOGINFO("EpochCount      :\t\t%06d", PNeuralNetCNN->trainning.epochCount);
		LOGINFO("SampleCount     :\t\t%06d", PNeuralNetCNN->trainning.sampleCount);
		LOGINFO("LabelIndex      :\t\t%06d", PNeuralNetCNN->trainning.labelIndex);
		LOGINFO("BatchCount      :\t\t%06d", PNeuralNetCNN->trainning.batchCount);
		LOGINFO("Iterations      :\t\t%06d", PNeuralNetCNN->trainning.iterations);

		LOGINFO("AverageCostLoss :\t\t%.6f", PNeuralNetCNN->trainning.sum_cost_loss / PNeuralNetCNN->trainning.sampleCount);
		LOGINFO("L1_decay_loss   :\t\t%.6f", PNeuralNetCNN->trainning.sum_l1_decay_loss / PNeuralNetCNN->trainning.sampleCount);
		LOGINFO("L2_decay_loss   :\t\t%.6f", PNeuralNetCNN->trainning.sum_l2_decay_loss / PNeuralNetCNN->trainning.sampleCount);
		LOGINFO("TrainingAccuracy:\t\t%.6f", PNeuralNetCNN->trainning.trainingAccuracy / PNeuralNetCNN->trainning.sampleCount);
		LOGINFO("TestingAccuracy :\t\t%.6f", PNeuralNetCNN->trainning.testingAccuracy / PNeuralNetCNN->trainning.sampleCount);

		if (PNeuralNetCNN->trainning.iterations > 0)
			avg_iterations_time = PNeuralNetCNN->totalTime / PNeuralNetCNN->trainning.iterations;

		LOGINFO("TotalElapsedTime:\t\t%08lld", PNeuralNetCNN->totalTime);
		LOGINFO("ForwardTime     :\t\t%05lld", PNeuralNetCNN->fwTime);
		LOGINFO("BackwardTime    :\t\t%05lld", PNeuralNetCNN->bwTime);
		LOGINFO("OptimTime       :\t\t%05lld", PNeuralNetCNN->optimTime);
		LOGINFO("AvgBatchTime    :\t\t%05lld", avg_iterations_time);
		LOGINFO("AvgSampleTime   :\t\t%05lld", PNeuralNetCNN->totalTime / PNeuralNetCNN->trainning.sampleCount);
		// if (PNeuralNetCNN->trainning.batchCount > 1 || PNeuralNetCNN->trainning.batchCount > 1)
		printf("\033[19A");
	}
	#else
	LOGINFOR("DatasetTotal:%06d DatasetIndex:%06d EpochCount:%06d SampleCount:%06d LabelIndex:%06d BatchCount:%06d Iterations:%06d",
			 PNeuralNet->trainning.datasetTotal,
			 PNeuralNet->trainning.trinning_dataset_index,
			 PNeuralNet->trainning.epochCount,
			 PNeuralNet->trainning.sampleCount,
			 PNeuralNet->trainning.labelIndex,
			 PNeuralNet->trainning.batchCount,
			 PNeuralNet->trainning.iterations);

	LOGINFOR("AvgCostLoss:%.6f L1_decay_loss:%.6f L2_decay_loss:%.6f TrainingAccuracy:%.6f TestingAccuracy:%.6f",
			 PNeuralNet->trainning.cost_loss_sum / PNeuralNet->trainning.sampleCount,
			 PNeuralNet->trainning.l1_decay_loss / PNeuralNet->trainning.sampleCount,
			 PNeuralNet->trainning.l2_decay_loss / PNeuralNet->trainning.sampleCount,
			 PNeuralNet->trainning.trainingAccuracy / PNeuralNet->trainning.sampleCount,
			 PNeuralNet->trainning.testingAccuracy / PNeuralNet->trainning.sampleCount);

	if (PNeuralNet->trainning.iterations > 0)
		avg_iterations_time = PNeuralNet->totalTime / PNeuralNet->trainning.iterations;

	LOGINFOR("TotalTime:%lld ForwardTime:%05lld BackwardTime:%05lld OptimTime:%05lld AvgBatchTime:%05lld AvgSampleTime:%05lld",
			 PNeuralNet->totalTime,
			 PNeuralNet->fwTime,
			 PNeuralNet->bwTime,
			 PNeuralNet->optimTime,
			 avg_iterations_time,
			 PNeuralNet->totalTime / PNeuralNet->trainning.sampleCount);
	#endif
}
