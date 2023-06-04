/////////////////////////////////////////////////////////////////////////////////////////////
/*
 *  ann-configuragion.c
 *  Home Page :http://www.1234998.top
 *  Created on: June 04, 2023
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

#define NET_CIFAR10_NAME "Cifar10"
#define NET_CIFAR100_NAME "Cifar100"

// 预定义两个网络模型示例分别学习Cifar10和Cifar100
TPNeuralNet PNeuralNetCNN_Cifar10 = NULL;
TPNeuralNet PNeuralNetCNN_Cifar100 = NULL;


// TLayerOption InputOption = {Layer_Type_Input, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption ConvOption = {Layer_Type_Convolution, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption PoolOption = {Layer_Type_Pool, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption ReluOption = {Layer_Type_ReLu, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption FullyConnOption = {Layer_Type_FullyConnection, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption SoftMaxOption = {Layer_Type_SoftMax, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
TLayerOption LayerOption = { Layer_Type_None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };


/////////////////////////////////////////////////////////////////////////////////////////////
/// @brief /////////////////////////////////////////////////////////////////////////////////
/// @param 网络示例1
void NeuralNetInit_Cifar10_11(void)//总共11层网络，含四个权重层，三个卷积一个全连接
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

	LOG("\n");
	NeuralNetInitLeaningParameter(PNeuralNetCNN_Cifar10);
	NeuralNetPrintNetInformation(PNeuralNetCNN_Cifar10);
}

/// @brief ///////////////////////////////////////////////////////
/// @param 网络示例2
void NeuralNetInit_Cifar100(void)//总共11层网络，含四个权重层，三个卷积一个全连接
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

	LOG("\n");
	PNeuralNetCNN_Cifar100->printNetLayersInfor(PNeuralNetCNN_Cifar100);
	LOG("\n");
	NeuralNetInitLeaningParameter(PNeuralNetCNN_Cifar100);
	NeuralNetPrintNetInformation(PNeuralNetCNN_Cifar100);
}
////////////////////////////////////////////////////////////////////////////////////////////
///更深的网络更多的滤波器
///加入随机翻转
TPNeuralNet NeuralNetInit_C_CNN_9(char* NetName)//总共20层，7个卷积层2个全连接层
{
	TPNeuralNet pNeuralNetCNN = NeuralNetCNNCreate(NetName);
	TPLayer pNetLayer;
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Input;
	LayerOption.in_w = 32;
	LayerOption.in_h = 32;
	LayerOption.in_depth = 3;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///1////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///2////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///3////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.stride = 2;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///4////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///5////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///6////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.stride = 2;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///7//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///8//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///9//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///10//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///11//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///12//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///13//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///14//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///15//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///16//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///17//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.stride = 2;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///18//////////////////////////////////////////////////////////////

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_FullyConnection;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_depth = LayerOption.in_w * LayerOption.in_h * LayerOption.in_depth;
	LayerOption.filter_number = 32;
	LayerOption.out_depth = LayerOption.filter_number;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.bias = 0;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///19//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///20//////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_SoftMax;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.out_depth = LayerOption.in_depth * LayerOption.in_w * LayerOption.in_h; // 10;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);


	LOG("\n");
	pNeuralNetCNN->printNetLayersInfor(pNeuralNetCNN);
	LOG("\n");
	NeuralNetInitLeaningParameter(pNeuralNetCNN);
	NeuralNetPrintNetInformation(pNeuralNetCNN);

	return pNeuralNetCNN;
}

//一个类似VGG16的网络结构
TPNeuralNet NeuralNetInit_C_CNN_16(char* NetName)//结构类似VGG16
{
	TPNeuralNet pNeuralNetCNN = NeuralNetCNNCreate(NetName);
	TPLayer pNetLayer;
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Input;
	LayerOption.in_w = 32;
	LayerOption.in_h = 32;
	LayerOption.in_depth = 3;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	///1////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.stride = 2;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///2////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.stride = 2;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///3////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///4////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.stride = 2;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///5////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///6////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Pool;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_w = 2;
	LayerOption.filter_h = 2;
	LayerOption.filter_depth = LayerOption.in_depth;
	LayerOption.stride = 2;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	///7////////////////////////////////////////////////////////////////
	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_FullyConnection;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.filter_depth = LayerOption.in_w * LayerOption.in_h * LayerOption.in_depth;
	LayerOption.filter_number = 32;
	LayerOption.out_depth = LayerOption.filter_number;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.bias = 0;
	LayerOption.l1_decay_rate = 1;
	LayerOption.l2_decay_rate = 1;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);


	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
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
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	pNetLayer = pNeuralNetCNN->layers[pNeuralNetCNN->depth - 1];
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_SoftMax;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	LayerOption.out_depth = LayerOption.in_depth * LayerOption.in_w * LayerOption.in_h; // 10;
	pNeuralNetCNN->init(pNeuralNetCNN, &LayerOption);

	LOG("\n");
	pNeuralNetCNN->printNetLayersInfor(pNeuralNetCNN);
	LOG("\n");
	NeuralNetInitLeaningParameter(pNeuralNetCNN);

	NeuralNetPrintNetInformation(pNeuralNetCNN);

	return pNeuralNetCNN;
}

