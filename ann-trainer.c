/*
 * trainer-cnn.c
 *
 *  Created on: 2023��4��22��
 *      Author: lenovo
 */

#ifdef PLATFORM_STM32
#include "usart.h"
#include "octopus.h"
#endif

#include "string.h"
#include "ann-cnn.h"
#include "dataset_cifar.h"

///////////////////////////////////////////////////////////////////

TPNeuralNet PNeuralNetCNN = NULL;
// TLayerOption InputOption = {Layer_Type_Input, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption ConvOption = {Layer_Type_Convolution, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption PoolOption = {Layer_Type_Pool, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption ReluOption = {Layer_Type_ReLu, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption FullyConnOption = {Layer_Type_FullyConnection, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// TLayerOption SoftMaxOption = {Layer_Type_SoftMax, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
TLayerOption LayerOption = {Layer_Type_None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

void NeuralNet_Start()
{
	TPPicture pInputImage = NULL;
	TPrediction prediction;
	// float32_t trainingAccuracy = 0;
	if (PNeuralNetCNN == NULL || PNeuralNetCNN->backward == NULL || PNeuralNetCNN->forward == NULL || PNeuralNetCNN->init == NULL || PNeuralNetCNN->train == NULL || PNeuralNetCNN->depth < 5)
	{
		LOGINFO("Neural Net CNN is not init!!!\n");
		return;
	}
	PNeuralNetCNN->trainning.datasetIndex = 0;
	// PNeuralNetCNN->trainning.trainingAccuracy = 0;
	PNeuralNetCNN->trainning.trainningGoing = true;
	while (PNeuralNetCNN->trainning.trainningGoing)
	{
		pInputImage = (TPPicture)Dataset_GetTrainningPic(PNeuralNetCNN->trainning.datasetIndex, PNeuralNetCNN->trainningParam.data_type);
		if (pInputImage != NULL)
		{
			PNeuralNetCNN->trainning.labelIndex = pInputImage->labelIndex;
			PNeuralNetCNN->trainning.sampleCount++;
			// LOG("\n/////////////////////////////////////////////////////////////////////////////////////////////////////\n");
			// LOG("trainning.sampleCount=%d trainning.imageIndex=%d image_label=%d\n", PNeuralNetCNN->trainning.sampleCount, PNeuralNetCNN->trainning.datasetIndex, PNeuralNetCNN->trainning.labelIndex);

			PNeuralNetCNN->train(PNeuralNetCNN, pInputImage->volume);
			// PNeuralNetCNN->getPredictions(PNeuralNetCNN);
			PNeuralNetCNN->getMaxPrediction(PNeuralNetCNN, &prediction);
			if (prediction.labelIndex == pInputImage->labelIndex)
			{

				PNeuralNetCNN->trainning.trainingAccuracy++;
			}

			// if (PNeuralNetCNN->trainning.batchCount % PNeuralNetCNN->trainningParam.batch_size == 0)
			{
				LOG("\n");
				//LOGINFO("MaxPredictionValue=%d MaxPredictionLikehook=%f ExpectedValue=%d ExpectedProbability=1,Loss=%f",
				//		prediction.labelIndex, prediction.likeliHood, PNeuralNetCNN->trainning.labelIndex, PNeuralNetCNN->trainning.cost_loss);
				PNeuralNetCNN->printTrainningInfo(PNeuralNetCNN);

				//printf("\033[4A");
			}
		}

		if (PNeuralNetCNN->trainning.datasetIndex >= PNeuralNetCNN->trainning.datasetTotal)
		{
			PNeuralNetCNN->trainning.datasetIndex = 0;
			PNeuralNetCNN->trainning.epochCount++;
		}
		else
		{
			PNeuralNetCNN->trainning.datasetIndex++;
		}

		// if (PNeuralNetCNN->trainning.sampleCount >= 100)
		//break;
	}
}

/// @brief ///////////////////////////////////////////////////////
/// @param
void InitNeuralNet_CNN(void)
{
	TPLayer pNetLayer;

	PNeuralNetCNN = CreateNeuralNetCNN();

	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_Input;
	LayerOption.in_w = 32;
	LayerOption.in_h = 32;
	LayerOption.in_depth = 3;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);
	///////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
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
	LayerOption.l1_decay_mul = 1;
	LayerOption.l2_decay_mul = 1;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);

	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);

	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
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
	LayerOption.filter_number = 1;
	LayerOption.stride = 2;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);
	///////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
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
	LayerOption.l1_decay_mul = 1;
	LayerOption.l2_decay_mul = 1;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);

	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);

	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
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
	LayerOption.filter_number = 1;
	LayerOption.stride = 2;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);

	/////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
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
	LayerOption.l1_decay_mul = 1;
	LayerOption.l2_decay_mul = 1;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);

	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_ReLu;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);

	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
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
	LayerOption.filter_number = 1;
	LayerOption.stride = 2;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);
	//////////////////////////////////////////////////////////////////
	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_FullyConnection;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;

	LayerOption.filter_w = 5;
	LayerOption.filter_h = 5;
	LayerOption.filter_depth = LayerOption.in_w * LayerOption.in_h * LayerOption.in_depth;
	LayerOption.filter_number = 10;

	LayerOption.out_depth = LayerOption.filter_number;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;

	LayerOption.bias = 0;
	LayerOption.l1_decay_mul = 0;
	LayerOption.l2_decay_mul = 1;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);

	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	//	pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	memset(&LayerOption, 0, sizeof(TLayerOption));
	LayerOption.LayerType = Layer_Type_SoftMax;
	LayerOption.in_w = pNetLayer->out_w;
	LayerOption.in_h = pNetLayer->out_h;
	LayerOption.in_depth = pNetLayer->out_depth;
	LayerOption.out_depth = 10;
	LayerOption.out_h = 1;
	LayerOption.out_w = 1;
	PNeuralNetCNN->init(PNeuralNetCNN, &LayerOption);
	pNetLayer = PNeuralNetCNN->layers[PNeuralNetCNN->depth - 1];
	// LOG("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d\n", PNeuralNetCNN->depth - 1, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
	// pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
	// LOG("\n////////////////////////////////////////////////////////////////////////////////////\n");
	for (uint16_t out_d = 0; out_d < PNeuralNetCNN->depth; out_d++)
	{
		pNetLayer = PNeuralNetCNN->layers[out_d];
		if (pNetLayer->LayerType == Layer_Type_Convolution)
		{
			LOGINFO("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d fileterNumber=%d size=%dx%dx%d %s", out_d, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
					pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber,
					((TPConvLayer)pNetLayer)->filters->_w, ((TPConvLayer)pNetLayer)->filters->_h, ((TPConvLayer)pNetLayer)->filters->_depth, PNeuralNetCNN->getName(pNetLayer->LayerType));
		}
		else if (pNetLayer->LayerType == Layer_Type_FullyConnection)
		{
			LOGINFO("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d fileterNumber=%d size=%dx%dx%d %s", out_d, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
					pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber,
					((TPFullyConnLayer)pNetLayer)->filters->_w, ((TPFullyConnLayer)pNetLayer)->filters->_h, ((TPFullyConnLayer)pNetLayer)->filters->_depth, PNeuralNetCNN->getName(pNetLayer->LayerType));
		}
		else
		{
			LOGINFO("NeuralNetCNN[%02d,%02d]:in_w=%2d, in_h=%2d, in_depth=%2d, out_w=%2d, out_h=%2d, out_depth=%2d %s", out_d, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
					pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, PNeuralNetCNN->getName(pNetLayer->LayerType));
		}
	}
}

////////////////////////////////////////////////////////////////////
/// @brief /////////////////////////////////////////////////////////
void InitLeaningParameter(void)
{
	if (PNeuralNetCNN == NULL)
	{
		LOG("PNeuralNetCNN is null,please create a neural net cnn first!");
		return;
	}
	PNeuralNetCNN->trainningParam.data_type = Cifar10;
	PNeuralNetCNN->trainningParam.optimize_method = Optm_Adam;
	PNeuralNetCNN->trainningParam.batch_size = 10;
	PNeuralNetCNN->trainningParam.l1_decay = 0;
	PNeuralNetCNN->trainningParam.l2_decay = 0.0001;
	PNeuralNetCNN->trainningParam.beta1 = 0.9;
	PNeuralNetCNN->trainningParam.beta2 = 0.999;
	PNeuralNetCNN->trainningParam.eps = 0.0000001;
	PNeuralNetCNN->trainningParam.learning_rate = 0.005;
	PNeuralNetCNN->trainningParam.momentum = 0.9;
	PNeuralNetCNN->trainningParam.bias = 0.1;
	//
	PNeuralNetCNN->trainning.datasetIndex = 0;
	PNeuralNetCNN->trainning.datasetTotal = 10000;
	PNeuralNetCNN->trainning.sampleCount = 0;
	PNeuralNetCNN->trainning.epochCount = 0;
	PNeuralNetCNN->trainning.batchCount = 0;
	PNeuralNetCNN->trainning.sum_cost_loss = 0;
	PNeuralNetCNN->trainning.l1_decay_loss = 0;
	PNeuralNetCNN->trainning.l2_decay_loss = 0;
	PNeuralNetCNN->trainning.pResponseResults = NULL;
	PNeuralNetCNN->trainning.pPredictions = NULL;
	PNeuralNetCNN->trainning.responseCount = 0;
	PNeuralNetCNN->trainning.predictionCount = 0;
	PNeuralNetCNN->trainning.gsumCount = 0;
	PNeuralNetCNN->trainning.gsum1 = NULL;
	PNeuralNetCNN->trainning.gsum2 = NULL;

	LOG("[LeaningParameters]:data_type:%s optimize_method=%d batch_size=%d l1_decay=%f l2_decay=%f beta1=%f beta2=%f eps=%f learning_rate=%f,momentum=%f bias=%f\n",
		GetDataSetName(PNeuralNetCNN->trainningParam.data_type),
		PNeuralNetCNN->trainningParam.optimize_method,
		PNeuralNetCNN->trainningParam.batch_size,
		PNeuralNetCNN->trainningParam.l1_decay,
		PNeuralNetCNN->trainningParam.l2_decay,
		PNeuralNetCNN->trainningParam.beta1,
		PNeuralNetCNN->trainningParam.beta2,
		PNeuralNetCNN->trainningParam.eps,
		PNeuralNetCNN->trainningParam.learning_rate,
		PNeuralNetCNN->trainningParam.momentum,
		PNeuralNetCNN->trainningParam.bias);
}

void PrintNetInformation(void)
{
	uint16_t inputVolCount = 0;
	uint16_t outputVolCount = 0;
	uint16_t filterCount = 0;
	uint32_t filterLength = 0;
	uint16_t biasCount = 0;
	uint16_t others = 0;
	uint32_t outLength = 0;
	float32_t totalSize = 0;
	if (PNeuralNetCNN == NULL || PNeuralNetCNN->backward == NULL || PNeuralNetCNN->forward == NULL || PNeuralNetCNN->init == NULL || PNeuralNetCNN->train == NULL || PNeuralNetCNN->depth < 5)
	{
		LOGINFO("Neural Net CNN is not init!!!\n");
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
	// totalSize = (filterLength *sizeof(float32_t)) / 1024;
	LOG("[NeuralNetCNNInfor]:in_v_count:%d out_v_count:%d bias_count:%d others:%d filter_count:%d filter_length:%d filter_size:%.2fk out_length:%d out_size:%.2fk total_size:%.2fk",
		inputVolCount, outputVolCount, biasCount, others, filterCount, filterLength, filterLength * sizeof(float32_t) / 1024, outLength, outLength * sizeof(float32_t) / 1024, totalSize);
}
