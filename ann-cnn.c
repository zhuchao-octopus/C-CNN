/*****************************************************************************************************
 * ann-cnn.c
 *
 *  Created on: Mar 29, 2023
 *  Author: M
 ******************************************************************************************************
 ******************************************************************************************************/

#include "ann-cnn.h"

#ifdef PLATFORM_STM32
#include "usart.h"
#include "octopus.h"
#endif

char *CNNTypeName[] = {"Input", "Convolution", "ReLu", "Pool", "FullyConnection", "SoftMax", "None"};

void TestDSPFloatProcess(float32_t f1, float32_t f2)
{
	// float32_t fv;
	// fv = arm_sin_f32(3.1415926 / 7);
	printf("hello got value %f %f\r\n", f1, f2);
}

time_t GetTimestamp(void)
{
	time_t t = clock();
	time_t tick = t * 1000 / CLOCKS_PER_SEC;
	return tick;
	// return time(NULL);
}
double GenerateRandomNumber()
{
	// 设置随机数种子
	//srand(time(NULL));
	// 生成0到RAND_MAX之间的随机整数
	int randInt = rand();
	// 将随机整数映射到-1到1之间的浮点数范围
	double randFloat = (double)randInt / RAND_MAX; // 将整数归一化到0到1之间
	// double randomNum = (randFloat * 2.0) - 1.0;     // 将范围映射到-1到1之间
	return randFloat;
}
float32_t GenerateGaussRandom(void)
{
	float32_t c, u, v, r;
	static bool return_v = false;
	static float32_t v_val;
	c = 0;
	if (return_v)
	{
		return_v = false;
		return v_val;
	}
#ifdef PLATFORM_STM32
	u = 2 * random() - 1;
	v = 2 * random() - 1;
#else
	u = 2 * rand() - 1;
	v = 2 * rand() - 1;
#endif
	r = u * u + v * v;
	if ((r == 0) || (r > 1))
	{
		return GenerateGaussRandom();
	}

#ifdef PLATFORM_STM32
	arm_sqrt_f32(-2 * log10(r) / r, &c);
#else
	c = sqrt(-2 * log10(r) / r);
#endif
	v_val = v * c;
	return_v = true;
	return u * c;
}

float32_t GenerateGaussRandom1(void)
{
	static float32_t v1, v2, s;
	static int start = 0;
	float32_t x;
	if (start == 0)
	{
		do
		{
			float32_t u1 = (float32_t)rand() / RAND_MAX;
			float32_t u2 = (float32_t)rand() / RAND_MAX;
			v1 = 2 * u1 - 1;
			v2 = 2 * u2 - 1;
			s = v1 * v1 + v2 * v2;
		} while (s >= 1 || s == 0);

		x = v1 * sqrt(-2 * log(s) / s);
	}
	else
	{
		x = v2 * sqrt(-2 * log(s) / s);
	}
	start = 1 - start;
	return x;
}

double GenerateGaussRandom2(void)
{
	static double n2 = 0.0;
	static int n2_cached = 0;
	double d;
	if (!n2_cached)
	{
		double x, y, r;
		do
		{
			x = 2.0 * rand() / RAND_MAX - 1;
			y = 2.0 * rand() / RAND_MAX - 1;
			r = x * x + y * y;
		} while (r >= 1.0 || r == 0.0);
#ifdef PLATFORM_STM32
		arm_sqrt_f32(-2 * log10(r) / r, &d);
#else
		d = sqrt(-2.0 * log(r) / r);
#endif
		double n1 = x * d;
		n2 = y * d;
		n2_cached = 1;
		return n1;
	}
	else
	{
		n2_cached = 0;
		return n2;
	}
}

// 生成高斯分布随机数序列期望为μ、方差为σ2=Variance
// 若随机变量X服从一个数学期望为μ、方差为σ2的正态分布记为N(μ，σ2)
// 其概率密度函数为正态分布的期望值μ决定了其位置，其标准差σ决定了分布的幅度
// 当μ = 0,σ = 1时的正态分布是标准正态分布。
float32_t NeuralNet_GetGaussRandom(double mul, float32_t Variance)
{
	return mul + GenerateGaussRandom2() * Variance;
}
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
TPTensor MakeTensor(uint32_t Length)
{
	TPTensor tPTensor = malloc(sizeof(TTensor));
	if (tPTensor != NULL)
	{
		tPTensor->length = Length;
		tPTensor->buffer = malloc(tPTensor->length * sizeof(float32_t));
		if (tPTensor->buffer != NULL)
			memset(tPTensor->buffer, 0, tPTensor->length * sizeof(float32_t));
	}
	return tPTensor;
}

void TensorInit(TPTensor PTensor, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias)
{
	uint32_t n = W * H * Depth;
	PTensor = MakeTensor(n);
	TensorFillZero(PTensor);
}

void TensorFillZero(TPTensor PTensor)
{
	if (PTensor->length > 0)
		memset(PTensor->buffer, 0, PTensor->length * sizeof(float32_t));
}

void TensorFillWith(TPTensor PTensor, float32_t Bias)
{
	for (int i = 0; i < PTensor->length; i++)
	{
		PTensor->buffer[i] = Bias;
	}
}

void TensorFillGauss(TPTensor PTensor)
{
	float32_t scale = 0;
#ifdef PLATFORM_STM32
	arm_sqrt_f32(1.0 / PTensor->length, &scale);
#else
	scale = sqrt(1.0 / PTensor->length);
#endif

	for (int i = 0; i < PTensor->length; i++)
	{
		float32_t v = NeuralNet_GetGaussRandom(0.00, scale);
		PTensor->buffer[i] = v;
	}
}

TPMaxMin TensorMaxMin(TPTensor PTensor)
{
	TPMaxMin pMaxMin = malloc(sizeof(TMaxMin));
	pMaxMin->max = MINFLOAT_NEGATIVE_NUMBER;
	pMaxMin->min = MAXFLOAT_POSITIVE_NUMBER;
	for (int i = 0; i < PTensor->length; i++)
	{
		if (PTensor->buffer[i] > pMaxMin->max)
			pMaxMin->max = PTensor->buffer[i];
		if (PTensor->buffer[i] < pMaxMin->min)
			pMaxMin->min = PTensor->buffer[i];
	}
	return pMaxMin;
}
void TensorFree(TPTensor PTensor)
{
	if (PTensor != NULL)
	{
		free(PTensor->buffer);
		PTensor->length = 0;
		free(PTensor);
		PTensor = NULL;
	}
}

void TensorSave(FILE *pFile, TPTensor PTensor)
{
	if (pFile == NULL)
	{
		LOG("Error opening file NULL");
		return;
	}
	if (PTensor->buffer != NULL && PTensor->length > 0)
		fwrite(PTensor->buffer, 1, sizeof(float32_t) * PTensor->length, pFile);
}

void TensorLoad(FILE *pFile, TPTensor PTensor)
{
	if (pFile == NULL)
	{
		LOG("Error opening file NULL");
		return;
	}
	if (PTensor->buffer != NULL && PTensor->length > 0)
		fread(PTensor->buffer, 1, sizeof(float32_t) * PTensor->length, pFile);
}
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
TPVolume MakeVolume(uint16_t W, uint16_t H, uint16_t Depth)
{
	TPVolume tPVolume = malloc(sizeof(TVolume));
	if (tPVolume != NULL)
	{
		tPVolume->_w = W;
		tPVolume->_h = H;
		tPVolume->_depth = Depth;
		tPVolume->init = VolumeInit;
		tPVolume->free = TensorFree;
		tPVolume->fillZero = TensorFillZero;
		tPVolume->fillGauss = TensorFillGauss;
		tPVolume->setValue = VolumeSetValue;
		tPVolume->getValue = VolumeGetValue;
		tPVolume->addValue = VolumeAddValue;
		tPVolume->setGradValue = VolumeSetGradValue;
		tPVolume->getGradValue = VolumeGetGradValue;
		tPVolume->addGradValue = VolumeAddGradValue;
		tPVolume->flip = VolumeFlip;
		tPVolume->print = VolumePrint;
	}
	return tPVolume;
}

void VolumeInit(TPVolume PVolume, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias)
{
	PVolume->_w = W;
	PVolume->_h = H;
	PVolume->_depth = Depth;
	uint32_t n = PVolume->_w * PVolume->_h * PVolume->_depth;
	PVolume->weight = MakeTensor(n);
	PVolume->weight_d = MakeTensor(n);
	// TensorFillZero(PVolume->weight);
	// TensorFillZero(PVolume->weight_grad);
	TensorFillWith(PVolume->weight, Bias);
	TensorFillWith(PVolume->weight_d, Bias);
}

void VolumeFree(TPVolume PVolume)
{
	if (PVolume != NULL)
	{
		TensorFree(PVolume->weight);
		TensorFree(PVolume->weight_d);
		free(PVolume);
	}
	PVolume = NULL;
}

void VolumeSetValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value)
{
	uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
	PVolume->weight->buffer[index] = Value;
}

void VolumeAddValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value)
{
	uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
	PVolume->weight->buffer[index] = PVolume->weight->buffer[index] + Value;
}

float32_t VolumeGetValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth)
{
	uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
	return PVolume->weight->buffer[index];
}

void VolumeSetGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value)
{
	uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
	PVolume->weight_d->buffer[index] = Value;
}

void VolumeAddGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value)
{
	uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
	PVolume->weight_d->buffer[index] = PVolume->weight_d->buffer[index] + Value;
}

float32_t VolumeGetGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth)
{
	uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
	return PVolume->weight_d->buffer[index];
}
void VolumeFlip(TPVolume PVolume)
{
	uint16_t width = PVolume->_w;
	uint16_t height = PVolume->_h;
	uint16_t depth = PVolume->_depth;

	// 逐通道翻转
	for (uint16_t d = 0; d < depth; d++)
	{
		for (uint16_t y = 0; y < height; y++)
		{
			for (uint16_t x = 0; x < width / 2; x++)
			{
				// 交换像素位置
				float32_t temp = PVolume->weight->buffer[y * width * depth + x * depth + d];
				PVolume->weight->buffer[y * width * depth + x * depth + d] = PVolume->weight->buffer[y * width * depth + (width - x - 1) * depth + d];
				PVolume->weight->buffer[y * width * depth + (width - x - 1) * depth + d] = temp;
			}
		}
	}
}

TPFilters MakeFilters(uint16_t W, uint16_t H, uint16_t Depth, uint16_t FilterNumber)
{
	TPFilters tPFilters = malloc(sizeof(TFilters));
	if (tPFilters != NULL)
	{
		tPFilters->_w = W;
		tPFilters->_h = H;
		tPFilters->_depth = Depth;
		tPFilters->filterNumber = FilterNumber;
		tPFilters->volumes = malloc(sizeof(TPVolume) * tPFilters->filterNumber);
		if (tPFilters->volumes == NULL)
		{
			LOGERROR("tPFilters->volumes==NULL! W=%d H=%d Depth=%d FilterNumber=%d", W, H, Depth, FilterNumber);
			return NULL;
		}
		for (uint16_t i = 0; i < tPFilters->filterNumber; i++)
		{
			tPFilters->volumes[i] = MakeVolume(tPFilters->_w, tPFilters->_w, tPFilters->_depth);
		}
		tPFilters->init = VolumeInit;
		tPFilters->free = FilterVolumesFree;
	}
	return tPFilters;
}

bool FiltersResize(TPFilters PFilters, uint16_t W, uint16_t H, uint16_t Depth, uint16_t FilterNumber)
{
	if (W <= 0 || H <= 0 || Depth <= 0 || FilterNumber <= 0)
	{
		LOGERROR("Resize Filters failed! W=%d H=%d Depth=%d FilterNumber=%d", W, H, Depth, FilterNumber);
		return false;
	}
	if (PFilters != NULL)
	{
		PFilters->_w = W;
		PFilters->_h = H;
		PFilters->_depth = Depth;
		PFilters->filterNumber = FilterNumber;
		PFilters->free(PFilters);
		PFilters->volumes = malloc(sizeof(TPVolume) * PFilters->filterNumber);
		for (uint16_t i = 0; i < PFilters->filterNumber; i++)
		{
			PFilters->volumes[i] = MakeVolume(PFilters->_w, PFilters->_w, PFilters->_depth);
		}
		PFilters->init = VolumeInit;
		PFilters->free = FilterVolumesFree;
	}
	return true;
}

void FiltersFree(TPFilters PFilters)
{
	if (PFilters != NULL)
	{
		for (uint16_t d = 0; d < PFilters->filterNumber; d++)
		{
			VolumeFree(PFilters->volumes[d]);
		}
		free(PFilters);
		PFilters = NULL;
	}
}

void FilterVolumesFree(TPFilters PFilters)
{
	if (PFilters != NULL)
	{
		for (uint16_t d = 0; d < PFilters->filterNumber; d++)
		{
			VolumeFree(PFilters->volumes[d]);
		}
		PFilters->volumes = NULL;
	}
}
/// @brief ////////////////////////////////////////////////////////////
/// @param PVolume
/// @param wg 0:weight,1:weight_grad
void VolumePrint(TPVolume PVolume, uint8_t wg)
{
	if (PVolume->_h == 1 && PVolume->_w == 1)
	{
		if (wg == PRINTFLAG_WEIGHT)
			LOGINFOR("weight:PVolume->_depth=%d/%d", PVolume->_depth, PVolume->_depth);
		else
			LOGINFOR("grads:PVolume->_depth=%d/%d", PVolume->_depth, PVolume->_depth);
	}
	float32_t f32 = 0.00;
	for (uint16_t d = 0; d < PVolume->_depth; d++)
	{
		if (PVolume->_h == 1 && PVolume->_w == 1)
		{
			if (wg == PRINTFLAG_WEIGHT)
			{
				f32 = PVolume->getValue(PVolume, 0, 0, d);
				(d == 0) ? LOG(PRINTFLAG_FORMAT, f32) : LOG("," PRINTFLAG_FORMAT, f32);
			}
			else
			{
				f32 = PVolume->getGradValue(PVolume, 0, 0, d);
				(d == 0) ? LOG(PRINTFLAG_FORMAT, f32) : LOG("," PRINTFLAG_FORMAT, f32);
			}
		}
		else
		{
			if (wg == PRINTFLAG_WEIGHT)
				LOGINFOR("weight:PVolume->_depth=%d/%d %dx%d", d, PVolume->_depth, PVolume->_w, PVolume->_h);
			else
				LOGINFOR("grads:PVolume->_depth=%d/%d %dx%d", d, PVolume->_depth, PVolume->_w, PVolume->_h);
			for (uint16_t y = 0; y < PVolume->_h; y++)
			{
				for (uint16_t x = 0; x < PVolume->_w; x++)
				{
					if (wg == PRINTFLAG_WEIGHT)
					{
						f32 = PVolume->getValue(PVolume, x, y, d);
						(x == 0) ? LOG(PRINTFLAG_FORMAT, f32) : LOG("," PRINTFLAG_FORMAT, f32);
					}
					else
					{
						f32 = PVolume->getGradValue(PVolume, x, y, d);
						(x == 0) ? LOG(PRINTFLAG_FORMAT, f32) : LOG("," PRINTFLAG_FORMAT, f32);
					}
				}
				LOG("\n");
			}
		}
	}
	if (PVolume->_h == 1 && PVolume->_w == 1)
		LOG("\n");
}
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
/// @brief /////////////////////////////////////////////////////////////////////////////
/// @param PInputLayer
/// @param PLayerOption
void InputLayerInit(TPInputLayer PInputLayer, TPLayerOption PLayerOption)
{
	PInputLayer->layer.LayerType = PLayerOption->LayerType;
	PInputLayer->layer.in_v = NULL;
	PInputLayer->layer.out_v = NULL;

	PInputLayer->layer.in_w = PLayerOption->in_w;
	PInputLayer->layer.in_h = PLayerOption->in_h;
	PInputLayer->layer.in_depth = PLayerOption->in_depth;

	PInputLayer->layer.out_w = PLayerOption->out_w = PInputLayer->layer.in_w;
	PInputLayer->layer.out_h = PLayerOption->out_h = PInputLayer->layer.in_h;
	PInputLayer->layer.out_depth = PLayerOption->out_depth = PInputLayer->layer.in_depth;
}

void InputLayerForward(TPInputLayer PInputLayer, TPVolume PVolume)
{
	if (PVolume == NULL)
	{
		LOGERROR("%s is null", CNNTypeName[PInputLayer->layer.LayerType]);
		return;
	}
	if (PInputLayer->layer.in_v != NULL)
	{
		VolumeFree(PInputLayer->layer.in_v);
	}
	PInputLayer->layer.in_w = PVolume->_w;
	PInputLayer->layer.in_h = PVolume->_h;
	PInputLayer->layer.in_depth = PVolume->_depth;
	PInputLayer->layer.in_v = PVolume;

	PInputLayer->layer.out_w = PInputLayer->layer.in_w;
	PInputLayer->layer.out_h = PInputLayer->layer.in_h;
	PInputLayer->layer.out_depth = PInputLayer->layer.in_depth;
	PInputLayer->layer.out_v = PInputLayer->layer.in_v;
}

void InputLayerBackward(TPInputLayer PInputLayer)
{
}
void InputLayerFree(TPInputLayer PInputLayer)
{
	VolumeFree(PInputLayer->layer.in_v);
	VolumeFree(PInputLayer->layer.out_v);
}
void MatrixMultip(TPTensor PInTenser, TPTensor PFilter, TPTensor out)
{
}
/////////////////////////////////////////////////////////////////////////////
/// @brief //////////////////////////////////////////////////////////////////
/// @param PConvLayer
/// @param PLayerOption
void ConvolutionLayerInit(TPConvLayer PConvLayer, TPLayerOption PLayerOption)
{
	PConvLayer->layer.LayerType = PLayerOption->LayerType;
	PConvLayer->l1_decay_rate = PLayerOption->l1_decay_rate;
	PConvLayer->l2_decay_rate = PLayerOption->l2_decay_rate;
	PConvLayer->stride = PLayerOption->stride;
	PConvLayer->padding = PLayerOption->padding;
	PConvLayer->bias = PLayerOption->bias;

	// PConvLayer->filters->_w = PLayerOption->filter_w;
	// PConvLayer->filters->_h = PLayerOption->filter_h;
	// PConvLayer->filters->filterNumber = PLayerOption->filter_number;
	// PConvLayer->filters->_depth = PLayerOption->filter_depth;

	PConvLayer->layer.in_w = PLayerOption->in_w;
	PConvLayer->layer.in_h = PLayerOption->in_h;
	PConvLayer->layer.in_depth = PLayerOption->in_depth; // PLayerOption->filter_depth
	PConvLayer->layer.in_v = NULL;

	if (PLayerOption->filter_depth != PConvLayer->layer.in_depth)
		PLayerOption->filter_depth = PConvLayer->layer.in_depth;
	if (PLayerOption->filter_depth <= 0)
		PLayerOption->filter_depth = 3;
	PConvLayer->filters = MakeFilters(PLayerOption->filter_w, PLayerOption->filter_h, PLayerOption->filter_depth, PLayerOption->filter_number);

	PConvLayer->layer.out_w = floor((PConvLayer->layer.in_w + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
	PConvLayer->layer.out_h = floor((PConvLayer->layer.in_h + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
	PConvLayer->layer.out_depth = PConvLayer->filters->filterNumber;

	PConvLayer->layer.out_v = MakeVolume(PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth);
	PConvLayer->layer.out_v->init(PConvLayer->layer.out_v, PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth, 0);

	PConvLayer->biases = MakeVolume(1, 1, PConvLayer->layer.out_depth);
	PConvLayer->biases->init(PConvLayer->biases, 1, 1, PConvLayer->layer.out_depth, PConvLayer->bias);

	for (uint16_t i = 0; i < PConvLayer->layer.out_depth; i++)
	{
		PConvLayer->filters->init(PConvLayer->filters->volumes[i], PConvLayer->filters->_w, PConvLayer->filters->_h, PConvLayer->filters->_depth, 0);
		PConvLayer->filters->volumes[i]->fillGauss(PConvLayer->filters->volumes[i]->weight);
		// ConvLayer.filters->volumes[i]->fillGauss(ConvLayer.filters->volumes[i]->weight_grad);
	}

	PLayerOption->out_w = PConvLayer->layer.out_w;
	PLayerOption->out_h = PConvLayer->layer.out_h;
	PLayerOption->out_depth = PConvLayer->layer.out_depth;
}
/// @brief //////////////////////////////////////////////////////////////////////////
/// @brief //////////////////////////////////////////////////////////////////////////
/// @param PConvLayer
void convolutionLayerOutResize(TPConvLayer PConvLayer)
{
	uint16_t out_w = floor((PConvLayer->layer.in_w + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
	uint16_t out_h = floor((PConvLayer->layer.in_h + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
	uint16_t filter_depth = PConvLayer->layer.in_depth;

	if (PConvLayer->filters->_depth != filter_depth)
	{
		LOGINFOR("ConvLayer resize filters from %d x %d x%d to %d x %d x %d", PConvLayer->filters->_w, PConvLayer->filters->_h, PConvLayer->filters->_depth, out_w, out_h, filter_depth);
		FiltersFree(PConvLayer->filters);
		bool ret = FiltersResize(PConvLayer->filters, PConvLayer->filters->_w, PConvLayer->filters->_h, filter_depth, PConvLayer->filters->filterNumber);
		if (!ret)
			LOGERROR("Resize Filters failed! W=%d H=%d Depth=%d FilterNumber=%d", PConvLayer->filters->_w, PConvLayer->filters->_h, filter_depth, PConvLayer->filters->filterNumber);
	}
	if (PConvLayer->layer.out_w != out_w || PConvLayer->layer.out_h != out_h)
	{
		LOGINFOR("ConvLayer resize out_v from %d x %d to %d x %d", PConvLayer->layer.out_w, PConvLayer->layer.out_h, out_w, out_h);
		PConvLayer->layer.out_w = out_w;
		PConvLayer->layer.out_h = out_h;

		if (PConvLayer->layer.out_v != NULL)
		{
			VolumeFree(PConvLayer->layer.out_v);
		}
		PConvLayer->layer.out_v = MakeVolume(PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth);
		PConvLayer->layer.out_v->init(PConvLayer->layer.out_v, PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth, 0);
	}
}
/// @brief //////////////////////////////////////////////////////////////////////////
/// @param PConvLayer
void ConvolutionLayerForward(TPConvLayer PConvLayer)
{	
	uint16_t padding_x = -PConvLayer->padding;
	uint16_t padding_y = -PConvLayer->padding;
	TPVolume inVolume = PConvLayer->layer.in_v;
	TPVolume outVolume = PConvLayer->layer.out_v;
	uint16_t in_x, in_y;
	float32_t sum = 0.00;
	convolutionLayerOutResize(PConvLayer);
	// for (uint16_t out_d = 0; out_d < PConvLayer->filters->_depth; out_d++)
	for (uint32_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
	{
		TPVolume filter = PConvLayer->filters->volumes[out_d];
		padding_x = -PConvLayer->padding;
		padding_y = -PConvLayer->padding;
		for (uint32_t out_y = 0; out_y < PConvLayer->layer.out_h; out_y++)
		{
			padding_x = -PConvLayer->padding;
			for (uint32_t out_x = 0; out_x < PConvLayer->layer.out_w; out_x++)
			{
				sum = 0.00;
				for (uint32_t filter_y = 0; filter_y < PConvLayer->filters->_h; filter_y++)
				{
					in_y = filter_y + padding_y;
					if (in_y < 0 || in_y >= inVolume->_h)
						continue;

					for (uint32_t filter_x = 0; filter_x < PConvLayer->filters->_w; filter_x++)
					{
						in_x = filter_x + padding_x;
						if (in_x < 0 || in_x >= inVolume->_w)
							continue;

						uint32_t fx = (filter->_w * filter_y + filter_x) * filter->_depth;
						uint32_t ix = (inVolume->_w * in_y + in_x) * inVolume->_depth;
						for (uint32_t filter_d = 0; filter_d < PConvLayer->filters->_depth; filter_d++)
						{
							// sum = sum + filter->getValue(filter, filter_x, filter_y, filter_d) * inVolume->getValue(inVolume, in_x, in_y, filter_d);
							sum = sum + filter->weight->buffer[fx + filter_d] * inVolume->weight->buffer[ix + filter_d];
						}
					}
				}
				sum = sum + PConvLayer->biases->weight->buffer[out_d];
				uint32_t out_idx = (outVolume->_w * out_y + out_x) * outVolume->_depth + out_d;
				outVolume->weight->buffer[out_idx] = sum;
				padding_x = padding_x + PConvLayer->stride;
			}
			padding_y = padding_y + PConvLayer->stride;
		}
	}
}


void DepthwisePointwiseConvolutionInit(TPConvLayer PConvLayer, TPLayerOption PLayerOption)
{
	PConvLayer->layer.LayerType = PLayerOption->LayerType;
	PConvLayer->l1_decay_rate = PLayerOption->l1_decay_rate;
	PConvLayer->l2_decay_rate = PLayerOption->l2_decay_rate;
	PConvLayer->stride = PLayerOption->stride;
	PConvLayer->padding = PLayerOption->padding;
	PConvLayer->bias = PLayerOption->bias;

	PConvLayer->layer.in_w = PLayerOption->in_w;
	PConvLayer->layer.in_h = PLayerOption->in_h;
	PConvLayer->layer.in_depth = PLayerOption->in_depth; // PLayerOption->filter_depth
	PConvLayer->layer.in_v = NULL;

	if (PLayerOption->filter_depth != PConvLayer->layer.in_depth)
		PLayerOption->filter_depth = PConvLayer->layer.in_depth;
	if (PLayerOption->filter_depth <= 0)
		PLayerOption->filter_depth = 3;

	PLayerOption->filter_number = 2;
	PConvLayer->filters = MakeFilters(PLayerOption->filter_w, PLayerOption->filter_h, PLayerOption->filter_depth, PLayerOption->filter_number);

	PConvLayer->layer.out_w = floor((PConvLayer->layer.in_w + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
	PConvLayer->layer.out_h = floor((PConvLayer->layer.in_h + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
	PConvLayer->layer.out_depth = PConvLayer->layer.in_depth;// PConvLayer->filters->filterNumber;

	PConvLayer->layer.out_v = MakeVolume(PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth);
	PConvLayer->layer.out_v->init(PConvLayer->layer.out_v, PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth, 0);

	PConvLayer->biases = MakeVolume(1, 1, PConvLayer->layer.out_depth);
	PConvLayer->biases->init(PConvLayer->biases, 1, 1, PConvLayer->layer.out_depth, PConvLayer->bias);

	for (uint32_t i = 0; i < PConvLayer->layer.out_depth; i++)
	{
		PConvLayer->filters->init(PConvLayer->filters->volumes[i], PConvLayer->filters->_w, PConvLayer->filters->_h, PConvLayer->filters->_depth, 0);
		PConvLayer->filters->volumes[i]->fillGauss(PConvLayer->filters->volumes[i]->weight);
		// ConvLayer.filters->volumes[i]->fillGauss(ConvLayer.filters->volumes[i]->weight_grad);
	}

	PLayerOption->out_w = PConvLayer->layer.out_w;
	PLayerOption->out_h = PConvLayer->layer.out_h;
	PLayerOption->out_depth = PConvLayer->layer.out_depth;
}
void DepthwisePointwiseConvolution(TPConvLayer PConvLayer)
{
	uint16_t padding_x = -PConvLayer->padding;
	uint16_t padding_y = -PConvLayer->padding;
	TPVolume inVolume = PConvLayer->layer.in_v;
	TPVolume outVolume = PConvLayer->layer.out_v;
	// Depthwise convolution
	TPVolume depthwiseFilter = PConvLayer->filters->volumes[0];
	// Pointwise convolution
	TPVolume pointwiseFilter = PConvLayer->filters->volumes[1];

	for (uint32_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
	{	
		for (uint32_t out_y = 0; out_y < PConvLayer->layer.out_h; out_y++)
		{
			padding_x = -PConvLayer->padding;

			for (uint32_t out_x = 0; out_x < PConvLayer->layer.out_w; out_x++)
			{
				float32_t sum = 0.0;

				for (uint32_t filter_y = 0; filter_y < PConvLayer->filters->_h; filter_y++)
				{
					uint32_t in_y = filter_y + padding_y;

					if (in_y < 0 || in_y >= inVolume->_h)
						continue;

					for (uint16_t filter_x = 0; filter_x < PConvLayer->filters->_w; filter_x++)
					{
						uint32_t in_x = filter_x + padding_x;

						if (in_x < 0 || in_x >= inVolume->_w)
							continue;

						uint32_t fx = (depthwiseFilter->_w * filter_y + filter_x) * depthwiseFilter->_depth;
						uint32_t ix = (inVolume->_w * in_y + in_x) * inVolume->_depth;

						for (uint32_t filter_d = 0; filter_d < depthwiseFilter->_depth; filter_d++)
						{
							sum += depthwiseFilter->weight->buffer[fx + filter_d] * inVolume->weight->buffer[ix + filter_d];
						}
					}
				}

				uint32_t out_idx = (outVolume->_w * out_y + out_x) * outVolume->_depth + out_d;
				outVolume->weight->buffer[out_idx] = sum;

				padding_x += PConvLayer->stride;
			}

			padding_y += PConvLayer->stride;
		}
	}

	

	for (uint32_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
	{
		for (uint32_t out_y = 0; out_y < PConvLayer->layer.out_h; out_y++)
		{
			for (uint32_t out_x = 0; out_x < PConvLayer->layer.out_w; out_x++)
			{
				uint32_t out_idx = (outVolume->_w * out_y + out_x) * outVolume->_depth + out_d;
				float32_t depthwise_out = outVolume->weight->buffer[out_idx];
				float32_t pointwise_out = depthwise_out * pointwiseFilter->weight->buffer[out_d];
				outVolume->weight->buffer[out_idx] = pointwise_out;
			}
		}
	}
}

/// @brief ////////////////////////////////////////////////////////////////////////////////
/// @param PConvLayer
/// y = wx+b 求关于in w b的偏导数
void ConvolutionLayerBackward(TPConvLayer PConvLayer)
{
	float32_t out_grad = 0.00;
	uint16_t padding_x = -PConvLayer->padding;
	uint16_t padding_y = -PConvLayer->padding;
	uint16_t in_x, in_y;
	TPVolume inVolume = PConvLayer->layer.in_v;
	TPVolume outVolume = PConvLayer->layer.out_v;
	inVolume->fillZero(inVolume->weight_d);

	for (uint32_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
	{
		TPVolume filter = PConvLayer->filters->volumes[out_d];
		padding_x = -PConvLayer->padding;
		padding_y = -PConvLayer->padding;
		for (uint32_t out_y = 0; out_y < PConvLayer->layer.out_h; out_y++)
		{
			padding_x = -PConvLayer->padding;
			for (uint32_t out_x = 0; out_x < PConvLayer->layer.out_w; out_x++)
			{
				out_grad = outVolume->getGradValue(outVolume, out_x, out_y, out_d);
				// assert(!IsFloatUnderflow(out_grad));
				for (uint32_t filter_y = 0; filter_y < PConvLayer->filters->_h; filter_y++)
				{
					in_y = filter_y + padding_y;
					if (in_y < 0 || in_y >= inVolume->_h)
						continue;

					for (uint32_t filter_x = 0; filter_x < PConvLayer->filters->_w; filter_x++)
					{
						in_x = filter_x + padding_x;
						if (in_x < 0 || in_x >= inVolume->_w)
							continue;

						uint32_t fx = (filter->_w * filter_y + filter_x) * filter->_depth;
						uint32_t ix = (inVolume->_w * in_y + in_x) * inVolume->_depth;
						for (uint32_t filter_d = 0; filter_d < PConvLayer->filters->_depth; filter_d++)
						{
							filter->weight_d->buffer[fx + filter_d] = filter->weight_d->buffer[fx + filter_d] + inVolume->weight->buffer[ix + filter_d] * out_grad;
							inVolume->weight_d->buffer[ix + filter_d] = inVolume->weight_d->buffer[ix + filter_d] + filter->weight->buffer[fx + filter_d] * out_grad;
						}
					}
				}
				PConvLayer->biases->weight_d->buffer[out_d] = PConvLayer->biases->weight_d->buffer[out_d] + out_grad;
				padding_x = padding_x + PConvLayer->stride;
			}
			padding_y = padding_y + PConvLayer->stride;
		}
	}
}

TPResponse *ConvolutionLayerGetParamsAndGradients(TPConvLayer PConvLayer)
{
	if (PConvLayer->layer.out_depth <= 0)
		return NULL;
	TPResponse *tPResponses = malloc(sizeof(TPResponse) * (PConvLayer->layer.out_depth + 1));
	if (tPResponses == NULL)
		return NULL;
	for (uint32_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
	{
		TPResponse PResponse = malloc(sizeof(TResponse));
		if (PResponse != NULL)
		{
			PResponse->filterWeight = PConvLayer->filters->volumes[out_d]->weight;
			PResponse->filterGrads = PConvLayer->filters->volumes[out_d]->weight_d;
			PResponse->l1_decay_rate = PConvLayer->l1_decay_rate;
			PResponse->l2_decay_rate = PConvLayer->l2_decay_rate;
			PResponse->fillZero = TensorFillZero;
			PResponse->free = TensorFree;
			tPResponses[out_d] = PResponse;
		}
	}

	TPResponse PResponse = malloc(sizeof(TResponse));
	if (PResponse != NULL)
	{
		PResponse->filterWeight = PConvLayer->biases->weight;
		PResponse->filterGrads = PConvLayer->biases->weight_d;
		PResponse->l1_decay_rate = 0;
		PResponse->l2_decay_rate = 0;
		PResponse->fillZero = TensorFillZero;
		PResponse->free = TensorFree;
		tPResponses[PConvLayer->layer.out_depth] = PResponse;
	}
	return tPResponses;
}

float32_t ConvolutionLayerBackwardLoss(TPConvLayer PConvLayer, int Y)
{
	return 0.00;
}

void ConvolutionLayerFree(TPConvLayer PConvLayer)
{
	VolumeFree(PConvLayer->layer.in_v);
	VolumeFree(PConvLayer->layer.out_v);
	VolumeFree(PConvLayer->biases);
	FiltersFree(PConvLayer->filters);

	if (PConvLayer->filters != NULL)
		free(PConvLayer->filters);
	free(PConvLayer);
}

/////////////////////////////////////////////////////////////////////////////
/// @brief //////////////////////////////////////////////////////////////////
/// @param PReluLayer
/// @param PLayerOption
void ReluLayerInit(TPReluLayer PReluLayer, TPLayerOption PLayerOption)
{
	PReluLayer->layer.LayerType = PLayerOption->LayerType;
	// PReluLayer->l1_decay_rate = LayerOption.l1_decay_rate;
	// PReluLayer->l2_decay_rate = LayerOption.l2_decay_rate;
	// PReluLayer->stride = LayerOption.stride;
	// PReluLayer->padding = LayerOption.padding;
	// PReluLayer->bias = LayerOption.bias;

	PReluLayer->layer.in_w = PLayerOption->in_w;
	PReluLayer->layer.in_h = PLayerOption->in_h;
	PReluLayer->layer.in_depth = PLayerOption->in_depth;
	PReluLayer->layer.in_v = NULL;

	PReluLayer->layer.out_w = PReluLayer->layer.in_w;
	PReluLayer->layer.out_h = PReluLayer->layer.in_h;
	PReluLayer->layer.out_depth = PReluLayer->layer.in_depth;

	PReluLayer->layer.out_v = MakeVolume(PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth);
	PReluLayer->layer.out_v->init(PReluLayer->layer.out_v, PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth, 0);

	PLayerOption->out_w = PReluLayer->layer.out_w;
	PLayerOption->out_h = PReluLayer->layer.out_h;
	PLayerOption->out_depth = PReluLayer->layer.out_depth;
}

void reluLayerOutResize(TPReluLayer PReluLayer)
{
	if (PReluLayer->layer.out_w != PReluLayer->layer.in_w || PReluLayer->layer.out_h != PReluLayer->layer.in_h || PReluLayer->layer.out_depth != PReluLayer->layer.in_depth)
	{
		LOGINFOR("ReluLayer resize out_v from %d x %d x %d to %d x %d x %d", PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth, PReluLayer->layer.in_w, PReluLayer->layer.in_h, PReluLayer->layer.in_depth);
		PReluLayer->layer.out_w = PReluLayer->layer.in_w;
		PReluLayer->layer.out_h = PReluLayer->layer.in_h;
		PReluLayer->layer.out_depth = PReluLayer->layer.in_depth;
		if (PReluLayer->layer.out_v != NULL)
		{
			VolumeFree(PReluLayer->layer.out_v);
		}
		PReluLayer->layer.out_v = MakeVolume(PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth);
		PReluLayer->layer.out_v->init(PReluLayer->layer.out_v, PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth, 0);
	}
}

void ReluLayerForward(TPReluLayer PReluLayer)
{
	/*for (uint16_t out_d = 0; out_d < PReluLayer->layer.out_depth; out_d++) {
	 for (uint16_t out_y = 0; out_y < PReluLayer->layer.out_h; out_y++) {
	 for (uint16_t out_x = 0; out_x < PReluLayer->layer.out_w; out_x++) {
	 }
	 }
	 }*/
	reluLayerOutResize(PReluLayer);
	for (uint32_t out_l = 0; out_l < PReluLayer->layer.out_v->weight->length; out_l++)
	{
		if (PReluLayer->layer.in_v->weight->buffer[out_l] < 0)
			PReluLayer->layer.out_v->weight->buffer[out_l] = 0;
		else
			PReluLayer->layer.out_v->weight->buffer[out_l] = PReluLayer->layer.in_v->weight->buffer[out_l];
	}
}
/// @brief //////////////////////////////////////////////////////////////////////////////////////
/// @param PReluLayer /

void ReluLayerBackward(TPReluLayer PReluLayer)
{
	for (uint32_t out_l = 0; out_l < PReluLayer->layer.in_v->weight->length; out_l++)
	{
		if (PReluLayer->layer.out_v->weight->buffer[out_l] <= 0)
			PReluLayer->layer.in_v->weight_d->buffer[out_l] = 0;
		else
			PReluLayer->layer.in_v->weight_d->buffer[out_l] = PReluLayer->layer.out_v->weight_d->buffer[out_l];
	}
}

float32_t ReluLayerBackwardLoss(TPReluLayer PReluLayer, int Y)
{
	return 0.00;
}

void ReluLayerFree(TPReluLayer PReluLayer)
{
	VolumeFree(PReluLayer->layer.in_v);
	VolumeFree(PReluLayer->layer.out_v);
	free(PReluLayer);
}
////////////////////////////////////////////////////////////////////////////
/// @brief /////////////////////////////////////////////////////////////////
/// @param PPoolLayer
/// @param PLayerOption /
void PoolLayerInit(TPPoolLayer PPoolLayer, TPLayerOption PLayerOption)
{
	PPoolLayer->layer.LayerType = PLayerOption->LayerType;
	// PPoolLayer->l1_decay_rate = PLayerOption->l1_decay_rate;
	// PPoolLayer->l2_decay_rate = PLayerOption->l2_decay_rate;
	PPoolLayer->stride = PLayerOption->stride;
	PPoolLayer->padding = PLayerOption->padding;
	// PPoolLayer->bias = PLayerOption->bias;
	// 池化核不需要分配张量空间
	PPoolLayer->filter = MakeVolume(PLayerOption->filter_w, PLayerOption->filter_h, PLayerOption->filter_depth);

	PPoolLayer->layer.in_w = PLayerOption->in_w;
	PPoolLayer->layer.in_h = PLayerOption->in_h;
	PPoolLayer->layer.in_depth = PLayerOption->in_depth;
	PPoolLayer->layer.in_v = NULL;

	PPoolLayer->layer.out_w = floor((PPoolLayer->layer.in_w + PPoolLayer->padding * 2 - PPoolLayer->filter->_w) / PPoolLayer->stride + 1);
	PPoolLayer->layer.out_h = floor((PPoolLayer->layer.in_h + PPoolLayer->padding * 2 - PPoolLayer->filter->_h) / PPoolLayer->stride + 1);
	PPoolLayer->layer.out_depth = PPoolLayer->layer.in_depth;

	PPoolLayer->layer.out_v = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
	PPoolLayer->layer.out_v->init(PPoolLayer->layer.out_v, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0.0);

	PPoolLayer->switchxy = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
	PPoolLayer->switchxy->init(PPoolLayer->switchxy, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);
	// PPoolLayer->switchy = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
	// PPoolLayer->switchy->init(PPoolLayer->switchy, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);
	// PPoolLayer->switchy->free(PPoolLayer->switchy->weight_d);
	// PPoolLayer->switchx->free(PPoolLayer->switchx->weight_d);
	// uint16_t out_length = PPoolLayer->layer.out_w * PPoolLayer->layer.out_h * PPoolLayer->layer.out_depth;
	// PPoolLayer->switchx = MakeTensor(out_length);
	// PPoolLayer->switchy = MakeTensor(out_length);

	PLayerOption->out_w = PPoolLayer->layer.out_w;
	PLayerOption->out_h = PPoolLayer->layer.out_h;
	PLayerOption->out_depth = PPoolLayer->layer.out_depth;
}

void poolLayerOutResize(TPPoolLayer PPoolLayer)
{
	uint32_t out_w = floor((PPoolLayer->layer.in_w + PPoolLayer->padding * 2 - PPoolLayer->filter->_w) / PPoolLayer->stride + 1);
	uint32_t out_h = floor((PPoolLayer->layer.in_h + PPoolLayer->padding * 2 - PPoolLayer->filter->_h) / PPoolLayer->stride + 1);
	if (PPoolLayer->layer.out_w != out_w || PPoolLayer->layer.out_h != out_h || PPoolLayer->layer.out_depth != PPoolLayer->layer.in_depth)
	{
		LOGINFOR("PoolLayer resize out_v from %d x %d x %d to %d x %d x %d", PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, out_w, out_h, PPoolLayer->layer.in_depth);
		PPoolLayer->layer.out_w = out_w;
		PPoolLayer->layer.out_h = out_h;
		PPoolLayer->layer.out_depth = PPoolLayer->layer.in_depth;
		if (PPoolLayer->layer.out_v != NULL)
		{
			VolumeFree(PPoolLayer->layer.out_v);
			VolumeFree(PPoolLayer->switchxy);
			// VolumeFree(PPoolLayer->switchy);
		}
		PPoolLayer->layer.out_v = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
		PPoolLayer->layer.out_v->init(PPoolLayer->layer.out_v, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);

		PPoolLayer->switchxy = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
		PPoolLayer->switchxy->init(PPoolLayer->switchxy, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);
		// PPoolLayer->switchy = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
		// PPoolLayer->switchy->init(PPoolLayer->switchy, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);
		// PPoolLayer->switchy->free(PPoolLayer->switchy->weight_d);
		// PPoolLayer->switchx->free(PPoolLayer->switchx->weight_d);
	}
}

void PoolLayerForward(TPPoolLayer PPoolLayer)
{
	float32_t max_value = MINFLOAT_NEGATIVE_NUMBER;
	float32_t value = 0;
	uint16_t x = -PPoolLayer->padding;
	uint16_t y = -PPoolLayer->padding;
	uint16_t ox, oy, inx, iny;
	TPVolume inVolu = PPoolLayer->layer.in_v;
	TPVolume outVolu = PPoolLayer->layer.out_v;
	poolLayerOutResize(PPoolLayer);
	outVolu->fillZero(outVolu->weight);

	for (uint32_t out_d = 0; out_d < PPoolLayer->layer.out_depth; out_d++)
	{
		x = -PPoolLayer->padding;
		y = -PPoolLayer->padding;
		for (uint32_t out_y = 0; out_y < PPoolLayer->layer.out_h; out_y++)
		{
			x = -PPoolLayer->padding;

			for (uint32_t out_x = 0; out_x < PPoolLayer->layer.out_w; out_x++)
			{
				max_value = MINFLOAT_NEGATIVE_NUMBER;
				inx = -1;
				iny = -1;
				for (uint32_t filter_y = 0; filter_y < PPoolLayer->filter->_h; filter_y++)
				{
					oy = filter_y + y;
					if (oy < 0 && oy >= inVolu->_h)
						continue;
					for (uint32_t filter_x = 0; filter_x < PPoolLayer->filter->_w; filter_x++)
					{
						ox = filter_x + x;
						if (ox >= 0 && ox < inVolu->_w && oy >= 0 && oy < inVolu->_h)
						{
							value = inVolu->getValue(inVolu, ox, oy, out_d);
							if (value > max_value)
							{
								max_value = value;
								inx = ox;
								iny = oy;
							}
						}
					}
				}
				outVolu->setValue(outVolu, out_x, out_y, out_d, max_value);
				PPoolLayer->switchxy->setValue(PPoolLayer->switchxy, out_x, out_y, out_d, inx);
				PPoolLayer->switchxy->setGradValue(PPoolLayer->switchxy, out_x, out_y, out_d, iny);
				x = x + PPoolLayer->stride;
			}
			y = y + PPoolLayer->stride;
		}
	}
}
/// @brief ////////////////////////////////////////////////////////////////////////////////////
/// @param PPoolLayer /
///
void PoolLayerBackward(TPPoolLayer PPoolLayer)
{
	float32_t grad_value = 0.00;
	TPVolume inVolu = PPoolLayer->layer.in_v;
	TPVolume outVolu = PPoolLayer->layer.out_v;
	inVolu->fillZero(inVolu->weight_d);
	uint32_t x, y;
	for (uint32_t out_d = 0; out_d < PPoolLayer->layer.out_depth; out_d++)
	{
		x = -PPoolLayer->padding;
		y = -PPoolLayer->padding;
		for (uint32_t out_y = 0; out_y < PPoolLayer->layer.out_h; out_y++)
		{
			x = -PPoolLayer->padding;
			for (uint32_t out_x = 0; out_x < PPoolLayer->layer.out_w; out_x++)
			{
				grad_value = outVolu->getGradValue(outVolu, out_x, out_y, out_d);
				uint32_t ox = PPoolLayer->switchxy->getValue(PPoolLayer->switchxy, out_x, out_y, out_d);
				uint32_t oy = PPoolLayer->switchxy->getGradValue(PPoolLayer->switchxy, out_x, out_y, out_d);
				inVolu->addGradValue(inVolu, ox, oy, out_d, grad_value);
				x = x + PPoolLayer->stride;
			}
			y = y + PPoolLayer->stride;
		}
	}
}

float32_t PoolLayerBackwardLoss(TPPoolLayer PPoolLayer, int Y)
{
	return 0.00;
}

void PoolLayerFree(TPPoolLayer PPoolLayer)
{
	VolumeFree(PPoolLayer->layer.in_v);
	VolumeFree(PPoolLayer->layer.out_v);
	VolumeFree(PPoolLayer->switchxy);
	// VolumeFree(PPoolLayer->switchy);
	free(PPoolLayer);
}
////////////////////////////////////////////////////////////////////////////////
// FullyConnLayer
/// @brief ////////////////////////////////////////////////////////////////////
/// @param PFullyConnLayer
/// @param PLayerOption
void FullyConnLayerInit(TPFullyConnLayer PFullyConnLayer, TPLayerOption PLayerOption)
{
	PFullyConnLayer->layer.LayerType = PLayerOption->LayerType;
	PFullyConnLayer->l1_decay_rate = PLayerOption->l1_decay_rate;
	PFullyConnLayer->l2_decay_rate = PLayerOption->l2_decay_rate;
	// PFullyConnLayer->stride = LayerOption.stride;
	// PFullyConnLayer->padding = LayerOption.padding;
	PFullyConnLayer->bias = PLayerOption->bias;
	// PFullyConnLayer->filter._w = LayerOption.filter_w;
	// PFullyConnLayer->filter._h = LayerOption.filter_h;
	// PFullyConnLayer->filter._depth = LayerOption.filter_depth;
	PLayerOption->filter_w = 1;
	PLayerOption->filter_h = 1;

	PFullyConnLayer->layer.in_w = PLayerOption->in_w;
	PFullyConnLayer->layer.in_h = PLayerOption->in_h;
	PFullyConnLayer->layer.in_depth = PLayerOption->in_depth;
	PFullyConnLayer->layer.in_v = NULL;

	uint32_t inputPoints = PFullyConnLayer->layer.in_w * PFullyConnLayer->layer.in_h * PFullyConnLayer->layer.in_depth;

	PFullyConnLayer->filters = MakeFilters(PLayerOption->filter_w, PLayerOption->filter_h, inputPoints, PLayerOption->filter_number);

	for (uint32_t i = 0; i < PFullyConnLayer->filters->filterNumber; i++)
	{
		PFullyConnLayer->filters->init(PFullyConnLayer->filters->volumes[i], PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputPoints, 0);
		PFullyConnLayer->filters->volumes[i]->fillGauss(PFullyConnLayer->filters->volumes[i]->weight);
	}

	PFullyConnLayer->layer.out_w = 1;
	PFullyConnLayer->layer.out_h = 1;
	PFullyConnLayer->layer.out_depth = PFullyConnLayer->filters->filterNumber;
	PFullyConnLayer->layer.out_v = MakeVolume(PFullyConnLayer->layer.out_w, PFullyConnLayer->layer.out_h, PFullyConnLayer->layer.out_depth);
	PFullyConnLayer->layer.out_v->init(PFullyConnLayer->layer.out_v, PFullyConnLayer->layer.out_w, PFullyConnLayer->layer.out_h,
									   PFullyConnLayer->layer.out_depth, 0);

	PFullyConnLayer->biases = MakeVolume(1, 1, PFullyConnLayer->layer.out_depth);
	PFullyConnLayer->biases->init(PFullyConnLayer->biases, 1, 1, PFullyConnLayer->layer.out_depth, PFullyConnLayer->bias);

	PLayerOption->out_w = PFullyConnLayer->layer.out_w;
	PLayerOption->out_h = PFullyConnLayer->layer.out_h;
	PLayerOption->out_depth = PFullyConnLayer->layer.out_depth;
}

void fullConnLayerOutResize(TPFullyConnLayer PFullyConnLayer)
{
	uint32_t inputLength = PFullyConnLayer->layer.in_w * PFullyConnLayer->layer.in_h * PFullyConnLayer->layer.in_depth;
	if (PFullyConnLayer->filters->_depth != inputLength)
	{
		LOGINFOR("FullyConnLayer resize filters from %d x %d x %d to %d x %d x %d", PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, PFullyConnLayer->filters->_depth, PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputLength);
		FiltersFree(PFullyConnLayer->filters);
		bool ret = FiltersResize(PFullyConnLayer->filters, PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputLength, PFullyConnLayer->filters->filterNumber);
		if (ret)
		{
			for (uint16_t i = 0; i < PFullyConnLayer->filters->filterNumber; i++)
			{
				PFullyConnLayer->filters->init(PFullyConnLayer->filters->volumes[i], PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputLength, 0);
				PFullyConnLayer->filters->volumes[i]->fillGauss(PFullyConnLayer->filters->volumes[i]->weight);
			}
		}
	}
}
void FullyConnLayerForward(TPFullyConnLayer PFullyConnLayer)
{
	float32_t sum = 0.00;
	TPVolume inVolu = PFullyConnLayer->layer.in_v;
	TPVolume outVolu = PFullyConnLayer->layer.out_v;
	outVolu->fillZero(outVolu->weight);
	fullConnLayerOutResize(PFullyConnLayer);

	for (uint32_t out_d = 0; out_d < PFullyConnLayer->layer.out_depth; out_d++)
	{
		TPVolume filter = PFullyConnLayer->filters->volumes[out_d];
		sum = 0.00;

		uint32_t inputPoints = inVolu->_w * inVolu->_h * inVolu->_depth;

		for (uint32_t ip = 0; ip < inputPoints; ip++)
		{
			if (filter->weight->length == 0)
				sum = sum + inVolu->weight->buffer[ip];
			else
				sum = sum + inVolu->weight->buffer[ip] * filter->weight->buffer[ip];
		}

		sum = sum + PFullyConnLayer->biases->weight->buffer[out_d];
		outVolu->weight->buffer[out_d] = sum;
	}
}

/// @brief /////////////////////////////////////////////////////////////////////////////////////////////////
/// @param PFullyConnLayer
/// y = wx+b 求关于in w b的偏导数
void FullyConnLayerBackward(TPFullyConnLayer PFullyConnLayer)
{
	float32_t grad_value = 0.00;
	TPVolume inVolu = PFullyConnLayer->layer.in_v;
	TPVolume outVolu = PFullyConnLayer->layer.out_v;
	inVolu->fillZero(inVolu->weight_d);

	for (uint32_t out_d = 0; out_d < PFullyConnLayer->layer.out_depth; out_d++)
	{
		TPVolume filter = PFullyConnLayer->filters->volumes[out_d];
		grad_value = outVolu->weight_d->buffer[out_d];

		uint32_t inputPoints = inVolu->_w * inVolu->_h * inVolu->_depth;

		for (uint32_t out_l = 0; out_l < inputPoints; out_l++)
		{
			inVolu->weight_d->buffer[out_l] = inVolu->weight_d->buffer[out_l] + filter->weight->buffer[out_l] * grad_value;
			filter->weight_d->buffer[out_l] = filter->weight_d->buffer[out_l] + inVolu->weight->buffer[out_l] * grad_value;
		}
		PFullyConnLayer->biases->weight_d->buffer[out_d] = PFullyConnLayer->biases->weight_d->buffer[out_d] * grad_value;
	}
}

float32_t FullyConnLayerBackwardLoss(TPFullyConnLayer PFullyConnLayer, int Y)
{
	return 0.00;
}

TPResponse *FullyConnLayerGetParamsAndGrads(TPFullyConnLayer PFullyConnLayer)
{
	if (PFullyConnLayer->layer.out_depth <= 0)
		return NULL;
	TPResponse *tPResponses = malloc(sizeof(TPResponse) * (PFullyConnLayer->layer.out_depth + 1));
	if (tPResponses == NULL)
		return NULL;
	for (uint32_t out_d = 0; out_d < PFullyConnLayer->layer.out_depth; out_d++)
	{
		TPResponse PResponse = malloc(sizeof(TResponse));
		if (PResponse != NULL)
		{
			PResponse->filterWeight = PFullyConnLayer->filters->volumes[out_d]->weight;
			PResponse->filterGrads = PFullyConnLayer->filters->volumes[out_d]->weight_d;
			PResponse->l1_decay_rate = PFullyConnLayer->l1_decay_rate;
			PResponse->l2_decay_rate = PFullyConnLayer->l2_decay_rate;
			PResponse->fillZero = TensorFillZero;
			PResponse->free = TensorFree;
			tPResponses[out_d] = PResponse;
		}
	}
	TPResponse PResponse = malloc(sizeof(TResponse));
	if (PResponse != NULL)
	{
		PResponse->filterWeight = PFullyConnLayer->biases->weight;
		PResponse->filterGrads = PFullyConnLayer->biases->weight_d;
		PResponse->l1_decay_rate = 0;
		PResponse->l2_decay_rate = 0;
		PResponse->fillZero = TensorFillZero;
		PResponse->free = TensorFree;
		tPResponses[PFullyConnLayer->layer.out_depth] = PResponse;
	}
	return tPResponses;
}

void FullyConnLayerFree(TPFullyConnLayer PFullyConnLayer)
{
	VolumeFree(PFullyConnLayer->layer.in_v);
	VolumeFree(PFullyConnLayer->layer.out_v);
	VolumeFree(PFullyConnLayer->biases);
	FiltersFree(PFullyConnLayer->filters);
	// for (uint16_t i = 0; i < PFullyConnLayer->filters->_depth; i++)
	//{
	//	PFullyConnLayer->filters->free(PFullyConnLayer->filters->volumes[i]);
	// }
	if (PFullyConnLayer->filters != NULL)
		free(PFullyConnLayer->filters);
	free(PFullyConnLayer);
}

////////////////////////////////////////////////////////////////////////////////
// Softmax
void SoftmaxLayerInit(TPSoftmaxLayer PSoftmaxLayer, TPLayerOption PLayerOption)
{
	PSoftmaxLayer->layer.LayerType = PLayerOption->LayerType;
	PSoftmaxLayer->layer.in_w = PLayerOption->in_w;
	PSoftmaxLayer->layer.in_h = PLayerOption->in_h;
	PSoftmaxLayer->layer.in_depth = PLayerOption->in_depth;
	PSoftmaxLayer->layer.in_v = NULL;

	PSoftmaxLayer->layer.out_w = 1;
	PSoftmaxLayer->layer.out_h = 1;
	PSoftmaxLayer->layer.out_depth = PLayerOption->in_w * PLayerOption->in_h * PLayerOption->in_depth;

	PSoftmaxLayer->layer.out_v = MakeVolume(PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth);
	PSoftmaxLayer->layer.out_v->init(PSoftmaxLayer->layer.out_v, PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth, 0);
	PSoftmaxLayer->exp = MakeTensor(PSoftmaxLayer->layer.out_depth);

	PLayerOption->out_w = PSoftmaxLayer->layer.out_w;
	PLayerOption->out_h = PSoftmaxLayer->layer.out_h;
	PLayerOption->out_depth = PSoftmaxLayer->layer.out_depth;
}
/// @brief ////////////////////////////////////////////////////////////////////////
/// @param PSoftmaxLayer
void softmaxLayOutResize(TPSoftmaxLayer PSoftmaxLayer)
{
	uint16_t inputLength = PSoftmaxLayer->layer.in_depth * PSoftmaxLayer->layer.in_w * PSoftmaxLayer->layer.in_h;
	if (PSoftmaxLayer->layer.out_depth != inputLength)
	{
		LOGINFOR("Softmax resize out_v from %d x %d x %d to %d x %d x %d", PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth, 1, 1, inputLength);
		PSoftmaxLayer->layer.out_w = 1;
		PSoftmaxLayer->layer.out_h = 1;
		PSoftmaxLayer->layer.out_depth = inputLength;

		if (PSoftmaxLayer->layer.out_v != NULL)
		{
			VolumeFree(PSoftmaxLayer->layer.out_v);
			TensorFree(PSoftmaxLayer->exp);
		}
		PSoftmaxLayer->layer.out_v = MakeVolume(PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth);
		PSoftmaxLayer->layer.out_v->init(PSoftmaxLayer->layer.out_v, PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth, 0);
		PSoftmaxLayer->exp = MakeTensor(PSoftmaxLayer->layer.out_depth);
	}
}
/// @brief ////////////////////////////////////////////////////////////////////
/// @param PSoftmaxLayer
/// PSoftmaxLayer->exp Probability Distribution
/// PSoftmaxLayer->layer.out_v->weight Probability Distribution
// 归一化后的真数当作概率分布
void SoftmaxLayerForward(TPSoftmaxLayer PSoftmaxLayer)
{
	float32_t max_value = MINFLOAT_NEGATIVE_NUMBER;
	float32_t sum = 0.0;
	float32_t expv = 0.0;
	float32_t temp = 0.0;
	TPVolume inVolu = PSoftmaxLayer->layer.in_v;
	TPVolume outVolu = PSoftmaxLayer->layer.out_v;

	softmaxLayOutResize(PSoftmaxLayer);
	outVolu->fillZero(outVolu->weight);

	for (uint32_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
	{
		if (inVolu->weight->buffer[out_d] > max_value)
		{
			max_value = inVolu->weight->buffer[out_d];
		}
	}
	for (uint32_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
	{
		temp = inVolu->weight->buffer[out_d] - max_value;
		expv = exp(temp);
		PSoftmaxLayer->exp->buffer[out_d] = expv;
		sum = sum + expv;
	}
	for (uint32_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
	{
		PSoftmaxLayer->exp->buffer[out_d] = PSoftmaxLayer->exp->buffer[out_d] / sum;
		PSoftmaxLayer->layer.out_v->weight->buffer[out_d] = PSoftmaxLayer->exp->buffer[out_d];
	}
}
/// @brief ///////////////////////////////////////////////////////////////////////////////
/// @param PSoftmaxLayer
/// dw为代价函数的值，训练输出与真实值之间差
void SoftmaxLayerBackward(TPSoftmaxLayer PSoftmaxLayer)
{
	TPVolume inVolu = PSoftmaxLayer->layer.in_v;
	TPVolume outVolu = PSoftmaxLayer->layer.out_v;
	inVolu->fillZero(outVolu->weight_d);
	float32_t dw; // 计算 delta weight
	for (uint32_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
	{
		if (out_d == PSoftmaxLayer->expected_value)
			dw = -(1 - PSoftmaxLayer->exp->buffer[out_d]);
		else
			dw = PSoftmaxLayer->exp->buffer[out_d];

		inVolu->weight_d->buffer[out_d] = dw;
	}
}
/// @brief ////////////////////////////////////////////////////////////////////////////
/// @param PSoftmaxLayer
/// @return
// 交叉熵Cross-Entropy损失函数，衡量模型输出的概率分布与真实标签的差异。
// 交叉熵的计算公式如下：
// CE = -Σylog(ŷ)
// 其中，y是真实标签的概率分布，ŷ是模型输出的概率分布。
// 训练使得损失函数的值无限逼近0，对应的幂exp无限接近1
float32_t SoftmaxLayerBackwardLoss(TPSoftmaxLayer PSoftmaxLayer)
{
	float32_t exp = PSoftmaxLayer->exp->buffer[PSoftmaxLayer->expected_value];
	if (exp > 0)
		return -log10(exp);
	else
		return 0;
}

void SoftmaxLayerFree(TPSoftmaxLayer PSoftmaxLayer)
{
	VolumeFree(PSoftmaxLayer->layer.in_v);
	VolumeFree(PSoftmaxLayer->layer.out_v);
	TensorFree(PSoftmaxLayer->exp);
	free(PSoftmaxLayer);
}

////////////////////////////////////////////////////////////////////////////////////
/// @brief /////////////////////////////////////////////////////////////////////////
/// @param PNeuralNet
/// @param PLayerOption
void NeuralNetInit(TPNeuralNet PNeuralNet, TPLayerOption PLayerOption)
{
	if (PNeuralNet == NULL)
		return;
	void *pLayer = NULL;
	switch (PLayerOption->LayerType)
	{
	case Layer_Type_Input:
	{
		PNeuralNet->depth = 1;
		// free(PNeuralNet->layers);
		PNeuralNet->layers = malloc(sizeof(TPLayer) * PNeuralNet->depth);
		if (PNeuralNet->layers != NULL)
		{
			TPInputLayer InputLayer = malloc(sizeof(TInputLayer));
			InputLayer->init = InputLayerInit;
			InputLayer->free = InputLayerFree;
			InputLayer->forward = InputLayerForward;
			InputLayer->backward = InputLayerBackward;
			InputLayer->computeLoss = NULL; // InputLayerBackwardLoss;
			// InputLayer->backwardOutput = NULL; // InputLayerBackwardOutput;
			InputLayerInit(InputLayer, PLayerOption);
			PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)InputLayer;
		}
		break;
	}
	case Layer_Type_Convolution:
	{
		PNeuralNet->depth++;
		pLayer = realloc(PNeuralNet->layers, sizeof(TPLayer) * PNeuralNet->depth);
		if (pLayer == NULL)
			break;
		PNeuralNet->layers = pLayer;
		if (PNeuralNet->layers != NULL)
		{
			TPConvLayer ConvLayer = malloc(sizeof(TConvLayer));
			ConvLayer->init = ConvolutionLayerInit;
			ConvLayer->free = ConvolutionLayerFree;
			ConvLayer->forward = ConvolutionLayerForward;
			ConvLayer->backward = ConvolutionLayerBackward;
			ConvLayer->computeLoss = ConvolutionLayerBackwardLoss;
			// ConvLayer->backwardOutput = ConvolutionLayerBackwardOutput;
			ConvLayer->getWeightsAndGrads = ConvolutionLayerGetParamsAndGradients;
			ConvolutionLayerInit(ConvLayer, PLayerOption);
			PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)ConvLayer;
		}
		break;
	}
	case Layer_Type_Pool:
	{
		PNeuralNet->depth++;
		pLayer = realloc(PNeuralNet->layers, sizeof(TPLayer) * PNeuralNet->depth);
		if (pLayer == NULL)
			break;
		PNeuralNet->layers = pLayer;
		if (PNeuralNet->layers != NULL)
		{
			TPPoolLayer PoolLayer = malloc(sizeof(TPoolLayer));
			PoolLayer->init = PoolLayerInit;
			PoolLayer->free = PoolLayerFree;
			PoolLayer->forward = PoolLayerForward;
			PoolLayer->backward = PoolLayerBackward;
			PoolLayer->computeLoss = PoolLayerBackwardLoss;
			// PoolLayer->backwardOutput = PoolLayerBackwardOutput;
			PoolLayerInit(PoolLayer, PLayerOption);
			PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)PoolLayer;
		}
		break;
	}
	case Layer_Type_ReLu:
	{
		PNeuralNet->depth++;
		pLayer = realloc(PNeuralNet->layers, sizeof(TPLayer) * PNeuralNet->depth);
		if (pLayer == NULL)
			break;
		PNeuralNet->layers = pLayer;
		if (PNeuralNet->layers != NULL)
		{
			TPReluLayer ReluLayer = malloc(sizeof(TReluLayer));
			ReluLayer->init = ReluLayerInit;
			ReluLayer->free = ReluLayerFree;
			ReluLayer->forward = ReluLayerForward;
			ReluLayer->backward = ReluLayerBackward;
			ReluLayer->computeLoss = ReluLayerBackwardLoss;
			// ReluLayer->backwardOutput = ReluLayerBackwardOutput;
			ReluLayerInit(ReluLayer, PLayerOption);
			PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)ReluLayer;
		}
		break;
	}
	case Layer_Type_FullyConnection:
	{
		PNeuralNet->depth++;
		pLayer = realloc(PNeuralNet->layers, sizeof(TPLayer) * PNeuralNet->depth);
		if (pLayer == NULL)
			break;
		PNeuralNet->layers = pLayer;
		if (PNeuralNet->layers != NULL)
		{
			TPFullyConnLayer FullyConnLayer = malloc(sizeof(TFullyConnLayer));
			FullyConnLayer->init = FullyConnLayerInit;
			FullyConnLayer->free = FullyConnLayerFree;
			FullyConnLayer->forward = FullyConnLayerForward;
			FullyConnLayer->backward = FullyConnLayerBackward;
			FullyConnLayer->computeLoss = FullyConnLayerBackwardLoss;
			// FullyConnLayer->backwardOutput = FullyConnLayerBackwardOutput;
			FullyConnLayer->getWeightsAndGrads = FullyConnLayerGetParamsAndGrads;
			FullyConnLayerInit(FullyConnLayer, PLayerOption);
			PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)FullyConnLayer;
		}
		break;
	}
	case Layer_Type_SoftMax:
	{
		PNeuralNet->depth++;
		pLayer = realloc(PNeuralNet->layers, sizeof(TPLayer) * PNeuralNet->depth);
		if (pLayer == NULL)
			break;
		PNeuralNet->layers = pLayer;
		if (PNeuralNet->layers != NULL)
		{
			TPSoftmaxLayer SoftmaxLayer = malloc(sizeof(TSoftmaxLayer));
			SoftmaxLayer->init = SoftmaxLayerInit;
			SoftmaxLayer->free = SoftmaxLayerFree;
			SoftmaxLayer->forward = SoftmaxLayerForward;
			SoftmaxLayer->backward = SoftmaxLayerBackward;
			SoftmaxLayer->computeLoss = SoftmaxLayerBackwardLoss;
			// SoftmaxLayer.backwardOutput = SoftmaxLayerBackwardOutput;
			SoftmaxLayerInit(SoftmaxLayer, PLayerOption);
			PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)SoftmaxLayer;
		}
		break;
	}
	default:
	{
		break;
	}
	}
}

void NeuralNetFree(TPNeuralNet PNeuralNet)
{
	if (PNeuralNet == NULL)
		return;
	for (uint16_t layerIndex = 0; layerIndex < PNeuralNet->depth; layerIndex++)
	{
		switch (PNeuralNet->layers[layerIndex]->LayerType)
		{
		case Layer_Type_Input:
		{
			InputLayerFree((TPInputLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_Convolution:
		{
			ConvolutionLayerFree((TPConvLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_Pool:
		{
			PoolLayerFree((TPPoolLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_ReLu:
		{
			ReluLayerFree((TPReluLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_FullyConnection:
		{
			FullyConnLayerFree((TPFullyConnLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_SoftMax:
		{
			SoftmaxLayerFree((TPSoftmaxLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		default:
		{
			break;
		}
		}
	}
	free(PNeuralNet);
	PNeuralNet = NULL;
}

void NeuralNetForward(TPNeuralNet PNeuralNet, TPVolume PVolume)
{
	if (PVolume == NULL)
		return;
	TPVolume out_v = NULL;
	TPInputLayer PInputLayer = (TPInputLayer)PNeuralNet->layers[0];
	if (PInputLayer->layer.LayerType != Layer_Type_Input)
	{
		return;
	}

	PInputLayer->forward(PInputLayer, PVolume);

	for (uint16_t layerIndex = 1; layerIndex < PNeuralNet->depth; layerIndex++)
	{
		// PNeuralNet->layers[layerIndex]->in_v->_w = PNeuralNet->layers[layerIndex - 1]->out_w;
		// PNeuralNet->layers[layerIndex]->in_v->_h = PNeuralNet->layers[layerIndex - 1]->out_h;
		// PNeuralNet->layers[layerIndex]->in_v->_depth = PNeuralNet->layers[layerIndex - 1]->out_depth;
		out_v = PNeuralNet->layers[layerIndex - 1]->out_v;
		if (out_v == NULL)
		{
			LOGERROR("Input volume is null, layerIndex=%d layereType=%d\n", layerIndex, PNeuralNet->layers[layerIndex - 1]->LayerType);
			break;
		}

		PNeuralNet->layers[layerIndex]->in_v = out_v;

		switch (PNeuralNet->layers[layerIndex]->LayerType)
		{
		case Layer_Type_Input:
		{
			//((TPInputLayer) PNeuralNet->Layers[layerIndex])->forward((TPInputLayer) PNeuralNet->Layers[layerIndex]);
			break;
		}
		case Layer_Type_Convolution:
		{
			((TPConvLayer)PNeuralNet->layers[layerIndex])->forward((TPConvLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_Pool:
		{
			((TPPoolLayer)PNeuralNet->layers[layerIndex])->forward((TPPoolLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_ReLu:
		{
			((TPReluLayer)PNeuralNet->layers[layerIndex])->forward((TPReluLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_FullyConnection:
		{
			((TPFullyConnLayer)PNeuralNet->layers[layerIndex])->forward((TPFullyConnLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_SoftMax:
		{
			((TPSoftmaxLayer)PNeuralNet->layers[layerIndex])->forward((TPSoftmaxLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		default:
			break;
		}
	}
}

void NeuralNetBackward(TPNeuralNet PNeuralNet)
{

	for (uint16_t layerIndex = PNeuralNet->depth - 1; layerIndex >= 0; layerIndex--)
	{
		switch (PNeuralNet->layers[layerIndex]->LayerType)
		{
		case Layer_Type_Input:
		{
			((TPInputLayer)PNeuralNet->layers[layerIndex])->backward((TPInputLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_Convolution:
		{
			((TPConvLayer)PNeuralNet->layers[layerIndex])->backward((TPConvLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_Pool:
		{
			((TPPoolLayer)PNeuralNet->layers[layerIndex])->backward((TPPoolLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_ReLu:
		{
			((TPReluLayer)PNeuralNet->layers[layerIndex])->backward((TPReluLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_FullyConnection:
		{
			((TPFullyConnLayer)PNeuralNet->layers[layerIndex])->backward((TPFullyConnLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		case Layer_Type_SoftMax:
		{
			((TPSoftmaxLayer)PNeuralNet->layers[layerIndex])->backward((TPSoftmaxLayer)PNeuralNet->layers[layerIndex]);
			break;
		}
		default:
			break;
		}
	}
}

void NeuralNetGetWeightsAndGrads(TPNeuralNet PNeuralNet)
{
	bool weight_flow = false;
	bool grads_flow = false;
	void *temp = NULL;
	if (PNeuralNet->trainning.pResponseResults != NULL)
	{
		free(PNeuralNet->trainning.pResponseResults);
		PNeuralNet->trainning.responseCount = 0;
		PNeuralNet->trainning.pResponseResults = NULL;
	}
	if (PNeuralNet->trainning.pResponseResults == NULL)
		temp = malloc(sizeof(TPResponse));

	for (uint16_t layerIndex = 1; layerIndex < PNeuralNet->depth; layerIndex++)
	{
		switch (PNeuralNet->layers[layerIndex]->LayerType)
		{
		case Layer_Type_Input:
		{
			// pResponseResult = ((TPInputLayer) PNeuralNet->layers[layerIndex])->getWeightsAndGrads();
			break;
		}

		case Layer_Type_Convolution:
		{
			TPResponse *pResponseResult = ((TPConvLayer)PNeuralNet->layers[layerIndex])->getWeightsAndGrads(((TPConvLayer)PNeuralNet->layers[layerIndex]));
			for (uint16_t i = 0; i <= ((TPConvLayer)PNeuralNet->layers[layerIndex])->layer.out_depth; i++)
			{
				temp = realloc(PNeuralNet->trainning.pResponseResults, sizeof(TPResponse) * (PNeuralNet->trainning.responseCount + 1));
				if (temp != NULL)
				{
#if 0
					for (uint16_t w_l = 0; w_l < pResponseResult[i]->filterWeight->length; w_l++)
					{
						if (IsFloatOverflow(pResponseResult[i]->filterWeight->buffer[w_l]))
						{
							PNeuralNet->trainning.overflow = true;
							weight_flow = true;
						}
					}
					for (uint16_t w_l = 0; w_l < pResponseResult[i]->filterGrads->length; w_l++)
					{
						if (IsFloatOverflow(pResponseResult[i]->filterGrads->buffer[w_l]))
						{
							PNeuralNet->trainning.overflow = true;
							grads_flow = true;
						}
					}
#endif
					PNeuralNet->trainning.pResponseResults = temp;
					PNeuralNet->trainning.pResponseResults[PNeuralNet->trainning.responseCount] = pResponseResult[i];
					PNeuralNet->trainning.responseCount++;
				}
			}
			if (PNeuralNet->trainning.underflow)
				LOGERROR("filterGrads underflow %s %d", NeuralNetGetLayerName(Layer_Type_Convolution), layerIndex);
			if (PNeuralNet->trainning.overflow)
			{
				if (grads_flow)
					LOGERROR("filterGrads overflow %s %d", NeuralNetGetLayerName(Layer_Type_Convolution), layerIndex);
				if (weight_flow)
					LOGERROR("filterWeight overflow %s %d", NeuralNetGetLayerName(Layer_Type_Convolution), layerIndex);
			}
			free(pResponseResult);
			break;
		}

		case Layer_Type_Pool:
		{
			// pResponseResult = ((TPPoolLayer) PNeuralNet->layers[layerIndex])->getWeightsAndGrads();
			break;
		}

		case Layer_Type_ReLu:
		{
			// pResponseResult = ((TPReluLayer) PNeuralNet->layers[layerIndex])->getWeightsAndGrads();
			break;
		}

		case Layer_Type_FullyConnection:
		{
			TPResponse *pResponseResult = ((TPFullyConnLayer)PNeuralNet->layers[layerIndex])->getWeightsAndGrads(((TPFullyConnLayer)PNeuralNet->layers[layerIndex]));
			for (uint16_t i = 0; i <= ((TPConvLayer)PNeuralNet->layers[layerIndex])->layer.out_depth; i++)
			{
				temp = realloc(PNeuralNet->trainning.pResponseResults, sizeof(TPResponse) * (PNeuralNet->trainning.responseCount + 1));
				if (temp != NULL)
				{
#if 0
					for (uint16_t w_l = 0; w_l < pResponseResult[i]->filterWeight->length; w_l++)
					{
						if (IsFloatOverflow(pResponseResult[i]->filterWeight->buffer[w_l]))
						{
							PNeuralNet->trainning.overflow = true;
							weight_flow = true;
						}
					}
					for (uint16_t w_l = 0; w_l < pResponseResult[i]->filterWeight->length; w_l++)
					{
						if (IsFloatOverflow(pResponseResult[i]->filterGrads->buffer[w_l]))
						{
							PNeuralNet->trainning.overflow = true;
							grads_flow = true;
						}
					}
#endif
					PNeuralNet->trainning.pResponseResults = temp;
					PNeuralNet->trainning.pResponseResults[PNeuralNet->trainning.responseCount] = pResponseResult[i];
					PNeuralNet->trainning.responseCount++;
				}
			}

			if (PNeuralNet->trainning.underflow)
				LOGERROR("filterGrads underflow %s %d", NeuralNetGetLayerName(Layer_Type_FullyConnection), layerIndex);
			if (PNeuralNet->trainning.overflow)
			{
				if (grads_flow)
					LOGERROR("filterGrads overflow %s %d", NeuralNetGetLayerName(Layer_Type_FullyConnection), layerIndex);
				if (weight_flow)
					LOGERROR("filterWeight overflow %s %d", NeuralNetGetLayerName(Layer_Type_FullyConnection), layerIndex);
			}
			free(pResponseResult);
			break;
		}

		case Layer_Type_SoftMax:
		{
			// pResponseResult = ((TPSoftmaxLayer) PNeuralNet->Layers[layerIndex])->getWeightsAndGrads();
			break;
		}
		default:
			break;
		}
	}
}

void NeuralNetComputeCostLoss(TPNeuralNet PNeuralNet, float32_t *CostLoss)
{
	TPSoftmaxLayer PSoftmaxLayer = ((TPSoftmaxLayer)PNeuralNet->layers[PNeuralNet->depth - 1]);
	PSoftmaxLayer->expected_value = PNeuralNet->trainning.labelIndex;
	*CostLoss = PSoftmaxLayer->computeLoss(PSoftmaxLayer);
}

void NeuralNetGetMaxPrediction(TPNeuralNet PNeuralNet, TPPrediction PPrediction)
{
	float32_t maxv = -1;
	uint16_t maxi = -1;
	// TPrediction *pPrediction = malloc(sizeof(TPrediction));
	TPSoftmaxLayer PSoftmaxLayer = ((TPSoftmaxLayer)PNeuralNet->layers[PNeuralNet->depth - 1]);
	for (uint16_t i = 0; i < PSoftmaxLayer->layer.out_v->weight->length; i++)
	{
		if (PSoftmaxLayer->layer.out_v->weight->buffer[i] > maxv)
		{
			maxv = PSoftmaxLayer->layer.out_v->weight->buffer[i];
			maxi = i;
		}
	}
	PPrediction->labelIndex = maxi;
	PPrediction->likeliHood = maxv;
}

void NeuralNetUpdatePrediction(TPNeuralNet PNeuralNet)
{
	const uint16_t defaultPredictionCount = 5;
	if (PNeuralNet->trainning.pPredictions == NULL)
	{
		PNeuralNet->trainning.pPredictions = malloc(sizeof(TPPrediction) * defaultPredictionCount);
		if (PNeuralNet->trainning.pPredictions == NULL)
			return;

		for (uint16_t i = 0; i < PNeuralNet->trainning.predictionCount; i++)
		{
			PNeuralNet->trainning.pPredictions[i] = NULL;
		}
	}

	if (PNeuralNet->trainning.predictionCount > 0)
	{
		for (uint16_t i = 0; i < PNeuralNet->trainning.predictionCount; i++)
		{
			free(PNeuralNet->trainning.pPredictions[i]);
		}
		PNeuralNet->trainning.predictionCount = 0;
	}

	if (PNeuralNet->trainning.pPredictions == NULL)
		return;
	TPSoftmaxLayer PSoftmaxLayer = ((TPSoftmaxLayer)PNeuralNet->layers[PNeuralNet->depth - 1]);
	for (uint32_t i = 0; i < PSoftmaxLayer->layer.out_v->weight->length; i++)
	{
		TPrediction *pPrediction = NULL;

		if (PNeuralNet->trainning.pPredictions[PNeuralNet->trainning.predictionCount] == NULL)
			pPrediction = malloc(sizeof(TPrediction));
		else
			pPrediction = PNeuralNet->trainning.pPredictions[PNeuralNet->trainning.predictionCount];

		if (pPrediction == NULL)
			break;
		pPrediction->labelIndex = i;
		pPrediction->likeliHood = PSoftmaxLayer->layer.out_v->weight->buffer[i];

		if (PNeuralNet->trainning.predictionCount < defaultPredictionCount)
		{
			PNeuralNet->trainning.pPredictions[PNeuralNet->trainning.predictionCount] = pPrediction;
			PNeuralNet->trainning.predictionCount++;
		}
		else
		{
			for (uint16_t i = 0; i < defaultPredictionCount; i++)
			{
				if (pPrediction->likeliHood > PNeuralNet->trainning.pPredictions[i]->likeliHood)
				{
					PNeuralNet->trainning.pPredictions[i] = pPrediction;
					break;
				}
			}
		}
	}
}

void NeuralNetPrintWeights(TPNeuralNet PNeuralNet, uint16_t LayerIndex, uint8_t InOut)
{
	// float32_t maxv = 0;
	// uint16_t maxi = 0;
	TPLayer pNetLayer = (TPLayer)(PNeuralNet->layers[LayerIndex]);
	// for (uint16_t i = 0; i < pNetLayer->out_v->weight->length; i++)
	//{
	//	LOGINFOR("LayerType=%s PNeuralNet->depth=%d weight=%f", CNNTypeName[pNetLayer->LayerType], PNeuralNet->depth - 1, pNetLayer->out_v->weight->buffer[i]);
	// }
	if (InOut == 0)
	{
		LOGINFOR("layers[%d] out_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
		if (pNetLayer->out_v != NULL)
			pNetLayer->out_v->print(pNetLayer->out_v, PRINTFLAG_WEIGHT);
		else
			LOGINFOR("pNetLayer->out_v=NULL");
	}
	else if (InOut == 1)
	{
		LOGINFOR("layers[%d] in_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth);
		if (pNetLayer->in_v != NULL)
			pNetLayer->in_v->print(pNetLayer->in_v, PRINTFLAG_WEIGHT);
		else
			LOGINFOR("pNetLayer->in_v=NULL");
	}
	else if (pNetLayer->LayerType == Layer_Type_Convolution)
	{
		for (uint16_t i = 0; i < ((TPConvLayer)pNetLayer)->filters->filterNumber; i++)
		{
			LOGINFOR("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d/%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], ((TPConvLayer)pNetLayer)->filters->_w, ((TPConvLayer)pNetLayer)->filters->_h, ((TPConvLayer)pNetLayer)->filters->_depth, i, ((TPConvLayer)pNetLayer)->filters->filterNumber);
			((TPConvLayer)pNetLayer)->filters->volumes[i]->print(((TPConvLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_WEIGHT);
		}
		LOGINFOR("biases:w=%d h=%d depth=%d", ((TPConvLayer)pNetLayer)->biases->_w, ((TPConvLayer)pNetLayer)->biases->_h, ((TPConvLayer)pNetLayer)->biases->_depth);
		((TPConvLayer)pNetLayer)->biases->print(((TPConvLayer)pNetLayer)->biases, PRINTFLAG_WEIGHT);
	}
	else if (pNetLayer->LayerType == Layer_Type_FullyConnection)
	{
		for (uint16_t i = 0; i < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; i++)
		{
			LOGINFOR("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d/%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], ((TPConvLayer)pNetLayer)->filters->_w, ((TPConvLayer)pNetLayer)->filters->_h, ((TPConvLayer)pNetLayer)->filters->_depth, i, ((TPConvLayer)pNetLayer)->filters->filterNumber);
			((TPFullyConnLayer)pNetLayer)->filters->volumes[i]->print(((TPFullyConnLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_WEIGHT);
		}
		LOGINFOR("biases:w=%d h=%d depth=%d", ((TPFullyConnLayer)pNetLayer)->biases->_w, ((TPFullyConnLayer)pNetLayer)->biases->_h, ((TPFullyConnLayer)pNetLayer)->biases->_depth);
		((TPFullyConnLayer)pNetLayer)->biases->print(((TPFullyConnLayer)pNetLayer)->biases, PRINTFLAG_WEIGHT);
	}
	else if (pNetLayer->LayerType == Layer_Type_SoftMax)
	{
		PNeuralNet->printTensor("EXP", ((TPSoftmaxLayer)pNetLayer)->exp);
	}
	else if (pNetLayer->LayerType == Layer_Type_Pool)
	{
		//((TPPoolLayer)pNetLayer)->filter->print(((TPPoolLayer)pNetLayer)->filter, PRINTFLAG_WEIGHT);
	}
}
/// <summary>
/// discard
/// </summary>
/// <param name="PNeuralNet"></param>
/// <param name="LayerIndex"></param>
/// <param name="InOut"></param>
void NeuralNetPrintFilters(TPNeuralNet PNeuralNet, uint16_t LayerIndex, uint8_t InOut)
{
	TPLayer pNetLayer = (PNeuralNet->layers[LayerIndex]);

	if (InOut == 0)
	{
		LOGINFOR("layers[%d] out_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
		pNetLayer->out_v->print(pNetLayer->out_v, PRINTFLAG_WEIGHT);
	}
	else if (InOut == 1)
	{
		LOGINFOR("layers[%d] in_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
		pNetLayer->in_v->print(pNetLayer->in_v, PRINTFLAG_WEIGHT);
	}
	else if (pNetLayer->LayerType == Layer_Type_Convolution)
	{
		LOGINFOR("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber);
		for (uint16_t i = 0; i < ((TPConvLayer)pNetLayer)->filters->filterNumber; i++)
		{
			((TPConvLayer)pNetLayer)->filters->volumes[i]->print(((TPConvLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_WEIGHT);
		}
		((TPConvLayer)pNetLayer)->biases->print(((TPConvLayer)pNetLayer)->biases, PRINTFLAG_WEIGHT);
	}
	else if (pNetLayer->LayerType == Layer_Type_FullyConnection)
	{
		LOGINFOR("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber);
		for (uint16_t i = 0; i < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; i++)
		{
			((TPFullyConnLayer)pNetLayer)->filters->volumes[i]->print(((TPFullyConnLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_WEIGHT);
		}
		((TPFullyConnLayer)pNetLayer)->biases->print(((TPFullyConnLayer)pNetLayer)->biases, PRINTFLAG_WEIGHT);
	}
}

void NeuralNetPrintGradients(TPNeuralNet PNeuralNet, uint16_t LayerIndex, uint8_t InOut)
{
	// float32_t maxv = 0;
	// uint16_t maxi = 0;
	TPLayer pNetLayer = ((TPLayer)PNeuralNet->layers[LayerIndex]);
	// for (uint16_t i = 0; i < pNetLayer->out_v->weight->length; i++)
	//{
	//	//LOGINFOR("LayerType=%s PNeuralNet->depth=%d weight_grad=%f", CNNTypeName[pNetLayer->LayerType], PNeuralNet->depth - 1, pNetLayer->out_v->weight_grad->buffer[i]);
	// }
	if (InOut == 0)
	{
		LOGINFOR("layers[%d] out_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
		pNetLayer->out_v->print(pNetLayer->out_v, PRINTFLAG_GRADS);
	}
	else if (InOut == 1)
	{
		LOGINFOR("layers[%d] in_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
		if (pNetLayer->in_v != NULL)
			pNetLayer->in_v->print(pNetLayer->in_v, PRINTFLAG_GRADS);
	}
	else if (pNetLayer->LayerType == Layer_Type_Convolution)
	{
		LOGINFOR("layers[%d] w=%d h=%d depth=%d filterNumber=%d %s", LayerIndex, pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber, CNNTypeName[pNetLayer->LayerType]);
		for (uint16_t i = 0; i < ((TPConvLayer)pNetLayer)->filters->filterNumber; i++)
		{
			((TPConvLayer)pNetLayer)->filters->volumes[i]->print(((TPConvLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_GRADS);
		}
	}
	else if (pNetLayer->LayerType == Layer_Type_FullyConnection)
	{
		LOGINFOR("layers[%d] w=%d h=%d depth=%d filterNumber=%d %s", LayerIndex, pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber, CNNTypeName[pNetLayer->LayerType]);
		for (uint16_t i = 0; i < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; i++)
		{
			((TPFullyConnLayer)pNetLayer)->filters->volumes[i]->print(((TPFullyConnLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_GRADS);
		}
	}
}

void NeuralNetPrint(char *Name, TPTensor PTensor)
{
	for (uint16_t i = 0; i < PTensor->length; i++)
	{
		if (i % 16 == 0)
			LOG("\n");
		LOG("%s[%d]=" PRINTFLAG_FORMAT, Name, i, PTensor->buffer[i]);
	}
	LOG("\n");
}

void NeuralNetPrintLayersInfor(TPNeuralNet PNeuralNet)
{
	TPLayer pNetLayer;
	for (uint16_t out_d = 0; out_d < PNeuralNet->depth; out_d++)
	{
		pNetLayer = PNeuralNet->layers[out_d];
		if (pNetLayer->LayerType == Layer_Type_Convolution)
		{
			LOG("[NeuralNetLayerInfor[%02d,%02d]]:in_w=%2d in_h=%2d in_depth=%2d out_w=%2d out_h=%2d out_depth=%2d %-15s fileterNumber=%d size=%dx%dx%d\n", out_d, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
				pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, PNeuralNet->getName(pNetLayer->LayerType), ((TPConvLayer)pNetLayer)->filters->filterNumber,
				((TPConvLayer)pNetLayer)->filters->_w, ((TPConvLayer)pNetLayer)->filters->_h, ((TPConvLayer)pNetLayer)->filters->_depth);
		}
		else if (pNetLayer->LayerType == Layer_Type_FullyConnection)
		{
			LOG("[NeuralNetLayerInfor[%02d,%02d]]:in_w=%2d in_h=%2d in_depth=%2d out_w=%2d out_h=%2d out_depth=%2d %-15s fileterNumber=%d size=%dx%dx%d\n", out_d, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
				pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, PNeuralNet->getName(pNetLayer->LayerType), ((TPConvLayer)pNetLayer)->filters->filterNumber,
				((TPFullyConnLayer)pNetLayer)->filters->_w, ((TPFullyConnLayer)pNetLayer)->filters->_h, ((TPFullyConnLayer)pNetLayer)->filters->_depth);
		}
		else
		{
			LOG("[NeuralNetLayerInfor[%02d,%02d]]:in_w=%2d in_h=%2d in_depth=%2d out_w=%2d out_h=%2d out_depth=%2d %s\n", out_d, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
				pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, PNeuralNet->getName(pNetLayer->LayerType));
		}
	}
}
void NeuralNetPrintTrainningInfor(TPNeuralNet PNeuralNet)
{
	time_t avg_iterations_time = 0;
#if 1
	LOGINFOR("DatasetTotal    :%09ld  ", PNeuralNet->trainning.datasetTotal);
	LOGINFOR("DatasetIndex    :%09ld  ", PNeuralNet->trainning.trinning_dataset_index);
	LOGINFOR("EpochCount      :%09ld  ", PNeuralNet->trainning.epochCount);
	LOGINFOR("SampleCount     :%09ld  ", PNeuralNet->trainning.sampleCount);
	LOGINFOR("LabelIndex      :%09ld  ", PNeuralNet->trainning.labelIndex);
	LOGINFOR("BatchCount      :%09ld  ", PNeuralNet->trainning.batchCount);
	LOGINFOR("Iterations      :%09ld  ", PNeuralNet->trainning.iterations);

	LOGINFOR("AverageCostLoss :%.6f", PNeuralNet->trainning.sum_cost_loss / PNeuralNet->trainning.sampleCount);
	LOGINFOR("L1_decay_loss   :%.6f", PNeuralNet->trainning.sum_l1_decay_loss / PNeuralNet->trainning.sampleCount);
	LOGINFOR("L2_decay_loss   :%.6f", PNeuralNet->trainning.sum_l2_decay_loss / PNeuralNet->trainning.sampleCount);
	LOGINFOR("TrainingAccuracy:%.6f", PNeuralNet->trainning.trainingAccuracy / PNeuralNet->trainning.sampleCount);
	LOGINFOR("TestingAccuracy :%.6f", PNeuralNet->trainning.testingAccuracy / PNeuralNet->trainning.sampleCount);

	if (PNeuralNet->trainning.iterations > 0)
		avg_iterations_time = PNeuralNet->totalTime / PNeuralNet->trainning.iterations;

	LOGINFOR("TotalElapsedTime:%9lld", PNeuralNet->totalTime);
	LOGINFOR("ForwardTime     :%09lld", PNeuralNet->fwTime);
	LOGINFOR("BackwardTime    :%09lld", PNeuralNet->bwTime);
	LOGINFOR("OptimTime       :%09lld", PNeuralNet->optimTime);
	LOGINFOR("AvgBatchTime    :%09lld", avg_iterations_time);
	LOGINFOR("AvgSampleTime   :%09lld", PNeuralNet->totalTime / PNeuralNet->trainning.sampleCount);
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

void NeuralNetTrain(TPNeuralNet PNeuralNet, TPVolume PVolume)
{
	TPTensor weight, grads;
	TPTensor accum_grads1; // 累计历史梯度
	TPTensor accum_grads2; // for Optm_Adadelta
	TPResponse *pResponseResults = NULL;
	TPResponse pResponse = NULL;

	float32_t l1_decay = 0.00;
	float32_t l2_decay = 0.00;
	float32_t l1_decay_grad = 0.00;
	float32_t l2_decay_grad = 0.00;
	float32_t bias1 = 0.00;
	float32_t bias2 = 0.00;

	float32_t delta_x = 0.00;
	float32_t gradij = 0.00;
	float32_t cost_loss = 0.00;
	float32_t l1_decay_loss = 0.00;
	float32_t l2_decay_loss = 0.00;

	accum_grads1 = NULL;
	accum_grads2 = NULL;
	time_t starTick = 0;
	/////////////////////////////////////////////////////////////////////////////////////
	PNeuralNet->optimTime = 0;
	starTick = GetTimestamp();
	PNeuralNet->forward(PNeuralNet, PVolume);
	PNeuralNet->getCostLoss(PNeuralNet, &cost_loss);
	PNeuralNet->fwTime = GetTimestamp() - starTick;
	starTick = GetTimestamp();
	PNeuralNet->backward(PNeuralNet);
	PNeuralNet->bwTime = GetTimestamp() - starTick;
	PNeuralNet->trainning.batchCount++;
	PNeuralNet->trainning.sum_cost_loss = PNeuralNet->trainning.sum_cost_loss + cost_loss;

	/////////////////////////////////////////////////////////////////////////////////////
	// 小批量阀值batch_size
	if (PNeuralNet->trainningParam.batch_size <= 0)
		PNeuralNet->trainningParam.batch_size = 15;
	if (PNeuralNet->trainning.batchCount % PNeuralNet->trainningParam.batch_size != 0)
		return;
	/////////////////////////////////////////////////////////////////////////////////////
	////小批量梯度下降
	starTick = GetTimestamp();
	PNeuralNet->trainning.iterations++;
	PNeuralNet->getWeightsAndGrads(PNeuralNet);
	if (PNeuralNet->trainning.underflow)
	{
		PNeuralNet->trainning.trainningGoing = false;
		LOGERROR("PNeuralNet->trainning.underflow = true");
		return;
	}
	if (PNeuralNet->trainning.overflow)
	{
		PNeuralNet->trainning.trainningGoing = false;
		LOGERROR("PNeuralNet->trainning.overflow = true");
		return;
	}
	///////////////////////////////////////////////////////////////////
	// LOGERROR("Debug:Stop");
	// PNeuralNet->trainning.trainningGoing = false;
	// return;
	//////////////////////////////////////////////////////////////////
	pResponseResults = PNeuralNet->trainning.pResponseResults;
	if (pResponseResults == NULL)
	{
		LOGERROR("No ResponseResults! pResponseResults=NULL");
		PNeuralNet->trainning.trainningGoing = false;
		return;
	}
	if (PNeuralNet->trainning.responseCount <= 0)
	{
		LOGERROR("No ResponseResults! trainning.responseCount <= 0");
		PNeuralNet->trainning.trainningGoing = false;
		return;
	}
	if (PNeuralNet->trainning.grads_sum1 == NULL || PNeuralNet->trainning.grads_sum_count <= 0)
	{
		PNeuralNet->trainning.grads_sum1 = malloc(sizeof(TPTensor) * PNeuralNet->trainning.responseCount);
		PNeuralNet->trainning.grads_sum2 = malloc(sizeof(TPTensor) * PNeuralNet->trainning.responseCount);
		PNeuralNet->trainning.grads_sum_count = PNeuralNet->trainning.responseCount;
		if (PNeuralNet->trainning.grads_sum1 == NULL || PNeuralNet->trainning.grads_sum2 == NULL)
		{
			LOGERROR("No trainning.grads_sum1 or trainning.grads_sum2 NULL");
			PNeuralNet->trainning.trainningGoing = false;
			return;
		}
		for (uint16_t i = 0; i < PNeuralNet->trainning.responseCount; i++)
		{
			PNeuralNet->trainning.grads_sum1[i] = MakeTensor(pResponseResults[i]->filterWeight->length);
			PNeuralNet->trainning.grads_sum2[i] = MakeTensor(pResponseResults[i]->filterWeight->length);
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////
	/// 计算更新整个网络层的权重参数
	for (uint16_t i = 0; i < PNeuralNet->trainning.responseCount; i++)
	{
		pResponse = pResponseResults[i];
		weight = pResponse->filterWeight;
		grads = pResponse->filterGrads;
		accum_grads1 = PNeuralNet->trainning.grads_sum1[i];
		accum_grads2 = PNeuralNet->trainning.grads_sum2[i];
		l1_decay = PNeuralNet->trainningParam.l1_decay_rate * pResponse->l1_decay_rate;
		l2_decay = PNeuralNet->trainningParam.l2_decay_rate * pResponse->l2_decay_rate;

		if (weight->length <= 0 || grads->length <= 0 || accum_grads1 == NULL || accum_grads2 == NULL)
		{
			LOGERROR("response weight->length=%d grads->length=%d grads_sum_count=%d", weight->length, grads->length, PNeuralNet->trainning.grads_sum_count);
			PNeuralNet->trainning.trainningGoing = false;
			break;
		}

		for (uint16_t j = 0; j < weight->length; j++) // update all weight parameters
		{
			l1_decay_loss = l1_decay_loss + l1_decay * abs(weight->buffer[j]);
			l2_decay_loss = l2_decay_loss + l2_decay * weight->buffer[j] * weight->buffer[j] / 2;
			l1_decay_grad = (weight->buffer[j] > 0) ? l1_decay : -l1_decay;
			l2_decay_grad = l2_decay * weight->buffer[j]; // 函数 y = ax^2 的导数是 dy/dx = 2ax
			gradij = (grads->buffer[j] + l1_decay_grad + l2_decay_grad) / PNeuralNet->trainningParam.batch_size;

			switch (PNeuralNet->trainningParam.optimize_method)
			{
			case Optm_Adam:
				accum_grads1->buffer[j] = accum_grads1->buffer[j] * PNeuralNet->trainningParam.beta1 + (1 - PNeuralNet->trainningParam.beta1) * gradij;
				accum_grads2->buffer[j] = accum_grads1->buffer[j] * PNeuralNet->trainningParam.beta2 + (1 - PNeuralNet->trainningParam.beta2) * gradij * gradij;
				bias1 = accum_grads1->buffer[j] * (1 - pow(PNeuralNet->trainningParam.beta1, PNeuralNet->trainning.batchCount));
				bias2 = accum_grads1->buffer[j] * (1 - pow(PNeuralNet->trainningParam.beta2, PNeuralNet->trainning.batchCount));
				delta_x = -PNeuralNet->trainningParam.learning_rate * bias1 / (sqrt(bias2) + PNeuralNet->trainningParam.eps);
				weight->buffer[j] = weight->buffer[j] + delta_x;
				break;
			case Optm_Adagrad:
				accum_grads1->buffer[j] = accum_grads1->buffer[j] + gradij * gradij;
				delta_x = -PNeuralNet->trainningParam.learning_rate / sqrt(accum_grads1->buffer[j] + PNeuralNet->trainningParam.eps) * gradij;
				weight->buffer[j] = weight->buffer[j] + delta_x;
				break;
			case Optm_Adadelta:
				accum_grads1->buffer[j] = accum_grads1->buffer[j] * PNeuralNet->trainningParam.momentum + (1 - PNeuralNet->trainningParam.momentum) * gradij * gradij;
				delta_x = -sqrt((accum_grads2->buffer[j] + PNeuralNet->trainningParam.eps) / (accum_grads1->buffer[j] + PNeuralNet->trainningParam.eps)) * gradij;
				accum_grads2->buffer[j] = accum_grads2->buffer[j] * PNeuralNet->trainningParam.momentum + (1 - PNeuralNet->trainningParam.momentum) * delta_x * delta_x;
				weight->buffer[j] = weight->buffer[j] + delta_x;
				break;
			default:
				break;
			} // switch
			grads->buffer[j] = 0;
		}
	}
	for (uint16_t i = 0; i < PNeuralNet->trainning.responseCount; i++)
	{
		pResponse = pResponseResults[i];
		free(pResponse);
	}
	free(PNeuralNet->trainning.pResponseResults);
	PNeuralNet->trainning.responseCount = 0;
	PNeuralNet->trainning.pResponseResults = NULL;
	PNeuralNet->optimTime = GetTimestamp() - starTick;

	PNeuralNet->trainning.sum_l1_decay_loss = (PNeuralNet->trainning.sum_l1_decay_loss + l1_decay_loss);
	PNeuralNet->trainning.sum_l2_decay_loss = (PNeuralNet->trainning.sum_l2_decay_loss + l2_decay_loss);
	// PNeuralNet->trainning.cost_loss_sum = PNeuralNet->trainning.cost_loss_sum + cost_loss;
}

void NeuralNetPredict(TPNeuralNet PNeuralNet, TPVolume PVolume)
{
	float32_t cost_loss = 0;
	PNeuralNet->forward(PNeuralNet, PVolume);
	PNeuralNet->getCostLoss(PNeuralNet, &cost_loss);
}
/// @brief ///////////////////////////////////////////////////////////////////////
/// @param PNeuralNet
void NeuralNetSaveWeights(TPNeuralNet PNeuralNet)
{
	FILE *pFile = NULL;
	char *name = NULL;
	if (PNeuralNet == NULL)
		return;
	if (PNeuralNet->name != NULL)
	{
		name = (char *)malloc(strlen(PNeuralNet->name) + strlen(NEURALNET_CNN_WEIGHT_FILE_NAME) + 3);
		sprintf(name, "%s_%02d%s", PNeuralNet->name, PNeuralNet->depth, NEURALNET_CNN_WEIGHT_FILE_NAME);
		pFile = fopen(name, "wb");
	}
	else
	{
		pFile = fopen(NEURALNET_CNN_WEIGHT_FILE_NAME, "wb");
	}
	if (pFile != NULL)
	{
		for (uint16_t layerIndex = 0; layerIndex < PNeuralNet->depth; layerIndex++)
		{
			TPLayer pNetLayer = (PNeuralNet->layers[layerIndex]);
			switch (pNetLayer->LayerType)
			{
			case Layer_Type_Input:
				break;
			case Layer_Type_Convolution:
			{
				for (uint16_t out_d = 0; out_d < ((TPConvLayer)pNetLayer)->filters->filterNumber; out_d++)
				{
					TensorSave(pFile, ((TPConvLayer)pNetLayer)->filters->volumes[out_d]->weight);
				}
				TensorSave(pFile, ((TPConvLayer)pNetLayer)->biases->weight);
				break;
			}
			case Layer_Type_Pool:
				break;
			case Layer_Type_ReLu:
				break;
			case Layer_Type_FullyConnection:
			{
				for (uint16_t out_d = 0; out_d < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; out_d++)
				{
					TensorSave(pFile, ((TPFullyConnLayer)pNetLayer)->filters->volumes[out_d]->weight);
				}
				TensorSave(pFile, ((TPFullyConnLayer)pNetLayer)->biases->weight);
				break;
			}
			case Layer_Type_SoftMax:
				break;
			default:
				break;
			}
		}
		fclose(pFile);
		LOGINFOR("save weights to file %s", name);
	}
	// if (name != NULL)
	//	free(name);
}

void NeuralNetLoadWeights(TPNeuralNet PNeuralNet)
{
	FILE *pFile = NULL;
	char *name = NULL;
	if (PNeuralNet == NULL)
		return;
	if (PNeuralNet->name != NULL)
	{
		name = (char *)malloc(strlen(PNeuralNet->name) + strlen(NEURALNET_CNN_WEIGHT_FILE_NAME) + 3);
		sprintf(name, "%s_%02d%s", PNeuralNet->name, PNeuralNet->depth, NEURALNET_CNN_WEIGHT_FILE_NAME);
		pFile = fopen(name, "rb");
	}
	else
	{
		pFile = fopen(NEURALNET_CNN_WEIGHT_FILE_NAME, "rb");
		name = NEURALNET_CNN_WEIGHT_FILE_NAME;
	}

	if (pFile != NULL)
	{
		for (uint16_t layerIndex = 0; layerIndex < PNeuralNet->depth; layerIndex++)
		{
			TPLayer pNetLayer = (PNeuralNet->layers[layerIndex]);
			switch (pNetLayer->LayerType)
			{
			case Layer_Type_Input:
				break;
			case Layer_Type_Convolution:
			{
				for (uint16_t out_d = 0; out_d < ((TPConvLayer)pNetLayer)->filters->filterNumber; out_d++)
				{
					TensorLoad(pFile, ((TPConvLayer)pNetLayer)->filters->volumes[out_d]->weight);
				}
				TensorLoad(pFile, ((TPConvLayer)pNetLayer)->biases->weight);
				break;
			}
			case Layer_Type_ReLu:
				break;
			case Layer_Type_Pool:
				break;
			case Layer_Type_FullyConnection:
			{
				for (uint16_t out_d = 0; out_d < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; out_d++)
				{
					TensorLoad(pFile, ((TPFullyConnLayer)pNetLayer)->filters->volumes[out_d]->weight);
				}
				TensorLoad(pFile, ((TPFullyConnLayer)pNetLayer)->biases->weight);
				break;
			}
			case Layer_Type_SoftMax:
				break;
			default:
				break;
			}
		}
		LOGINFOR("Loaded weights from file %s", name);
		fclose(pFile);
	}
	else
	{
		LOGINFOR("Loaded weights from file %s failed file not found!", name);
	}
	// if (name != NULL)
	//	free(name);
}

void NeuralNetSaveNet(TPNeuralNet PNeuralNet)
{
	FILE *pFile = NULL;
	char *name = NULL;
	if (PNeuralNet == NULL)
		return;
	if (PNeuralNet->name != NULL)
	{
		name = (char *)malloc(strlen(PNeuralNet->name) + strlen(NEURALNET_CNN_FILE_NAME));
		sprintf(name, "%s%s", PNeuralNet->name, NEURALNET_CNN_FILE_NAME);
		pFile = fopen(name, "wb");
	}
	else
	{
		pFile = fopen(NEURALNET_CNN_FILE_NAME, "wb");
	}
	if (pFile != NULL)
	{
		for (uint16_t layerIndex = 0; layerIndex < PNeuralNet->depth; layerIndex++)
		{
			TPLayer pNetLayer = (PNeuralNet->layers[layerIndex]);
			switch (pNetLayer->LayerType)
			{
			case Layer_Type_Input:
				break;
			case Layer_Type_Convolution:
				break;
			case Layer_Type_ReLu:
				break;
			case Layer_Type_Pool:
				break;
			case Layer_Type_FullyConnection:
				break;
			case Layer_Type_SoftMax:
				break;
			default:
				break;
			}
		}
		fclose(pFile);
		LOGINFOR("save weights to file %s", name);
	}
}

void NeuralNetLoadNet(TPNeuralNet PNeuralNet)
{
}

TPNeuralNet NeuralNetCNNCreate(char *name)
{
	TPNeuralNet PNeuralNet = malloc(sizeof(TNeuralNet));

	if (PNeuralNet == NULL)
	{
		LOGERROR("PNeuralNet==NULL!");
		return NULL;
	}
	PNeuralNet->name = name;
	PNeuralNet->init = NeuralNetInit;
	PNeuralNet->free = NeuralNetFree;
	PNeuralNet->forward = NeuralNetForward;
	PNeuralNet->backward = NeuralNetBackward;
	PNeuralNet->getWeightsAndGrads = NeuralNetGetWeightsAndGrads;
	PNeuralNet->getCostLoss = NeuralNetComputeCostLoss;
	PNeuralNet->getPredictions = NeuralNetUpdatePrediction;
	PNeuralNet->getMaxPrediction = NeuralNetGetMaxPrediction;
	PNeuralNet->train = NeuralNetTrain;
	PNeuralNet->predict = NeuralNetPredict;
	PNeuralNet->saveWeights = NeuralNetSaveWeights;
	PNeuralNet->loadWeights = NeuralNetLoadWeights;
	PNeuralNet->printGradients = NeuralNetPrintGradients;
	PNeuralNet->printWeights = NeuralNetPrintWeights;
	PNeuralNet->printTrainningInfor = NeuralNetPrintTrainningInfor;
	PNeuralNet->printNetLayersInfor = NeuralNetPrintLayersInfor;
	PNeuralNet->printTensor = NeuralNetPrint;
	PNeuralNet->getName = NeuralNetGetLayerName;
}

char *NeuralNetGetLayerName(TLayerType LayerType)
{
	return CNNTypeName[LayerType];
}
int NeuralNetAddLayer(TPNeuralNet PNeuralNet, TLayerOption LayerOption)
{
	TPLayer pNetLayer = NULL;
	if (PNeuralNet == NULL)
		return NEURALNET_ERROR_BASE;
	switch (LayerOption.LayerType)
	{
	case Layer_Type_Input:
		// LayerOption.in_w = LayerOption.in_w;
		// LayerOption.in_h = LayerOption.in_h;
		// LayerOption.in_depth = LayerOption.in_depth;
		if (PNeuralNet->depth > 0)
		{
			LOGINFOR("input layer need to add first");
			return LayerOption.LayerType + NEURALNET_ERROR_BASE;
		}
		PNeuralNet->init(PNeuralNet, &LayerOption);
		break;
	case Layer_Type_Convolution:
	{
		if (PNeuralNet->depth <= 0)
			return LayerOption.LayerType + NEURALNET_ERROR_BASE;
		pNetLayer = PNeuralNet->layers[PNeuralNet->depth - 1];
		LayerOption.LayerType = Layer_Type_Convolution;
		LayerOption.in_w = pNetLayer->out_w;
		LayerOption.in_h = pNetLayer->out_h;
		LayerOption.in_depth = pNetLayer->out_depth;
		LayerOption.filter_w = LayerOption.filter_w;
		LayerOption.filter_h = LayerOption.filter_h;
		LayerOption.filter_depth = LayerOption.in_depth;
		LayerOption.filter_number = LayerOption.filter_number;
		LayerOption.stride = LayerOption.stride;
		LayerOption.padding = LayerOption.padding;
		LayerOption.bias = LayerOption.bias;
		LayerOption.l1_decay_rate = LayerOption.l1_decay_rate;
		LayerOption.l2_decay_rate = LayerOption.l2_decay_rate;
		PNeuralNet->init(PNeuralNet, &LayerOption);
		return LayerOption.LayerType;
		break;
	}
	case Layer_Type_ReLu:
		if (PNeuralNet->depth <= 0)
			return LayerOption.LayerType + NEURALNET_ERROR_BASE;
		pNetLayer = PNeuralNet->layers[PNeuralNet->depth - 1];
		LayerOption.in_w = pNetLayer->out_w;
		LayerOption.in_h = pNetLayer->out_h;
		LayerOption.in_depth = pNetLayer->out_depth;
		PNeuralNet->init(PNeuralNet, &LayerOption);
		return LayerOption.LayerType;
		break;
	case Layer_Type_Pool:
		if (PNeuralNet->depth <= 0)
			return LayerOption.LayerType + NEURALNET_ERROR_BASE;
		pNetLayer = PNeuralNet->layers[PNeuralNet->depth - 1];
		LayerOption.LayerType = Layer_Type_Pool;
		LayerOption.in_w = pNetLayer->out_w;
		LayerOption.in_h = pNetLayer->out_h;
		LayerOption.in_depth = pNetLayer->out_depth;
		LayerOption.filter_w = LayerOption.filter_w;
		LayerOption.filter_h = LayerOption.filter_h;
		LayerOption.filter_depth = LayerOption.in_depth;
		PNeuralNet->init(PNeuralNet, &LayerOption);
		return LayerOption.LayerType;
		break;
	case Layer_Type_FullyConnection:
	{
		if (PNeuralNet->depth <= 0)
			return LayerOption.LayerType + NEURALNET_ERROR_BASE;
		pNetLayer = PNeuralNet->layers[PNeuralNet->depth - 1];
		LayerOption.in_w = pNetLayer->out_w;
		LayerOption.in_h = pNetLayer->out_h;
		LayerOption.in_depth = pNetLayer->out_depth;

		LayerOption.filter_w = LayerOption.filter_w;
		LayerOption.filter_h = LayerOption.filter_h;
		LayerOption.filter_depth = LayerOption.in_w * LayerOption.in_h * LayerOption.in_depth;
		LayerOption.filter_number = LayerOption.filter_number;
		LayerOption.out_depth = LayerOption.filter_number;
		LayerOption.out_h = 1;
		LayerOption.out_w = 1;
		LayerOption.bias = LayerOption.bias;
		LayerOption.l1_decay_rate = LayerOption.l1_decay_rate;
		LayerOption.l2_decay_rate = LayerOption.l2_decay_rate;
		PNeuralNet->init(PNeuralNet, &LayerOption);
		return LayerOption.LayerType;
		break;
	}
	case Layer_Type_SoftMax:
		if (PNeuralNet->depth <= 0)
			return LayerOption.LayerType + NEURALNET_ERROR_BASE;
		pNetLayer = PNeuralNet->layers[PNeuralNet->depth - 1];
		LayerOption.in_w = pNetLayer->out_w;
		LayerOption.in_h = pNetLayer->out_h;
		LayerOption.in_depth = pNetLayer->out_depth;

		LayerOption.out_h = 1;
		LayerOption.out_w = 1;
		LayerOption.out_depth = LayerOption.in_depth * LayerOption.in_w * LayerOption.in_h;
		return LayerOption.LayerType;
		break;
	default:
		break;
	}
	return NEURALNET_ERROR_BASE;
}
