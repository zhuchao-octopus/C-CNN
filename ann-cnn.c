/*****************************************************************************************************
 * ann-cnn.c
 *
 *  Created on: Mar 29, 2023
 *      Author: M
 ******************************************************************************************************
 ******************************************************************************************************/
#ifdef PLATFORM_STM32
#include "usart.h"
#include "octopus.h"
#endif

#include "ann-cnn.h"

char *CNNTypeName[] = {"Input", "Convolution", "ReLu", "Pool", "FullyConnection", "SoftMax", "None"};

void testDSPFloatProcess(void)
{
	float32_t fv;
	// fv = arm_sin_f32(3.1415926 / 7);
	// printf("sin=%.5f\r\n", fv);
}

time_t GetTimestamp(void)
{
	return time(NULL);
}

double GaussRandom2(void)
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
float32_t GaussRandom1(void)
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
float32_t GaussRandom(void)
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
	if ((r = 0) || (r > 1))
	{
		return GaussRandom();
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

// 生成高斯分布随机数序列期望为μ、方差为σ2=Variance
// 若随机变量X服从一个数学期望为μ、方差为σ2的正态分布记为N(μ，σ2)
// 其概率密度函数为正态分布的期望值μ决定了其位置，其标准差σ决定了分布的幅度
// 当μ = 0,σ = 1时的正态分布是标准正态分布。
float32_t NeuralNet_GetGaussRandom(double mul, float32_t Variance)
{
	return mul + GaussRandom2() * Variance;
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

void InitTensor(TPTensor PTensor, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias)
{
	uint32_t n = W * H * Depth;
	PTensor = MakeTensor(n);
	FillZero(PTensor);
}

void FillZero(TPTensor PTensor)
{
	if (PTensor->length > 0)
		memset(PTensor->buffer, 0, PTensor->length * sizeof(float32_t));
}

void FillWith(TPTensor PTensor, float32_t Bias)
{
	for (int i = 0; i < PTensor->length; i++)
	{
		PTensor->buffer[i] = Bias;
	}
}

void FillGauss(TPTensor PTensor)
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

void FreeTensor(TPTensor PTensor)
{
	if (PTensor != NULL)
	{
		free(PTensor->buffer);
		PTensor->length = 0;
		free(PTensor);
		PTensor = NULL;
	}
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
		tPVolume->init = InitVolume;
		tPVolume->free = FreeTensor;
		tPVolume->fillZero = FillZero;
		tPVolume->fillGauss = FillGauss;
		tPVolume->setValue = VolumeSetValue;
		tPVolume->getValue = VolumeGetValue;
		tPVolume->getGradValue = VolumeGetValue;
		tPVolume->setGradValue = VolumeSetGradValue;
		tPVolume->addGradValue = VolumeAddGradValue;
		tPVolume->print = VolumePrint;
	}
	return tPVolume;
}

void InitVolume(TPVolume PVolume, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias)
{
	PVolume->_w = W;
	PVolume->_h = H;
	PVolume->_depth = Depth;
	int n = PVolume->_w * PVolume->_h * PVolume->_depth;
	PVolume->weight = MakeTensor(n);
	PVolume->weight_d = MakeTensor(n);
	// FillZero(PVolume->weight);
	// FillZero(PVolume->weight_grad);
	FillWith(PVolume->weight, Bias);
	FillWith(PVolume->weight_d, Bias);
}

void FreeVolume(TPVolume PVolume)
{
	if (PVolume != NULL)
	{
		FreeTensor(PVolume->weight);
		FreeTensor(PVolume->weight_d);
		free(PVolume);
	}
	PVolume = NULL;
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
		for (uint16_t i = 0; i < tPFilters->filterNumber; i++)
		{
			tPFilters->volumes[i] = MakeVolume(tPFilters->_w, tPFilters->_w, tPFilters->_depth);
		}
		tPFilters->init = InitVolume;
		tPFilters->free = FreeFilterVolumes;
	}
	return tPFilters;
}

bool ResizeFilters(TPFilters PFilters, uint16_t W, uint16_t H, uint16_t Depth, uint16_t FilterNumber)
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
		PFilters->init = InitVolume;
		PFilters->free = FreeFilterVolumes;
	}
	return true;
}

void FreeFilters(TPFilters PFilters)
{
	if (PFilters != NULL)
	{
		for (uint16_t d = 0; d < PFilters->filterNumber; d++)
		{
			FreeVolume(PFilters->volumes[d]);
		}
		free(PFilters);
		PFilters = NULL;
	}
}

void FreeFilterVolumes(TPFilters PFilters)
{
	if (PFilters != NULL)
	{
		for (uint16_t d = 0; d < PFilters->filterNumber; d++)
		{
			FreeVolume(PFilters->volumes[d]);
		}
		PFilters->volumes = NULL;
	}
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
/// @brief ////////////////////////////////////////////////////////////
/// @param PVolume
/// @param wg 0:weight,1:weight_grad
void VolumePrint(TPVolume PVolume, uint8_t wg)
{
	if (PVolume->_h == 1 && PVolume->_w == 1)
	{
		if (wg == 0)
			LOGINFO("weight:PVolume->_depth=%d/%d", PVolume->_depth, PVolume->_depth);
		else
			LOGINFO("grads:PVolume->_depth=%d/%d", PVolume->_depth, PVolume->_depth);
	}
	float32_t f32 = 0.00;
	for (uint16_t d = 0; d < PVolume->_depth; d++)
	{
		if (PVolume->_h == 1 && PVolume->_w == 1)
		{
			if (wg == 0)
			{
				f32 = PVolume->getValue(PVolume, 0, 0, d);
				(d == 0) ? LOG("%f", f32) : LOG(",%f", f32);
			}
			else
			{
				f32 = PVolume->getGradValue(PVolume, 0, 0, d);
				(d == 0) ? LOG("%f", f32) : LOG(",%f", f32);
			}
		}
		else
		{
			if (wg == 0)
				LOGINFO("weight:PVolume->_depth=%d/%d %dx%d", d, PVolume->_depth, PVolume->_w, PVolume->_h);
			else
				LOGINFO("grads:PVolume->_depth=%d/%d %dx%d", d, PVolume->_depth, PVolume->_w, PVolume->_h);
			for (uint16_t y = 0; y < PVolume->_h; y++)
			{
				for (uint16_t x = 0; x < PVolume->_w; x++)
				{
					if (wg == 0)
					{
						f32 = PVolume->getValue(PVolume, x, y, d);
						(x == 0) ? LOG("%f", f32) : LOG(",%f", f32);
					}
					else
					{
						f32 = PVolume->getGradValue(PVolume, x, y, d);
						(x == 0) ? LOG("%f", f32) : LOG(",%f", f32);
					}
				}
				LOG("\n");
			}
		}
	}
	if (PVolume->_h == 1 && PVolume->_w == 1)
		LOG("\n");
}
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
	FreeVolume(PInputLayer->layer.in_v);
	FreeVolume(PInputLayer->layer.out_v);
}
/////////////////////////////////////////////////////////////////////////////
/// @brief //////////////////////////////////////////////////////////////////
/// @param PConvLayer
/// @param PLayerOption
void ConvolutionLayerInit(TPConvLayer PConvLayer, TPLayerOption PLayerOption)
{
	PConvLayer->layer.LayerType = PLayerOption->LayerType;
	PConvLayer->l1_decay_mul = PLayerOption->l1_decay_mul;
	PConvLayer->l2_decay_mul = PLayerOption->l2_decay_mul;
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
	PConvLayer->layer.out_v->init(PConvLayer->layer.out_v, PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth, PConvLayer->bias);

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

	if (PConvLayer->layer.out_w != out_w || PConvLayer->layer.out_h != out_h || PConvLayer->filters->_depth != filter_depth)
	{
		LOGINFO("ConvLayer resize out_v from %d x %d to %d x %d", PConvLayer->layer.out_w, PConvLayer->layer.out_h, out_w, out_h);
		FreeFilters(PConvLayer->filters);
		bool ret = ResizeFilters(PConvLayer->filters, PConvLayer->filters->_w, PConvLayer->filters->_h, filter_depth, PConvLayer->filters->filterNumber);
		if (!ret)
			LOGERROR("Resize Filters failed! W=%d H=%d Depth=%d FilterNumber=%d", PConvLayer->filters->_w, PConvLayer->filters->_h, filter_depth, PConvLayer->filters->filterNumber);

		PConvLayer->layer.out_w = out_w;
		PConvLayer->layer.out_h = out_h;
		if (PConvLayer->layer.out_v != NULL)
		{
			FreeVolume(PConvLayer->layer.out_v);
		}
		PConvLayer->layer.out_v = MakeVolume(PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth);
		PConvLayer->layer.out_v->init(PConvLayer->layer.out_v, PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth, PConvLayer->bias);
	}
}
/// @brief //////////////////////////////////////////////////////////////////////////
/// @param PConvLayer
void ConvolutionLayerForward(TPConvLayer PConvLayer)
{
	float32_t sum = 0.00;
	uint16_t x = -PConvLayer->padding;
	uint16_t y = -PConvLayer->padding;
	uint16_t ix11, ix22, ox, oy;
	TPVolume inVolu = PConvLayer->layer.in_v;
	TPVolume outVolu = PConvLayer->layer.out_v;

	convolutionLayerOutResize(PConvLayer);
	// outVolu->fillZero(outVolu->weight);
	// outVolu->fillZero(outVolu->weight_d);
	//  for (uint16_t out_d = 0; out_d < PConvLayer->filters->_depth; out_d++)
	for (uint16_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
	{
		TPVolume filter = PConvLayer->filters->volumes[out_d];
		x = -PConvLayer->padding;
		y = -PConvLayer->padding;
		for (uint16_t out_y = 0; out_y < PConvLayer->layer.out_h; out_y++)
		{
			x = -PConvLayer->padding;
			for (uint16_t out_x = 0; out_x < PConvLayer->layer.out_w; out_x++)
			{
				sum = 0.00;
				for (uint16_t filter_y = 0; filter_y < PConvLayer->filters->_h; filter_y++)
				{
					oy = filter_y + y;
					if (oy < 0 || oy >= inVolu->_h)
						continue;

					for (uint16_t filter_x = 0; filter_x < PConvLayer->filters->_w; filter_x++)
					{
						ox = filter_x + x;
						if (ox < 0 || ox >= inVolu->_w)
							continue;

						ix11 = (filter->_w * filter_y + filter_x) * filter->_depth;
						ix22 = (inVolu->_w * (oy) + ox) * inVolu->_depth;
						for (uint16_t filter_d = 0; filter_d < PConvLayer->filters->_depth; filter_d++)
						{
							sum = sum + filter->weight->buffer[ix11 + filter_d] * inVolu->weight->buffer[ix22 + filter_d];
						}
					}
				}
				sum = sum + PConvLayer->biases->weight->buffer[out_d];
				uint16_t ix33 = (outVolu->_w * out_y + out_x) * outVolu->_depth + out_d;
				outVolu->weight->buffer[ix33] = sum;
				x = x + PConvLayer->stride;
			}
			y = y + PConvLayer->stride;
		}
	}
}

void ConvolutionLayerBackward(TPConvLayer PConvLayer)
{
	// float32_t sum = 0.00;
	float32_t out_grad = 0.00;
	uint16_t x = -PConvLayer->padding;
	uint16_t y = -PConvLayer->padding;
	uint16_t ix11, ix22, ox, oy;
	TPVolume inVolu = PConvLayer->layer.in_v;
	TPVolume outVolu = PConvLayer->layer.out_v;
	inVolu->fillZero(inVolu->weight_d);

	for (uint16_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
	{
		TPVolume filter = PConvLayer->filters->volumes[out_d];
		x = -PConvLayer->padding;
		y = -PConvLayer->padding;
		for (uint16_t out_y = 0; out_y < PConvLayer->layer.out_h; out_y++)
		{
			x = -PConvLayer->padding;
			for (uint16_t out_x = 0; out_x < PConvLayer->layer.out_w; out_x++)
			{
				out_grad = outVolu->getGradValue(outVolu, out_x, out_y, out_d);
				for (uint16_t filter_y = 0; filter_y < PConvLayer->filters->_h; filter_y++)
				{
					oy = filter_y + y;
					if (oy < 0 || oy >= inVolu->_h)
						continue;

					for (uint16_t filter_x = 0; filter_x < PConvLayer->filters->_w; filter_x++)
					{
						ox = filter_x + x;
						if (ox < 0 || ox >= inVolu->_w)
							continue;

						ix11 = (filter->_w * filter_y + filter_x) * filter->_depth;
						ix22 = (inVolu->_w * (oy) + ox) * inVolu->_depth;
						for (uint16_t filter_d = 0; filter_d < PConvLayer->filters->_depth; filter_d++)
						{
							filter->weight_d->buffer[ix11 + filter_d] = filter->weight_d->buffer[ix11 + filter_d] + inVolu->weight->buffer[ix22 + filter_d] * out_grad;
							inVolu->weight_d->buffer[ix22 + filter_d] = inVolu->weight_d->buffer[ix22 + filter_d] + filter->weight->buffer[ix11 + filter_d] * out_grad;
						}
					}
				}

				PConvLayer->biases->weight_d->buffer[out_d] = PConvLayer->biases->weight_d->buffer[out_d] + out_grad;
				x = x + PConvLayer->stride;
			}
			y = y + PConvLayer->stride;
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
	for (uint16_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
	{
		TPResponse PResponse = malloc(sizeof(TResponse));
		if (PResponse != NULL)
		{
			PResponse->filter = PConvLayer->filters->volumes[out_d]->weight;
			PResponse->filterGrads = PConvLayer->filters->volumes[out_d]->weight_d;
			PResponse->l1_decay_mul = PConvLayer->l1_decay_mul;
			PResponse->l2_decay_mul = PConvLayer->l2_decay_mul;
			PResponse->fillZero = FillZero;
			PResponse->free = FreeTensor;
			tPResponses[out_d] = PResponse;
		}
	}

	TPResponse PResponse = malloc(sizeof(TResponse));
	if (PResponse != NULL)
	{
		PResponse->filter = PConvLayer->biases->weight;
		PResponse->filterGrads = PConvLayer->biases->weight_d;
		PResponse->l1_decay_mul = PConvLayer->l1_decay_mul;
		PResponse->l2_decay_mul = PConvLayer->l2_decay_mul;
		PResponse->fillZero = FillZero;
		PResponse->free = FreeTensor;
		tPResponses[PConvLayer->layer.out_depth] = PResponse;
	}
	return tPResponses;
}

float32_t ConvolutionLayerBackwardLoss(TPConvLayer PConvLayer, int Y)
{
	return 0.00;
}

float32_t ConvolutionLayerBackwardOutput(TPConvLayer PConvLayer, TTensor Tensor)
{
	return 0.00;
}

void ConvolutionLayerFree(TPConvLayer PConvLayer)
{
	FreeVolume(PConvLayer->layer.in_v);
	FreeVolume(PConvLayer->layer.out_v);
	FreeVolume(PConvLayer->biases);
	FreeFilters(PConvLayer->filters);

	// for (uint16_t i = 0; i < PConvLayer->filters->_depth; i++)
	//{
	//	PConvLayer->filters->free(PConvLayer->filters->volumes[i]);
	// }
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
	// PReluLayer->l1_decay_mul = LayerOption.l1_decay_mul;
	// PReluLayer->l2_decay_mul = LayerOption.l2_decay_mul;
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
		LOGINFO("ReluLayer resize out_v from %d x %d x %d to %d x %d x %d", PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth, PReluLayer->layer.in_w, PReluLayer->layer.in_h, PReluLayer->layer.in_depth);
		PReluLayer->layer.out_w = PReluLayer->layer.in_w;
		PReluLayer->layer.out_h = PReluLayer->layer.in_h;
		PReluLayer->layer.out_depth = PReluLayer->layer.in_depth;
		if (PReluLayer->layer.out_v != NULL)
		{
			FreeVolume(PReluLayer->layer.out_v);
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
	for (uint16_t out_l = 0; out_l < PReluLayer->layer.out_v->weight->length; out_l++)
	{
		if (PReluLayer->layer.in_v->weight->buffer[out_l] < 0)
			PReluLayer->layer.out_v->weight->buffer[out_l] = 0;
		else
			PReluLayer->layer.out_v->weight->buffer[out_l] = PReluLayer->layer.in_v->weight->buffer[out_l];
	}
}

void ReluLayerBackward(TPReluLayer PReluLayer)
{
	for (uint16_t out_l = 0; out_l < PReluLayer->layer.in_v->weight->length; out_l++)
	{
		if (PReluLayer->layer.out_v->weight->buffer[out_l] < 0)
			PReluLayer->layer.in_v->weight_d->buffer[out_l] = 0;
		else
			PReluLayer->layer.in_v->weight->buffer[out_l] = PReluLayer->layer.out_v->weight_d->buffer[out_l];
	}
}

float32_t ReluLayerBackwardLoss(TPReluLayer PReluLayer, int Y)
{
	return 0.00;
}

float32_t ReluLayerBackwardOutput(TPReluLayer PReluLayer, TTensor Tensor)
{
	return 0.00;
}

void ReluLayerFree(TPReluLayer PReluLayer)
{
	free(PReluLayer);
}
////////////////////////////////////////////////////////////////////////////
/// @brief /////////////////////////////////////////////////////////////////
/// @param PPoolLayer
/// @param PLayerOption /
void PoolLayerInit(TPPoolLayer PPoolLayer, TPLayerOption PLayerOption)
{
	PPoolLayer->layer.LayerType = PLayerOption->LayerType;
	// PPoolLayer->l1_decay_mul = PLayerOption->l1_decay_mul;
	// PPoolLayer->l2_decay_mul = PLayerOption->l2_decay_mul;
	PPoolLayer->stride = PLayerOption->stride;
	PPoolLayer->padding = PLayerOption->padding;
	// PPoolLayer->bias = PLayerOption->bias;

	PPoolLayer->filter._w = PLayerOption->filter_w;
	PPoolLayer->filter._h = PLayerOption->filter_h;
	PPoolLayer->filter._depth = PLayerOption->filter_depth;

	PPoolLayer->layer.in_w = PLayerOption->in_w;
	PPoolLayer->layer.in_h = PLayerOption->in_h;
	PPoolLayer->layer.in_depth = PLayerOption->in_depth;
	PPoolLayer->layer.in_v = NULL;

	PPoolLayer->layer.out_w = floor((PPoolLayer->layer.in_w + PPoolLayer->padding * 2 - PPoolLayer->filter._w) / PPoolLayer->stride + 1);
	PPoolLayer->layer.out_h = floor((PPoolLayer->layer.in_h + PPoolLayer->padding * 2 - PPoolLayer->filter._h) / PPoolLayer->stride + 1);
	PPoolLayer->layer.out_depth = PPoolLayer->layer.in_depth;

	PPoolLayer->layer.out_v = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
	PPoolLayer->layer.out_v->init(PPoolLayer->layer.out_v, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);

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
	uint16_t out_w = floor((PPoolLayer->layer.in_w + PPoolLayer->padding * 2 - PPoolLayer->filter._w) / PPoolLayer->stride + 1);
	uint16_t out_h = floor((PPoolLayer->layer.in_h + PPoolLayer->padding * 2 - PPoolLayer->filter._h) / PPoolLayer->stride + 1);
	if (PPoolLayer->layer.out_w != out_w || PPoolLayer->layer.out_h != out_h || PPoolLayer->layer.out_depth != PPoolLayer->layer.in_depth)
	{
		LOGINFO("PoolLayer resize out_v from %d x %d x %d to %d x %d x %d", PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, out_w, out_h, PPoolLayer->layer.in_depth);
		PPoolLayer->layer.out_w = out_w;
		PPoolLayer->layer.out_h = out_h;
		PPoolLayer->layer.out_depth = PPoolLayer->layer.in_depth;
		if (PPoolLayer->layer.out_v != NULL)
		{
			FreeVolume(PPoolLayer->layer.out_v);
			FreeVolume(PPoolLayer->switchxy);
			// FreeVolume(PPoolLayer->switchy);
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
	float32_t max_value = MINI_NEGATIVE_NUMBER;
	float32_t value = 0;
	uint16_t x = -PPoolLayer->padding;
	uint16_t y = -PPoolLayer->padding;
	uint16_t ox, oy, inx, iny;
	TPVolume inVolu = PPoolLayer->layer.in_v;
	TPVolume outVolu = PPoolLayer->layer.out_v;
	poolLayerOutResize(PPoolLayer);
	outVolu->fillZero(outVolu->weight);

	for (uint16_t out_d = 0; out_d < PPoolLayer->layer.out_depth; out_d++)
	{
		x = -PPoolLayer->padding;
		y = -PPoolLayer->padding;
		for (uint16_t out_y = 0; out_y < PPoolLayer->layer.out_h; out_y++)
		{
			x = -PPoolLayer->padding;

			for (uint16_t out_x = 0; out_x < PPoolLayer->layer.out_w; out_x++)
			{
				max_value = MINI_NEGATIVE_NUMBER;
				inx = -1;
				iny = -1;
				for (uint16_t filter_y = 0; filter_y < PPoolLayer->filter._h; filter_y++)
				{
					oy = filter_y + y;
					if (oy < 0 && oy >= inVolu->_h)
						continue;
					for (uint16_t filter_x = 0; filter_x < PPoolLayer->filter._w; filter_x++)
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

void PoolLayerBackward(TPPoolLayer PPoolLayer)
{
	float32_t grad_value = 0.00;
	TPVolume inVolu = PPoolLayer->layer.in_v;
	TPVolume outVolu = PPoolLayer->layer.out_v;
	inVolu->fillZero(inVolu->weight_d);
	uint16_t x, y;
	for (uint16_t out_d = 0; out_d < PPoolLayer->layer.out_depth; out_d++)
	{
		x = -PPoolLayer->padding;
		y = -PPoolLayer->padding;
		for (uint16_t out_y = 0; out_y < PPoolLayer->layer.out_h; out_y++)
		{
			x = -PPoolLayer->padding;
			for (uint16_t out_x = 0; out_x < PPoolLayer->layer.out_w; out_x++)
			{
				grad_value = outVolu->getGradValue(outVolu, out_x, out_y, out_d);
				uint16_t ox = PPoolLayer->switchxy->getValue(PPoolLayer->switchxy, out_x, out_y, out_d);
				uint16_t oy = PPoolLayer->switchxy->getGradValue(PPoolLayer->switchxy, out_x, out_y, out_d);
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

float32_t PoolLayerBackwardOutput(TPPoolLayer PPoolLayer, TTensor Tensor)
{
	return 0.00;
}

void PoolLayerFree(TPPoolLayer PPoolLayer)
{
	FreeVolume(PPoolLayer->layer.in_v);
	FreeVolume(PPoolLayer->layer.out_v);
	FreeVolume(PPoolLayer->switchxy);
	// FreeVolume(PPoolLayer->switchy);
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
	// PFullyConnLayer->l1_decay_mul = LayerOption.l1_decay_mul;
	// PFullyConnLayer->l2_decay_mul = LayerOption.l2_decay_mul;
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

	for (uint16_t i = 0; i < PFullyConnLayer->filters->filterNumber; i++)
	{
		PFullyConnLayer->filters->init(PFullyConnLayer->filters->volumes[i], PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputPoints, 0);
		PFullyConnLayer->filters->volumes[i]->fillGauss(PFullyConnLayer->filters->volumes[i]->weight);
	}

	PFullyConnLayer->layer.out_w = 1;
	PFullyConnLayer->layer.out_h = 1;
	PFullyConnLayer->layer.out_depth = PFullyConnLayer->filters->filterNumber;
	PFullyConnLayer->layer.out_v = MakeVolume(PFullyConnLayer->layer.out_w, PFullyConnLayer->layer.out_h, PFullyConnLayer->layer.out_depth);
	PFullyConnLayer->layer.out_v->init(PFullyConnLayer->layer.out_v, PFullyConnLayer->layer.out_w, PFullyConnLayer->layer.out_h,
									   PFullyConnLayer->layer.out_depth, PFullyConnLayer->bias);

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
		LOGINFO("FullyConnLayer resize filters from %d x %d x %d to %d x %d x %d", PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, PFullyConnLayer->filters->_depth, PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputLength);
		FreeFilters(PFullyConnLayer->filters);
		bool ret = ResizeFilters(PFullyConnLayer->filters, PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputLength, PFullyConnLayer->filters->filterNumber);
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
	// outVolu->fillZero(outVolu->weight_d);
	//  uint16_t x, y;
	for (uint16_t out_d = 0; out_d < PFullyConnLayer->layer.out_depth; out_d++)
	{
		TPVolume filter = PFullyConnLayer->filters->volumes[out_d];
		sum = 0.00;

		uint16_t inputPoints = inVolu->_w * inVolu->_h * inVolu->_depth;

		for (uint16_t ip = 0; ip < inputPoints; ip++)
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
/// y = wx+b
void FullyConnLayerBackward(TPFullyConnLayer PFullyConnLayer)
{
	float32_t grad_value = 0.00;
	TPVolume inVolu = PFullyConnLayer->layer.in_v;
	TPVolume outVolu = PFullyConnLayer->layer.out_v;
	inVolu->fillZero(inVolu->weight_d);

	for (uint16_t out_d = 0; out_d < PFullyConnLayer->layer.out_depth; out_d++)
	{
		TPVolume filter = PFullyConnLayer->filters->volumes[out_d];
		grad_value = outVolu->weight_d->buffer[out_d];

		uint16_t inputPoints = inVolu->_w * inVolu->_h * inVolu->_depth;

		for (uint16_t out_l = 0; out_l < inputPoints; out_l++)
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

float32_t FullyConnLayerBackwardOutput(TPFullyConnLayer PFullyConnLayer, TTensor Tensor)
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
	for (uint16_t out_d = 0; out_d < PFullyConnLayer->layer.out_depth; out_d++)
	{
		TPResponse PResponse = malloc(sizeof(TResponse));
		if (PResponse != NULL)
		{
			PResponse->filter = PFullyConnLayer->filters->volumes[out_d]->weight;
			PResponse->filterGrads = PFullyConnLayer->filters->volumes[out_d]->weight_d;
			PResponse->l1_decay_mul = PFullyConnLayer->l1_decay_mul;
			PResponse->l2_decay_mul = PFullyConnLayer->l2_decay_mul;
			PResponse->fillZero = FillZero;
			PResponse->free = FreeTensor;
			tPResponses[out_d] = PResponse;
		}
	}
	TPResponse PResponse = malloc(sizeof(TResponse));
	if (PResponse != NULL)
	{
		PResponse->filter = PFullyConnLayer->biases->weight;
		PResponse->filterGrads = PFullyConnLayer->biases->weight_d;
		PResponse->l1_decay_mul = 0;
		PResponse->l2_decay_mul = 0;
		PResponse->fillZero = FillZero;
		PResponse->free = FreeTensor;
		tPResponses[PFullyConnLayer->layer.out_depth] = PResponse;
	}
	return tPResponses;
}

void FullyConnLayerFree(TPFullyConnLayer PFullyConnLayer)
{
	FreeVolume(PFullyConnLayer->layer.in_v);
	FreeVolume(PFullyConnLayer->layer.out_v);
	FreeVolume(PFullyConnLayer->biases);
	FreeFilters(PFullyConnLayer->filters);
	// for (uint16_t i = 0; i < PFullyConnLayer->filters->_depth; i++)
	//{
	//	PFullyConnLayer->filters->free(PFullyConnLayer->filters->volumes[i]);
	// }
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
		LOGINFO("Softmax resize out_v from %d x %d x %d to %d x %d x %d", PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth, 1, 1, inputLength);
		PSoftmaxLayer->layer.out_w = 1;
		PSoftmaxLayer->layer.out_h = 1;
		PSoftmaxLayer->layer.out_depth = inputLength;

		if (PSoftmaxLayer->layer.out_v != NULL)
		{
			FreeVolume(PSoftmaxLayer->layer.out_v);
			FreeTensor(PSoftmaxLayer->exp);
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
	float32_t max_value = MINI_NEGATIVE_NUMBER;
	float32_t sum = 0.0;
	float32_t expv = 0.0;
	float32_t temp = 0.0;
	TPVolume inVolu = PSoftmaxLayer->layer.in_v;
	TPVolume outVolu = PSoftmaxLayer->layer.out_v;

	softmaxLayOutResize(PSoftmaxLayer);
	outVolu->fillZero(outVolu->weight);

	for (uint16_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
	{
		if (inVolu->weight->buffer[out_d] > max_value)
		{
			max_value = inVolu->weight->buffer[out_d];
		}
	}
	for (uint16_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
	{
		temp = inVolu->weight->buffer[out_d] - max_value;
		expv = exp(temp);
		PSoftmaxLayer->exp->buffer[out_d] = expv;
		sum = sum + expv;
	}
	for (uint16_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
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
	for (uint16_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
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
	FreeVolume(PSoftmaxLayer->layer.in_v);
	FreeVolume(PSoftmaxLayer->layer.out_v);
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
		PNeuralNet->layers = malloc(sizeof(TPLayer) * PNeuralNet->depth);
		if (PNeuralNet->layers != NULL)
		{
			TPInputLayer InputLayer = malloc(sizeof(TInputLayer));
			InputLayer->init = InputLayerInit;
			InputLayer->free = InputLayerFree;
			InputLayer->forward = InputLayerForward;
			InputLayer->backward = InputLayerBackward;
			InputLayer->computeLoss = NULL;	   // InputLayerBackwardLoss;
			InputLayer->backwardOutput = NULL; // InputLayerBackwardOutput;
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
			ConvLayer->backwardOutput = ConvolutionLayerBackwardOutput;
			ConvLayer->getParamsAndGrads = ConvolutionLayerGetParamsAndGradients;
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
			PoolLayer->backwardOutput = PoolLayerBackwardOutput;
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
			ReluLayer->backwardOutput = ReluLayerBackwardOutput;
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
			FullyConnLayer->backwardOutput = FullyConnLayerBackwardOutput;
			FullyConnLayer->getParamsAndGrads = FullyConnLayerGetParamsAndGrads;
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
}

void NeuralNetForward(TPNeuralNet PNeuralNet, TPVolume PVolume)
{
	if (PVolume == NULL)
		return;
	TPVolume in_v = NULL;
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
		in_v = PNeuralNet->layers[layerIndex - 1]->out_v;
		if (in_v == NULL)
		{
			LOGERROR("Input volume is null, layerIndex=%d layereType=%d\n", layerIndex, PNeuralNet->layers[layerIndex - 1]->LayerType);
			break;
		}
		PNeuralNet->layers[layerIndex]->in_v = in_v;

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

void NeuralNetGetParamsAndGrads(TPNeuralNet PNeuralNet)
{
	// TPResponse *pResponseResult;
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
			// pResponseResult = ((TPInputLayer) PNeuralNet->layers[layerIndex])->getParamsAndGrads();
			break;
		}

		case Layer_Type_Convolution:
		{
			TPResponse *pResponseResult = ((TPConvLayer)PNeuralNet->layers[layerIndex])->getParamsAndGrads(((TPConvLayer)PNeuralNet->layers[layerIndex]));
			for (uint16_t i = 0; i <= ((TPConvLayer)PNeuralNet->layers[layerIndex])->layer.out_depth; i++)
			{
				temp = realloc(PNeuralNet->trainning.pResponseResults, sizeof(TPResponse) * (PNeuralNet->trainning.responseCount + 1));
				if (temp != NULL)
				{
					PNeuralNet->trainning.pResponseResults = temp;
					PNeuralNet->trainning.pResponseResults[PNeuralNet->trainning.responseCount] = pResponseResult[i];
					PNeuralNet->trainning.responseCount++;
				}
			}
			free(pResponseResult);
			break;
		}

		case Layer_Type_Pool:
		{
			// pResponseResult = ((TPPoolLayer) PNeuralNet->layers[layerIndex])->getParamsAndGrads();
			break;
		}

		case Layer_Type_ReLu:
		{
			// pResponseResult = ((TPReluLayer) PNeuralNet->layers[layerIndex])->getParamsAndGrads();
			break;
		}

		case Layer_Type_FullyConnection:
		{
			TPResponse *pResponseResult = ((TPFullyConnLayer)PNeuralNet->layers[layerIndex])->getParamsAndGrads(((TPFullyConnLayer)PNeuralNet->layers[layerIndex]));
			for (uint16_t i = 0; i <= ((TPConvLayer)PNeuralNet->layers[layerIndex])->layer.out_depth; i++)
			{
				temp = realloc(PNeuralNet->trainning.pResponseResults, sizeof(TPResponse) * (PNeuralNet->trainning.responseCount + 1));
				if (temp != NULL)
				{
					PNeuralNet->trainning.pResponseResults = temp;
					PNeuralNet->trainning.pResponseResults[PNeuralNet->trainning.responseCount] = pResponseResult[i];
					PNeuralNet->trainning.responseCount++;
				}
			}
			free(pResponseResult);
			break;
		}

		case Layer_Type_SoftMax:
		{
			// pResponseResult = ((TPSoftmaxLayer) PNeuralNet->Layers[layerIndex])->getParamsAndGrads();
			break;
		}
		default:
			break;
		}
	}
}

void NeuralNetComputeCostLoss(TPNeuralNet PNeuralNet)
{
	TPSoftmaxLayer PSoftmaxLayer = ((TPSoftmaxLayer)PNeuralNet->layers[PNeuralNet->depth - 1]);
	PSoftmaxLayer->expected_value = PNeuralNet->trainning.labelIndex;
	PNeuralNet->trainning.cost_loss = PSoftmaxLayer->computeLoss(PSoftmaxLayer);
	PNeuralNet->trainning.sum_cost_loss = PNeuralNet->trainning.sum_cost_loss + PNeuralNet->trainning.cost_loss;
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
	for (uint16_t i = 0; i < PSoftmaxLayer->layer.out_v->weight->length; i++)
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
	TPLayer pNetLayer = (PNeuralNet->layers[LayerIndex]);
	// for (uint16_t i = 0; i < pNetLayer->out_v->weight->length; i++)
	//{
	//	LOGINFO("LayerType=%s PNeuralNet->depth=%d weight=%f", CNNTypeName[pNetLayer->LayerType], PNeuralNet->depth - 1, pNetLayer->out_v->weight->buffer[i]);
	// }
	if (InOut == 0)
	{
		LOGINFO("layers[%d] out_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
		pNetLayer->out_v->print(pNetLayer->out_v, 0);
	}
	else if (InOut == 1)
	{
		LOGINFO("layers[%d] in_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth);
		pNetLayer->in_v->print(pNetLayer->in_v, 0);
	}
	else if (pNetLayer->LayerType = Layer_Type_Convolution)
	{
		for (uint16_t i = 0; i < ((TPConvLayer)pNetLayer)->filters->filterNumber; i++)
		{
			LOGINFO("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d/%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], ((TPConvLayer)pNetLayer)->filters->_w, ((TPConvLayer)pNetLayer)->filters->_h, ((TPConvLayer)pNetLayer)->filters->_depth, i, ((TPConvLayer)pNetLayer)->filters->filterNumber);
			((TPConvLayer)pNetLayer)->filters->volumes[i]->print(((TPConvLayer)pNetLayer)->filters->volumes[i], 0);
		}
	}
	else if (pNetLayer->LayerType = Layer_Type_FullyConnection)
	{
		for (uint16_t i = 0; i < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; i++)
		{
			LOGINFO("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d/%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], ((TPConvLayer)pNetLayer)->filters->_w, ((TPConvLayer)pNetLayer)->filters->_h, ((TPConvLayer)pNetLayer)->filters->_depth, i, ((TPConvLayer)pNetLayer)->filters->filterNumber);
			((TPFullyConnLayer)pNetLayer)->filters->volumes[i]->print(((TPFullyConnLayer)pNetLayer)->filters->volumes[i], 0);
		}
	}
}

void NeuralNetPrintFilters(TPNeuralNet PNeuralNet, uint16_t LayerIndex, uint8_t InOut)
{
	TPLayer pNetLayer = (PNeuralNet->layers[LayerIndex]);

	if (InOut == 0)
	{
		LOGINFO("layers[%d] out_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
		pNetLayer->out_v->print(pNetLayer->out_v, 0);
	}
	else if (InOut == 1)
	{
		LOGINFO("layers[%d] in_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
		pNetLayer->in_v->print(pNetLayer->in_v, 0);
	}
	else if (pNetLayer->LayerType = Layer_Type_Convolution)
	{
		LOGINFO("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber);
		for (uint16_t i = 0; i < ((TPConvLayer)pNetLayer)->filters->filterNumber; i++)
		{
			((TPConvLayer)pNetLayer)->filters->volumes[i]->print(((TPConvLayer)pNetLayer)->filters->volumes[i], 0);
		}
	}
	else if (pNetLayer->LayerType = Layer_Type_FullyConnection)
	{
		LOGINFO("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber);
		for (uint16_t i = 0; i < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; i++)
		{
			((TPFullyConnLayer)pNetLayer)->filters->volumes[i]->print(((TPFullyConnLayer)pNetLayer)->filters->volumes[i], 0);
		}
	}
}

void NeuralNetPrintGradients(TPNeuralNet PNeuralNet, uint16_t LayerIndex, uint8_t InOut)
{
	// float32_t maxv = 0;
	// uint16_t maxi = 0;
	TPLayer pNetLayer = ((TPLayer)PNeuralNet->layers[LayerIndex]);
	// for (uint16_t i = 0; i < pNetLayer->out_v->weight->length; i++)
	//{
	//	//LOGINFO("LayerType=%s PNeuralNet->depth=%d weight_grad=%f", CNNTypeName[pNetLayer->LayerType], PNeuralNet->depth - 1, pNetLayer->out_v->weight_grad->buffer[i]);
	// }
	if (InOut == 0)
		pNetLayer->out_v->print(pNetLayer->out_v, 1);
	else if (InOut == 1)
		pNetLayer->in_v->print(pNetLayer->in_v, 1);
	else if (pNetLayer->LayerType = Layer_Type_Convolution)
	{
		for (uint16_t i = 0; i < ((TPConvLayer)pNetLayer)->filters->filterNumber; i++)
		{
			((TPConvLayer)pNetLayer)->filters->volumes[i]->print(((TPConvLayer)pNetLayer)->filters->volumes[i], 1);
		}
	}
	else if (pNetLayer->LayerType = Layer_Type_FullyConnection)
	{
		for (uint16_t i = 0; i < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; i++)
		{
			((TPFullyConnLayer)pNetLayer)->filters->volumes[i]->print(((TPFullyConnLayer)pNetLayer)->filters->volumes[i], 1);
		}
	}
}

void NeuralNetPrint(char *Name, TPTensor PTensor)
{
	for (uint16_t i = 0; i < PTensor->length; i++)
	{
		if (i % 16 == 0)
			LOG("\n");
		LOG("%s[%d]=%f ", Name, i, PTensor->buffer[i]);
	}
	LOG("\n");
}

void NeuralNetPrintTrainningInfo(TPNeuralNet PNeuralNet)
{	

	LOGINFO("datasetTotal:%d datasetIndex:%d epochCount:%d sampleCount:%d labelIndex:%d batchCount:%d",
			PNeuralNet->trainning.datasetTotal,
			PNeuralNet->trainning.datasetIndex,
			PNeuralNet->trainning.epochCount,
			PNeuralNet->trainning.sampleCount,
			PNeuralNet->trainning.labelIndex,
			PNeuralNet->trainning.batchCount);

	LOGINFO("costLost:%f l1_decay_loss:%f l2_decay_loss:%f trainingAccuracy:%f testingAccuracy:%f",
			PNeuralNet->trainning.sum_cost_loss / PNeuralNet->trainning.sampleCount / 1.00,
			PNeuralNet->trainning.l1_decay_loss / PNeuralNet->trainning.sampleCount/ 1.00,
			PNeuralNet->trainning.l2_decay_loss / PNeuralNet->trainning.sampleCount/ 1.00,
			PNeuralNet->trainning.trainingAccuracy / PNeuralNet->trainning.sampleCount/ 1.00,
			PNeuralNet->trainning.testingAccuracy / PNeuralNet->trainning.sampleCount / 1.00);
	LOGINFO("forward ticks:%d backward ticks:%d optimTime ticks:%d",
			PNeuralNet->fwTime,
			PNeuralNet->bwTime,
			PNeuralNet->optimTime);
}

void NeuralNetTrain(TPNeuralNet PNeuralNet, TPVolume PVolume)
{
	TPTensor weight, grads, tempPTensor1, tempPTensor2;
	TPResponse *pResponseResults;
	TPResponse pResponse;
	float32_t costLost;
	float32_t l1_decay;
	float32_t l2_decay;
	float32_t l1_decay_grad;
	float32_t l2_decay_grad;
	float32_t gradij;
	float32_t bias1, bias2;
	float32_t dx;
	tempPTensor1 = NULL;
	tempPTensor2 = NULL;
	float32_t l1_decay_loss = 0;
	float32_t l2_decay_loss = 0;
	uint32_t starTick;
	// uint32_t endTick;
	/////////////////////////////////////////////////////////////////////////////////////
	starTick = GetTimestamp();
	PNeuralNet->forward(PNeuralNet, PVolume);
	PNeuralNet->getCostLoss(PNeuralNet);
	PNeuralNet->fwTime = GetTimestamp() - starTick;
	// PNeuralNet->printWeights(PNeuralNet, PNeuralNet->depth - 2, 1);
	// PNeuralNet->printWeights(PNeuralNet, PNeuralNet->depth - 2, 0);

	PNeuralNet->printWeights(PNeuralNet, PNeuralNet->depth - 1, 1);
	PNeuralNet->printWeights(PNeuralNet, PNeuralNet->depth - 1, 0);
	PNeuralNet->print("EXP", ((TPSoftmaxLayer)PNeuralNet->layers[PNeuralNet->depth - 1])->exp);
    ///////////////////////////////////////////////////////////////////////////////////////
	//debug 
	PNeuralNet->trainning.trainningGoing = false;
	return;

	starTick = GetTimestamp();
	PNeuralNet->backward(PNeuralNet);
	PNeuralNet->bwTime = GetTimestamp() - starTick;

	PNeuralNet->trainning.batchCount++;
	if (PNeuralNet->trainningParam.batch_size <= 0)
		return;
	// 小批量阀值
	if (PNeuralNet->trainning.batchCount % PNeuralNet->trainningParam.batch_size != 0)
	return;
	////////////////////////////////////////////////////////////////////////////
	////小批量梯度计算
	starTick = GetTimestamp();
	PNeuralNet->getParamsAndGrads(PNeuralNet);
	if (PNeuralNet->trainning.responseCount <= 0)
	{
		LOGERROR("No ResponseResults! trainning.responseCount <= 0");
		PNeuralNet->trainning.trainningGoing = false;
		return;
	}
	pResponseResults = PNeuralNet->trainning.pResponseResults;
	if (pResponseResults == NULL)
	{
		LOGERROR("No ResponseResults! pResponseResults=NULL");
		return;
	}
	if (PNeuralNet->trainning.gsum1 == NULL || PNeuralNet->trainning.gsumCount <= 0)
	{
		PNeuralNet->trainning.gsum1 = malloc(sizeof(TPTensor) * PNeuralNet->trainning.responseCount);
		PNeuralNet->trainning.gsum2 = malloc(sizeof(TPTensor) * PNeuralNet->trainning.responseCount);

		if (PNeuralNet->trainning.gsum1 == NULL || PNeuralNet->trainning.gsum2 == NULL)
		{
			LOGERROR("No trainning.gsum1 or trainning.gsum2=NULL");
			return;
		}
		for (uint16_t i = 0; i < PNeuralNet->trainning.responseCount; i++)
		{
			PNeuralNet->trainning.gsum1[i] = MakeTensor(pResponseResults[i]->filter->length);
			PNeuralNet->trainning.gsum2[i] = MakeTensor(pResponseResults[i]->filter->length);
		}
	}

	for (uint16_t i = 0; i < PNeuralNet->trainning.responseCount; i++)
	{
		pResponse = pResponseResults[i];
		weight = pResponse->filter;
		grads = pResponse->filterGrads;
		if (weight->length <= 0 || grads->length <= 0)
		{
			LOGERROR("pResponse weight length <=0 ");
			break;
		}

		if (PNeuralNet->trainning.gsum1 != NULL && PNeuralNet->trainning.gsum2 != NULL)
		{
			// tempPTensor1 = MakeTensor(pResponse->filter->length);
			// tempPTensor2 = MakeTensor(pResponse->filter->length);
			tempPTensor1 = PNeuralNet->trainning.gsum1[i];
			tempPTensor2 = PNeuralNet->trainning.gsum2[i];
		}

		l1_decay = PNeuralNet->trainningParam.l1_decay * pResponse->l1_decay_mul;
		l2_decay = PNeuralNet->trainningParam.l2_decay * pResponse->l2_decay_mul;

		for (uint16_t j = 0; j < weight->length; j++) // update weight
		{
			l1_decay_loss = l1_decay_loss + l1_decay * abs(weight->buffer[j]);
			l2_decay_loss = l2_decay_loss + l2_decay * weight->buffer[j] * weight->buffer[j] / 2;

			if (weight->buffer[j] > 0)
				l1_decay_grad = l1_decay;
			else
				l1_decay_grad = -l1_decay;

			l2_decay_grad = l2_decay * weight->buffer[j];

			gradij = grads->buffer[j];
			gradij = (l1_decay_grad + l2_decay_grad + gradij) / PNeuralNet->trainningParam.batch_size;

			if (tempPTensor1 == NULL || tempPTensor2 == NULL)
			{
				LOGERROR("tempPTensor1 or tempPTensor2=NULL");
				break;
			}
			switch (PNeuralNet->trainningParam.optimize_method)
			{
			case Optm_Adam:
				tempPTensor1->buffer[j] = tempPTensor1->buffer[j] * PNeuralNet->trainningParam.beta1 + (1 - PNeuralNet->trainningParam.beta1) * gradij;
				tempPTensor2->buffer[j] = tempPTensor1->buffer[j] * PNeuralNet->trainningParam.beta2 + (1 - PNeuralNet->trainningParam.beta2) * gradij * gradij;

				bias1 = tempPTensor1->buffer[j] * (1 - pow(PNeuralNet->trainningParam.beta1, PNeuralNet->trainning.batchCount));
				bias2 = tempPTensor1->buffer[j] * (1 - pow(PNeuralNet->trainningParam.beta2, PNeuralNet->trainning.batchCount));
				dx = -PNeuralNet->trainningParam.learning_rate * bias1 / (sqrt(bias2) + PNeuralNet->trainningParam.eps);

				weight->buffer[j] = weight->buffer[j] + dx;
				break;
			case Optm_Adagrad:
				tempPTensor1->buffer[j] = tempPTensor1->buffer[j] + gradij * gradij;
				dx = -PNeuralNet->trainningParam.learning_rate / sqrt(tempPTensor1->buffer[j] + PNeuralNet->trainningParam.eps) * gradij;
				weight->buffer[j] = weight->buffer[j] + dx;
				break;
			default:
				break;
			}
		}
	}

	PNeuralNet->trainning.l1_decay_loss = PNeuralNet->trainning.l1_decay_loss + l1_decay_loss;
	PNeuralNet->trainning.l2_decay_loss = PNeuralNet->trainning.l2_decay_loss + l2_decay_loss;

	for (uint16_t i = 0; i < PNeuralNet->trainning.responseCount; i++)
	{
		pResponse = pResponseResults[i];
		free(pResponse);
	}
	free(PNeuralNet->trainning.pResponseResults);
	PNeuralNet->trainning.responseCount = 0;
	PNeuralNet->trainning.pResponseResults = NULL;
	PNeuralNet->optimTime = GetTimestamp() - starTick;
}

/// @brief ///////////////////////////////////////////////////////////////////////
/// @param PNeuralNet
void NeuralNetSave(TPNeuralNet PNeuralNet)
{
}

void NeuralNetLoad(TPNeuralNet PNeuralNet)
{
}

char *NeuralNetGetName(TLayerType LayerType)
{
	return CNNTypeName[LayerType];
}

TPNeuralNet CreateNeuralNetCNN()
{
	TPNeuralNet PNeuralNet = malloc(sizeof(TNeuralNet));
	PNeuralNet->init = NeuralNetInit;
	PNeuralNet->free = NeuralNetFree;
	PNeuralNet->forward = NeuralNetForward;
	PNeuralNet->backward = NeuralNetBackward;
	PNeuralNet->getParamsAndGrads = NeuralNetGetParamsAndGrads;
	PNeuralNet->getCostLoss = NeuralNetComputeCostLoss;
	PNeuralNet->getPredictions = NeuralNetUpdatePrediction;
	PNeuralNet->getMaxPrediction = NeuralNetGetMaxPrediction;
	PNeuralNet->train = NeuralNetTrain;
	PNeuralNet->save = NeuralNetSave;
	PNeuralNet->load = NeuralNetLoad;
	PNeuralNet->printGradients = NeuralNetPrintGradients;
	PNeuralNet->printWeights = NeuralNetPrintWeights;
	PNeuralNet->printTrainningInfo = NeuralNetPrintTrainningInfo;
	PNeuralNet->print = NeuralNetPrint;
	PNeuralNet->getName = NeuralNetGetName;
}
