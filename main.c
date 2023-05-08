// main.cpp: 定义应用程序的入口点。
//

#include "ann-cnn.h"

#if 1
int main()
{
	printf("\nHellow VS Code!\n");

#if defined(__STDC_VERSION__)
	printf("__STDC_VERSION__ Version %ld\n", __STDC_VERSION__);
#endif
	printf("////////////////////////////////////////////////////////////\n");
	LOGINFO("InitNeuralNet_CNN...");
	InitNeuralNet_CNN();
	LOGINFO("InitLeaningParameter...");
	InitLeaningParameter();
	PrintNetInformation();
	printf("\n");
	printf("0: exit");
	printf("\t\t\t1: start trainning\n");

	printf("\nplease select a menu item to continue:");
	int input_int = 3;
	// scanf("%d", &input_int);
	printf("\n");
	switch (input_int)
	{
	case 1:
		LOGINFO("NeuralNet_Start...");
		NeuralNet_Start();
		break;
	case 0:
		break;
	default:
		NeuralNet_Start();
		break;
	}

	return 0;
}
#endif

#if 0
int main()
{
	float *ptr;
	int i;
	float ft = -3.1415926;
	// 动态申请一块内存，存储10个浮点数
	ptr = (float *)malloc(10 * sizeof(float));

	// 检查内存是否申请成功
	if (ptr == NULL)
	{
		printf("内存申请失败！\n");
		exit(1);
	}

	// 初始化数组
	for (i = 0; i < 10; i++)
	{
		ptr[i] = (i * 0.1);
	}

	// 输出数组
	printf("输出数组////////////////////////////////////\n");
	for (i = 0; i < 10; i++)
	{
		printf("%.3f ", ptr[i]);
	}

	printf("\n///////////////////////////////////////////\n");
	printf("%f\n", (ft));
	printf("%f\n", 0.120002);
	// 释放内存
	free(ptr);

	TPVolume pv = MakeVolume(1, 1, 10);
	pv->init(pv, 1, 1, 10, 0.10);
	// pv->weight->buffer[0]=0.25;
	// pv->setValue(pv,0,0,6,6.1);
	// pv->print(pv, 0);

	pv->fillGauss(pv->weight);
	pv->print(pv, 0);

	return 0;
}
#endif
#if 0
int main()
{
	for (int i = 0; i < 50000; i++)
	{
		printf("qwer0=%d\n", i);
		printf("qwer1=%d\n", i);
		printf("qwer2=%d\n", i);
		//printf("\b\b\b");
		printf("\033[3A");
		
	}
}
#endif