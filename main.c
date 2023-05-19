/////////////////////////////////////////////////////////////////////////////////////////////
// main.cpp: 定义应用程序的入口点。
// test
/////////////////////////////////////////////////////////////////////////////////////////////
#include "ann-cnn.h"
#include "ann-dataset.h"
extern TPNeuralNet PNeuralNetCNN;
void showBanner(void);
/////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
	char str[32] = "";
	int cmd = 0;
	int layer = 0;
	showBanner();
#if defined(__STDC_VERSION__)
	printf("__STDC_VERSION__ Version %ld\n", __STDC_VERSION__);
#endif
	printf("version:1.0.0.0\n");
	printf("////////////////////////////////////////////////////////////////////////////////////\n");
	LOGINFO("InitNeuralNet_CNN...");
	NeuralNetCNNInit();
	LOGINFO("InitLeaningParameter...");
	NeuralNetCNNInitLeaningParameter();
	NeuralNetCNNPrintNetInformation();

	while (true)
	{
		printf("\n");
		printf("\nplease select a menu item to continue...\n");
		printf("00: Exit.\n");
		printf("01: Print weight usage:2 10,two parameters,the first is the command number and the second is the network layer index.\n");
		printf("02: Print gradients usage same as print weight.\n");
		printf("03: Print neural network information,displays the network structure information.\n");

		printf("04: Start trainning CIFAR-10 one by one,learn one CIFAR-10 picture at a time.\n");
		printf("05: Start trainning CIFAR-10 by batch,learn a batch CIFAR-10 picture at a time.\n");
		printf("06: Start trainning CIFAR-10 without saving weights,learning of the CIFAR-10 50,000 images do not save weights.\n");
		printf("07: Start trainning CIFAR-10 and saving weights,learning of the CIFAR-10 50,000 images and save weights to file cnn.w.\n");
		
		printf("08: Start trainning CIFAR-100 one by one,learn one CIFAR-100 picture at a time.\n");
		printf("09: Start trainning CIFAR-100 by batch,learn a batch CIFAR-100 picture at a time.\n");
		printf("10: Start trainning CIFAR-100 without saving weights,learning of the CIFAR-100 50,000 images do not save weights.\n");
		printf("11: Start trainning CIFAR-100 and saving weights,learning of the CIFAR-100 50,000 images and save weights to file cnn.w.\n");

		printf("12: Save weights to   file cnn.w\n");
		printf("13: Load weights from file cnn.w\n");
		printf("\nplease select a menu item to continue:");

		gets(str);
		sscanf(str, "%d %d", &cmd, &layer);
		printf("your choose command=%d layer=%d\n", cmd, layer);

		switch (cmd)
		{
		case 0:
			CloseDataset();
			return 0;
		case 1:
			if (PNeuralNetCNN != NULL)
			{
				PNeuralNetCNN->printWeights(PNeuralNetCNN, layer, 1);
				PNeuralNetCNN->printWeights(PNeuralNetCNN, layer, 0);
				PNeuralNetCNN->printWeights(PNeuralNetCNN, layer, 2);
			}
			break;
		case 2:
			if (PNeuralNetCNN != NULL)
			{
				PNeuralNetCNN->printGradients(PNeuralNetCNN, layer, 1);
				PNeuralNetCNN->printGradients(PNeuralNetCNN, layer, 0);
				PNeuralNetCNN->printGradients(PNeuralNetCNN, layer, 2);
			}
			break;
		case 3:
			NeuralNetCNNPrintLayerInfor();
			break;

		case 4:
			LOGINFO("NeuralNet start trainning...");
			PNeuralNetCNN->trainning.trainingSaving = false;
			PNeuralNetCNN->trainning.one_by_one = true;
			PNeuralNetCNN->trainning.batch_by_batch = false;
			NeuralNetStartTrainning();
			break;
		case 5:
			LOGINFO("NeuralNet start trainning...");
			PNeuralNetCNN->trainning.trainingSaving = false;
			PNeuralNetCNN->trainning.one_by_one = false;
			PNeuralNetCNN->trainning.batch_by_batch = true;
			NeuralNetStartTrainning();
			break;
		case 6:
			LOGINFO("NeuralNet start trainning...");
			PNeuralNetCNN->trainning.trainingSaving = false;
			PNeuralNetCNN->trainning.batch_by_batch = false;
			PNeuralNetCNN->trainning.one_by_one = false;
			NeuralNetStartTrainning();
			break;
		case 7:
			LOGINFO("NeuralNet start trainning...");
			PNeuralNetCNN->trainning.trainingSaving = true;
			PNeuralNetCNN->trainning.one_by_one = false;
			PNeuralNetCNN->trainning.batch_by_batch = false;
			NeuralNetStartTrainning();
			break;

		case 8:
			LOGINFO("NeuralNet start trainning...");
			PNeuralNetCNN->trainning.data_type = Cifar100;
			PNeuralNetCNN->trainning.trainingSaving = false;
			PNeuralNetCNN->trainning.one_by_one = true;
			PNeuralNetCNN->trainning.batch_by_batch = false;
			NeuralNetStartTrainning();
			break;
		case 9:
			LOGINFO("NeuralNet start trainning...");
			PNeuralNetCNN->trainning.data_type = Cifar100;
			PNeuralNetCNN->trainning.trainingSaving = false;
			PNeuralNetCNN->trainning.one_by_one = false;
			PNeuralNetCNN->trainning.batch_by_batch = true;
			NeuralNetStartTrainning();
			break;
		case 10:
			LOGINFO("NeuralNet start trainning...");
			PNeuralNetCNN->trainning.data_type = Cifar100;
			PNeuralNetCNN->trainning.trainingSaving = false;
			PNeuralNetCNN->trainning.batch_by_batch = false;
			PNeuralNetCNN->trainning.one_by_one = false;
			NeuralNetStartTrainning();
			break;
		case 11:
			LOGINFO("NeuralNet start trainning...");
			PNeuralNetCNN->trainning.data_type = Cifar100;
			PNeuralNetCNN->trainning.trainingSaving = true;
			PNeuralNetCNN->trainning.one_by_one = false;
			PNeuralNetCNN->trainning.batch_by_batch = false;
			NeuralNetStartTrainning();
			break;

		case 12:
			if (PNeuralNetCNN != NULL)
			{
				LOGINFO("NeuralNet saving...");
				PNeuralNetCNN->save(PNeuralNetCNN);
			}
			break;
		case 13:
			if (PNeuralNetCNN != NULL)
			{
				LOGINFO("NeuralNet loading...");
				PNeuralNetCNN->load(PNeuralNetCNN);
			}
			break;
		default:
			printf("\n");
			break;
		}
		fflush(stdin);
	}
	return 0;
}

void showBanner(void)
{
	char pwd_path[100]; // print work directory
	FILE *fp = fopen("../banner.txt", "r");

	if (fp != NULL)
	{
		while (fgets(pwd_path, sizeof(pwd_path), fp) != NULL)
		{
			printf("%s\n", pwd_path);
		}
	}

	if (fp != NULL)
		fclose(fp);
	// if (getcwd(pwd_path, 512) != NULL)
	//	LOGINFO("%s\n", pwd_path);
}
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