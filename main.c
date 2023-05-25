/////////////////////////////////////////////////////////////////////////////////////////////
// main.c: 定义应用程序的入口点。
// test demo
// 要运行本程序进行训练，需要下载C语言版本Cifar-10 Cifar-100 数据集
// 下载地址：http://www.cs.toronto.edu/~kriz/cifar.html
// 下载解压后放在工程目录下面
/////////////////////////////////////////////////////////////////////////////////////////////
#include "ann-cnn.h"
#include "ann-dataset.h"
#define NET_CIFAR10_NAME "Cifar10"
#define NET_CIFAR100_NAME "Cifar100"
TPNeuralNet PNeuralNetCNN_16 = NULL;
extern TPNeuralNet PNeuralNetCNN_Cifar10, PNeuralNetCNN_Cifar100;
extern void NeuralNetStartTrainning(TPNeuralNet PNeuralNetCNN);
extern void NeuralNetInitLeaningParameter(TPNeuralNet PNeuralNetCNN);
extern void NeuralNetPrintNetInformation(TPNeuralNet PNeuralNetCNN);
extern TPNeuralNet NeuralNetInit_C_CNN_16(char* NetName);
void showBanner(void);
/////////////////////////////////////////////////////////////////////////////////////////////
/// 要运行本程序进行训练，需要下载C语言版本Cifar-10 Cifar-100 数据集 
/// 下载地址：http://www.cs.toronto.edu/~kriz/cifar.html
/// 下载解压后放在工程目录下面
/// 创建两个深度学习网络PNeuralNetCNN_Cifar10, PNeuralNetCNN_Cifar100
/// 同时学习Cifar10和Cifar100数据集
/// 
int main()
{
	char cmd_str[32] = "";
	char net_name[32] = NET_CIFAR10_NAME;
	int net_cmd = 0;
	int net_layer = 0;
	int net_io = 0; //0:output,1 input,2:filters
	int log_lines = 0;
	showBanner();
#if defined(__STDC_VERSION__)
	printf("__STDC_VERSION__ Version %ld\n", __STDC_VERSION__);
#endif
	printf("version:1.0.0.0\n");
	printf("////////////////////////////////////////////////////////////////////////////////////\n");
	LOG("InitNeuralNet_CNN Cifar10...\n");
	NeuralNetInit_Cifar10_11();
	NeuralNetInitLeaningParameter(PNeuralNetCNN_Cifar10);
	NeuralNetPrintNetInformation(PNeuralNetCNN_Cifar10);

	LOG("\n");
	LOG("InitNeuralNet_CNN Cifar100...\n");
	NeuralNetInit_Cifar100();
	NeuralNetInitLeaningParameter(PNeuralNetCNN_Cifar100);
	NeuralNetPrintNetInformation(PNeuralNetCNN_Cifar100);

    //一个深层网络用来学习Cifar100数据集
	PNeuralNetCNN_16 = NeuralNetInit_C_CNN_16("Cifar100");
	while (true)
	{
		printf("\033[%dB", log_lines);
		log_lines = 20;
		// printf("\n");
		printf("\nplease select a menu item to continue...\n");
		printf("00: Exit.\n");
		printf("01: Print weight Usage:Command Layer Type Name,ex:1 10 0 Cifar10,the first is command,the second is layer index,\n");
		printf("    the third is type(1:input,0:output:2:filters) and the fourth is net name Cifar10/Cifar100.\n");
		printf("02: Print gradients Usage:2 10 0 Cifar10,it is same as print weight.\n");
		printf("03: Print neural network information,displays the network structure information.\n");

		printf("04: Start trainning CIFAR-10 one by one,learn one CIFAR-10 picture at a time.\n");
		printf("05: Start trainning CIFAR-10 batch by batch,learn a batch CIFAR-10 picture at a time.\n");
		printf("06: Start trainning CIFAR-10 learning of the CIFAR-10 50,000 images but do not save weights to file.\n");
		printf("07: Start trainning CIFAR-10 learning of the CIFAR-10 50,000 images and saving weights to file cnn.w.\n");

		printf("08: Start trainning CIFAR-100 one by one,learn one CIFAR-100 picture at a time.\n");
		printf("09: Start trainning CIFAR-100 batch by batch,learn a batch CIFAR-100 picture at a time.\n");
		printf("10: Start trainning CIFAR-100 learning of the CIFAR-100 50,000 images but do not save weights to file.\n");
		printf("11: Start trainning CIFAR-100 learning of the CIFAR-100 50,000 images and saving weights to file cnn.w.\n");

		printf("12: Start trainning CIFAR-100 and CIFAR-10 \n");

		printf("13: Save weights to   file cnn.w\n");
		printf("14: Load weights from file cnn.w\n");
		//printf("15: Save weights to bmp\n");
		printf("\nplease select a menu item to continue:");

		gets(cmd_str);
		sscanf(cmd_str, "%d %d %d %s", &net_cmd, &net_layer,&net_io, net_name);
		printf("command:%d layer:%d io:%d name:%s\n", net_cmd, net_layer, net_io, net_name);

		switch (net_cmd)
		{
		case 0:
			CloseTrainningDataset();
			CloseTestingDataset();
			return 0;
		case 1:
			if (PNeuralNetCNN_Cifar10 != NULL && (strcmp(net_name, NET_CIFAR10_NAME) == 0))
			{
				switch (net_io)
				{
				case 0:	PNeuralNetCNN_Cifar10->printWeights(PNeuralNetCNN_Cifar10, net_layer, 0); break;
				case 1:	PNeuralNetCNN_Cifar10->printWeights(PNeuralNetCNN_Cifar10, net_layer, 1); break;
				case 2:	PNeuralNetCNN_Cifar10->printWeights(PNeuralNetCNN_Cifar10, net_layer, 2); break;
				}
			}
			else if (PNeuralNetCNN_Cifar100 != NULL && (strcmp(net_name, NET_CIFAR100_NAME) == 0))
			{
				switch (net_io)
				{
				case 0:	PNeuralNetCNN_Cifar100->printWeights(PNeuralNetCNN_Cifar100, net_layer, 0); break;
				case 1:	PNeuralNetCNN_Cifar100->printWeights(PNeuralNetCNN_Cifar100, net_layer, 1); break;
				case 2:	PNeuralNetCNN_Cifar100->printWeights(PNeuralNetCNN_Cifar100, net_layer, 2); break;
				}
			}
			else
				LOG("Need three parameters");
			break;
		case 2:
			if (PNeuralNetCNN_Cifar10 != NULL && (strcmp(net_name, NET_CIFAR10_NAME) == 0))
			{
				switch (net_io)
				{
				case 0:	PNeuralNetCNN_Cifar10->printGradients(PNeuralNetCNN_Cifar10, net_layer, 0); break;
				case 1:	PNeuralNetCNN_Cifar10->printGradients(PNeuralNetCNN_Cifar10, net_layer, 1); break;
				case 2:	PNeuralNetCNN_Cifar10->printGradients(PNeuralNetCNN_Cifar10, net_layer, 2); break;
				}
			}
			else if (PNeuralNetCNN_Cifar100 != NULL && (strcmp(net_name, NET_CIFAR100_NAME) == 0))
			{
				switch (net_io)
				{
				case 0:	PNeuralNetCNN_Cifar100->printGradients(PNeuralNetCNN_Cifar100, net_layer, 0); break;
				case 1:	PNeuralNetCNN_Cifar100->printGradients(PNeuralNetCNN_Cifar100, net_layer, 1); break;
				case 2:	PNeuralNetCNN_Cifar100->printGradients(PNeuralNetCNN_Cifar100, net_layer, 2); break;
				}
			}
			else
				LOG("Need three parameters");
			break;
		case 3:
			PNeuralNetCNN_Cifar10->printNetLayersInfor(PNeuralNetCNN_Cifar10);
			PNeuralNetCNN_Cifar10->printNetLayersInfor(PNeuralNetCNN_Cifar100);
			break;

		case 4:
			LOGINFOR("NeuralNet start trainning...");
			PNeuralNetCNN_Cifar10->trainning.trainingSaving = false;
			PNeuralNetCNN_Cifar10->trainning.one_by_one = true;
			PNeuralNetCNN_Cifar10->trainning.batch_by_batch = false;
			PNeuralNetCNN_Cifar10->trainning.trainningGoing = true;
			NeuralNetStartTrainning(PNeuralNetCNN_Cifar10);
			break;
		case 5:
			LOGINFOR("NeuralNet start trainning...");
			PNeuralNetCNN_Cifar10->trainning.trainingSaving = false;
			PNeuralNetCNN_Cifar10->trainning.one_by_one = false;
			PNeuralNetCNN_Cifar10->trainning.batch_by_batch = true;
			PNeuralNetCNN_Cifar10->trainning.trainningGoing = true;
			NeuralNetStartTrainning(PNeuralNetCNN_Cifar10);
			break;
		case 6:
			LOGINFOR("NeuralNet start trainning...");
			PNeuralNetCNN_Cifar10->trainning.trainingSaving = false;
			PNeuralNetCNN_Cifar10->trainning.batch_by_batch = false;
			PNeuralNetCNN_Cifar10->trainning.one_by_one = false;
			PNeuralNetCNN_Cifar10->trainning.trainningGoing = true;
			NeuralNetStartTrainning(PNeuralNetCNN_Cifar10);
			break;
		case 7:
			LOGINFOR("NeuralNet start trainning...");
			PNeuralNetCNN_Cifar10->trainning.trainingSaving = true;
			PNeuralNetCNN_Cifar10->trainning.one_by_one = false;
			PNeuralNetCNN_Cifar10->trainning.batch_by_batch = false;
			PNeuralNetCNN_Cifar10->trainning.trainningGoing = true;
			NeuralNetStartTrainning(PNeuralNetCNN_Cifar10);
			break;

		case 8:
			LOGINFOR("NeuralNet start trainning...");
			PNeuralNetCNN_Cifar100->trainning.data_type = Cifar100;
			PNeuralNetCNN_Cifar100->trainning.trainingSaving = false;
			PNeuralNetCNN_Cifar100->trainning.one_by_one = true;
			PNeuralNetCNN_Cifar100->trainning.batch_by_batch = false;
			PNeuralNetCNN_Cifar100->trainning.trainningGoing = true;
			NeuralNetStartTrainning(PNeuralNetCNN_Cifar100);
			break;
		case 9:
			LOGINFOR("NeuralNet start trainning...");
			PNeuralNetCNN_Cifar100->trainning.data_type = Cifar100;
			PNeuralNetCNN_Cifar100->trainning.trainingSaving = false;
			PNeuralNetCNN_Cifar100->trainning.one_by_one = false;
			PNeuralNetCNN_Cifar100->trainning.batch_by_batch = true;
			PNeuralNetCNN_Cifar100->trainning.trainningGoing = true;
			NeuralNetStartTrainning(PNeuralNetCNN_Cifar100);
			break;
		case 10:
			LOGINFOR("NeuralNet start trainning...");
			PNeuralNetCNN_Cifar100->trainning.data_type = Cifar100;
			PNeuralNetCNN_Cifar100->trainning.trainingSaving = false;
			PNeuralNetCNN_Cifar100->trainning.batch_by_batch = false;
			PNeuralNetCNN_Cifar100->trainning.one_by_one = false;
			PNeuralNetCNN_Cifar100->trainning.trainningGoing = true;
			NeuralNetStartTrainning(PNeuralNetCNN_Cifar100);
			break;
		case 11:
			LOGINFOR("NeuralNet start trainning...");
			PNeuralNetCNN_Cifar100->trainning.data_type = Cifar100;
			PNeuralNetCNN_Cifar100->trainning.trainingSaving = true;
			PNeuralNetCNN_Cifar100->trainning.one_by_one = false;
			PNeuralNetCNN_Cifar100->trainning.batch_by_batch = false;
			PNeuralNetCNN_Cifar100->trainning.trainningGoing = true;
			NeuralNetStartTrainning(PNeuralNetCNN_Cifar100);
			break;

		case 12:
			LOGINFOR("NeuralNet start trainning...");
			PNeuralNetCNN_Cifar10->trainning.trainingSaving = true;
			PNeuralNetCNN_Cifar10->trainning.one_by_one = false;
			PNeuralNetCNN_Cifar10->trainning.batch_by_batch = true;

			PNeuralNetCNN_Cifar100->trainning.trainingSaving = true;
			PNeuralNetCNN_Cifar100->trainning.one_by_one = false;
			PNeuralNetCNN_Cifar100->trainning.batch_by_batch = true;

			while (true) // 同时训练两个网络
			{
				NeuralNetStartTrainning(PNeuralNetCNN_Cifar10);

				NeuralNetStartTrainning(PNeuralNetCNN_Cifar100);
			}
			break;
		case 13:
			LOGINFOR("NeuralNet saving...");
			PNeuralNetCNN_Cifar100->saveWeights(PNeuralNetCNN_Cifar10);
			PNeuralNetCNN_Cifar100->saveWeights(PNeuralNetCNN_Cifar100);
			break;
		case 14:
			LOGINFOR("NeuralNet loading...");
			PNeuralNetCNN_Cifar100->loadWeights(PNeuralNetCNN_Cifar10);
			PNeuralNetCNN_Cifar100->loadWeights(PNeuralNetCNN_Cifar100);
			break;
		case 15:
			TPPicture pic = Dataset_GetTestingPic(net_layer, Cifar10);
			TPVolume picv = pic->volume;
			//TPVolume picv = LoadBmpFileToVolume("test_fruit_and_vegetables_apple_9.bmp");
			if (picv != NULL)
			{
				SaveVolumeToBMP(picv, false, -1, 32, "test_32.bmp");
				picv->flip(picv);
				SaveVolumeToBMP(picv, false, -1, 32, "test_32-flip.bmp");
			}
			break;
		case 16:
			PrintBMP("test_airplane_airplane_3.bmp");
				break;
		case 17:
			CreateBMP("createBMP_24.bmp",32,32,24);
			CreateBMP("createBMP_32.bmp", 32, 32, 32);
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
	//	LOGINFOR("%s\n", pwd_path);
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