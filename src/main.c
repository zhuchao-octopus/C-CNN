/////////////////////////////////////////////////////////////////////////////////////////////
/*
 *  main.c test demo 定义应用程序的入口点
 *  Home Page :http://www.1234998.top
 *  Created on: May 20, 2023
 *  Author: M
 */
/////////////////////////////////////////////////////////////////////////////////////////////
// 要运行本程序进行训练，需要下载C语言版本Cifar-10 Cifar-100 数据集
// 下载地址：http://www.cs.toronto.edu/~kriz/cifar.html
// 下载解压后放在工程目录下面
/////////////////////////////////////////////////////////////////////////////////////////////

#include "ann-cnn.h"
#include "ann-dataset.h"
#include "ann-configuration.h"


#define NET_CIFAR10_NAME "Cifar10"
#define NET_CIFAR100_NAME "Cifar100"

//声明引用ann-configuragion.c 的连个网络
extern TPNeuralNet PNeuralNetCNN_Cifar10;
extern TPNeuralNet PNeuralNetCNN_Cifar100;

//再声明两个学习网络，深度分别为9和16
TPNeuralNet PNeuralNetCNN_9;
TPNeuralNet PNeuralNetCNN_16;

void NeuralNetStartTrainning(TPNeuralNet PNeuralNetCNN);
//void NeuralNetInitLeaningParameter(TPNeuralNet PNeuralNetCNN);
//void NeuralNetPrintNetInformation(TPNeuralNet PNeuralNetCNN);


void showBanner(void);

/////////////////////////////////////////////////////////////////////////////////////////////

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
    // 设置随机数种子
    srand(time(NULL));
    //一个4层简易网络用来学习Cifar10数据集
    LOG("InitNeuralNet_CNN Cifar10...\n");
    NeuralNetCreateAndInit_Cifar10();
    PNeuralNetCNN_Cifar10->trainning.data_type = Cifar10; //学习Cifar10数据集

    //一个4层简易网络用来学习Cifar100数据集
    printf("\n");
    LOG("\nInitNeuralNet_CNN Cifar100...\n");
    NeuralNetCreateAndInit_Cifar100();
    PNeuralNetCNN_Cifar100->trainning.data_type = Cifar100;  //学习Cifar100数据集

    //一个9层网络用来学习Cifar10数据集
    printf("\n");
    LOG("\nNeuralNetInit_C_CNN_9...\n");
    PNeuralNetCNN_9 = NeuralNetInit_C_CNN_9("C_CNN_9_Cifar10");
    PNeuralNetCNN_9->trainning.data_type = Cifar10; //学习Cifar10数据集


    //一个更深层网络用来学习Cifar100数据集，类似VGG16
    printf("\n");
    //LOG("\nNeuralNetInit_C_CNN_16...\n");//需要1G的内存
    //PNeuralNetCNN_16 = NeuralNetInit_C_CNN_16("C_CNN_16");
    //PNeuralNetCNN_16->trainning.data_type = Cifar100;  //学习Cifar100数据集

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
        printf("15: Start trainning PNeuralNetCNN_9\n");
        //printf("16: Start trainning PNeuralNetCNN_16\n");
        printf("\nplease select a menu item to continue:");

        gets(cmd_str);
        sscanf(cmd_str, "%d %d %d %s", &net_cmd, &net_layer, &net_io, net_name);
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
                case 0:
                    PNeuralNetCNN_Cifar10->printWeights(PNeuralNetCNN_Cifar10, net_layer, 0);
                    break;
                case 1:
                    PNeuralNetCNN_Cifar10->printWeights(PNeuralNetCNN_Cifar10, net_layer, 1);
                    break;
                case 2:
                    PNeuralNetCNN_Cifar10->printWeights(PNeuralNetCNN_Cifar10, net_layer, 2);
                    break;
                }
            }
            else if (PNeuralNetCNN_Cifar100 != NULL && (strcmp(net_name, NET_CIFAR100_NAME) == 0))
            {
                switch (net_io)
                {
                case 0:
                    PNeuralNetCNN_Cifar100->printWeights(PNeuralNetCNN_Cifar100, net_layer, 0);
                    break;
                case 1:
                    PNeuralNetCNN_Cifar100->printWeights(PNeuralNetCNN_Cifar100, net_layer, 1);
                    break;
                case 2:
                    PNeuralNetCNN_Cifar100->printWeights(PNeuralNetCNN_Cifar100, net_layer, 2);
                    break;
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
                case 0:
                    PNeuralNetCNN_Cifar10->printGradients(PNeuralNetCNN_Cifar10, net_layer, 0);
                    break;
                case 1:
                    PNeuralNetCNN_Cifar10->printGradients(PNeuralNetCNN_Cifar10, net_layer, 1);
                    break;
                case 2:
                    PNeuralNetCNN_Cifar10->printGradients(PNeuralNetCNN_Cifar10, net_layer, 2);
                    break;
                }
            }
            else if (PNeuralNetCNN_Cifar100 != NULL && (strcmp(net_name, NET_CIFAR100_NAME) == 0))
            {
                switch (net_io)
                {
                case 0:
                    PNeuralNetCNN_Cifar100->printGradients(PNeuralNetCNN_Cifar100, net_layer, 0);
                    break;
                case 1:
                    PNeuralNetCNN_Cifar100->printGradients(PNeuralNetCNN_Cifar100, net_layer, 1);
                    break;
                case 2:
                    PNeuralNetCNN_Cifar100->printGradients(PNeuralNetCNN_Cifar100, net_layer, 2);
                    break;
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
            PNeuralNetCNN_9->loadWeights(PNeuralNetCNN_9);
            //PNeuralNetCNN_16->loadWeights(PNeuralNetCNN_16);
            break;
        case 15:
            PNeuralNetCNN_9->trainning.trainingSaving = true;
            PNeuralNetCNN_9->trainning.one_by_one = false;
            PNeuralNetCNN_9->trainning.batch_by_batch = false;
            PNeuralNetCNN_9->trainning.trainningGoing = true;
            //PNeuralNetCNN_9->loadWeights(PNeuralNetCNN_9);
            PNeuralNetCNN_9->trainning.randomFlip = false;
            NeuralNetStartTrainning(PNeuralNetCNN_9);
            break;
        case 16:
            PNeuralNetCNN_16->trainning.trainingSaving = true;
            PNeuralNetCNN_16->trainning.one_by_one = false;
            PNeuralNetCNN_16->trainning.batch_by_batch = false;
            PNeuralNetCNN_16->trainning.trainningGoing = true;
            PNeuralNetCNN_16->loadWeights(PNeuralNetCNN_16);
            PNeuralNetCNN_16->trainning.randomFlip = true;
            NeuralNetStartTrainning(PNeuralNetCNN_16);
            break;


        case 20:
#if 0
            TPPicture pic = Dataset_GetTestingPic(net_layer, Cifar10);
            TPVolume picv = pic->volume;
            //TPVolume picv = LoadBmpFileToVolume("test_fruit_and_vegetables_apple_9.bmp");
            if (picv != NULL)
            {
                SaveVolumeToBMP(picv, false, -1, 32, "test_32.bmp");
                picv->flip(picv);
                SaveVolumeToBMP(picv, false, -1, 32, "test_32-flip.bmp");
            }
#endif
            break;
        case 21:
            PrintBMP("test_airplane_airplane_3.bmp");
            break;
        case 22:
            CreateBMP("createBMP_24.bmp", 32, 32, 24);
            CreateBMP("createBMP_32.bmp", 32, 32, 32);
            break;
        case 23:
            LOG("%f", GenerateRandomNumber());
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
    FILE* fp = fopen("../banner.txt", "r");

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

