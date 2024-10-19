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

#include "ann-cnn.h"          // 包含卷积神经网络的相关函数和结构定义
#include "ann-dataset.h"      // 包含数据集处理相关函数和结构定义
#include "ann-configuration.h" // 包含网络配置相关函数和结构定义

#define NET_CIFAR10_NAME "Cifar10"  // 定义CIFAR-10网络名称
#define NET_CIFAR100_NAME "Cifar100" // 定义CIFAR-100网络名称

// 声明引用：两个预先创建好的学习网络，在ann-configuration.c中定义
extern TPNeuralNet PNeuralNetCNN_Cifar10; // CIFAR-10神经网络的引用
extern TPNeuralNet PNeuralNetCNN_Cifar100; // CIFAR-100神经网络的引用

// 声明定义类型指针：用于动态创建学习网络，深度分别为9和16
TPNeuralNet PNeuralNetCNN_9;  // 指向深度为9的神经网络的指针
TPNeuralNet PNeuralNetCNN_16; // 指向深度为16的神经网络的指针

// 函数声明
void NeuralNetStartTrainning(TPNeuralNet PNeuralNetCNN); // 启动神经网络训练的函数
// void NeuralNetInitLeaningParameter(TPNeuralNet PNeuralNetCNN); // 初始化学习参数的函数（未使用）
// void NeuralNetPrintNetInformation(TPNeuralNet PNeuralNetCNN); // 打印网络信息的函数（未使用）

void showBanner(void); // 显示程序的欢迎横幅


/////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
    char cmd_str[32] = ""; // Buffer for user command input
    char net_name[32] = NET_CIFAR10_NAME; // Default network name for CIFAR-10
    int net_cmd = 0; // Command variable for user input
    int net_layer = 0; // Layer index for the neural network
    int net_io = 0; // 0: output, 1: input, 2: filters
    int log_lines = 0; // Number of log lines to display

    showBanner(); // Display the program banner

#if defined(__STDC_VERSION__)
    printf("__STDC_VERSION__ Version %ld\n", __STDC_VERSION__); // Print the C standard version
#endif
    printf("version:1.0.0.0\n"); // Print program version
    printf("////////////////////////////////////////////////////////////////////////////////////\n");

    // 设置随机数种子
    srand(time(NULL)); // Set random seed based on current time

    // Initialize a 4-layer simple network for learning the CIFAR-10 dataset
    LOG("InitNeuralNet_CNN Cifar10...\n");
    NeuralNetCreateAndInit_Cifar10(); // Create and initialize CIFAR-10 neural network
    PNeuralNetCNN_Cifar10->trainning.data_type = Cifar10; // Set data type for training to CIFAR-10

    // Initialize a 4-layer simple network for learning the CIFAR-100 dataset
    printf("\n");
    LOG("\nInitNeuralNet_CNN Cifar100...\n");
    NeuralNetCreateAndInit_Cifar100(); // Create and initialize CIFAR-100 neural network
    PNeuralNetCNN_Cifar100->trainning.data_type = Cifar100;  // Set data type for training to CIFAR-100

    // Initialize a 9-layer network for learning the CIFAR-10 dataset
    printf("\n");
    LOG("\nNeuralNetInit_C_CNN_9...\n");
    PNeuralNetCNN_9 = NeuralNetInit_C_CNN_9("C_CNN_9_Cifar10"); // Create a 9-layer network
    PNeuralNetCNN_9->trainning.data_type = Cifar10; // Set data type for training to CIFAR-10

    // Initialize a deeper network for learning the CIFAR-100 dataset, similar to VGG16
    printf("\n");
    // LOG("\nNeuralNetInit_C_CNN_16...\n"); // Uncomment if you need to initialize a 16-layer network
    // PNeuralNetCNN_16 = NeuralNetInit_C_CNN_16("C_CNN_16"); // Create a 16-layer network
    // PNeuralNetCNN_16->trainning.data_type = Cifar100;  // Set data type for training to CIFAR-100

    while (true) // Main program loop
    {
        printf("\033[%dB", log_lines); // Move cursor down by log_lines
        log_lines = 20; // Set log_lines for cursor movement
        printf("\nplease select a menu item to continue...\n"); // Prompt user for menu selection
        printf("00: Exit.\n");

        // Print weight usage menu
        printf("01: Print weight usage: Command Layer Index Type Name (e.g., 1 10 0 Cifar10). The first parameter is the command, the second is the layer index,\n");
        printf("    the third is the type (1: input, 0: output, 2: filters), and the fourth is the network name (Cifar10/Cifar100).\n");

        // Print gradient usage menu
        printf("02: Print gradient usage: 2 10 0 Cifar10. This format is the same as print weight.\n");

        // Print neural network information
        printf("03: Print neural network information: Displays the network architecture information.\n");

        // Start training commands for CIFAR-10
        printf("04: Start training CIFAR-10: Learn one CIFAR-10 image at a time.\n");
        printf("05: Start training CIFAR-10 by batch: Learn a batch of CIFAR-10 images at a time.\n");
        printf("06: Train on all 50,000 CIFAR-10 images but do not save weights to file.\n");
        printf("07: Train on all 50,000 CIFAR-10 images and save weights to file 'cnn.w'.\n");

        // Start training commands for CIFAR-100
        printf("08: Start training CIFAR-100: Learn one CIFAR-100 image at a time.\n");
        printf("09: Start training CIFAR-100 by batch: Learn a batch of CIFAR-100 images at a time.\n");
        printf("10: Train on all 50,000 CIFAR-100 images but do not save weights to file.\n");
        printf("11: Train on all 50,000 CIFAR-100 images and save weights to file 'cnn.w'.\n");

        printf("12: Start trainning CIFAR-100 and CIFAR-10 \n"); // Start training both datasets
        printf("13: Save weights to file cnn.w\n");
        printf("14: Load weights from file cnn.w\n");
        printf("15: Start trainning PNeuralNetCNN_9\n");
        // printf("16: Start trainning PNeuralNetCNN_16\n"); // Uncomment to enable training for the 16-layer network
        printf("\nplease select a menu item to continue:"); // Prompt user for input

        gets(cmd_str); // Get command input from the user
        sscanf(cmd_str, "%d %d %d %s", &net_cmd, &net_layer, &net_io, net_name); // Parse the command string
        printf("command:%d layer:%d io:%d name:%s\n", net_cmd, net_layer, net_io, net_name); // Print parsed command info


        switch (net_cmd)
        {
        case 0: // Close training and testing datasets
            CloseTrainningDataset(); // Close the training dataset
            CloseTestingDataset(); // Close the testing dataset
            return 0; // Exit the function

        case 1: // Print weights of the neural network
            if (PNeuralNetCNN_Cifar10 != NULL && (strcmp(net_name, NET_CIFAR10_NAME) == 0))
            {
                switch (net_io) // Check which weights to print
                {
                case 0: // Print output weights
                    PNeuralNetCNN_Cifar10->printWeights(PNeuralNetCNN_Cifar10, net_layer, 0);
                    break;
                case 1: // Print input weights
                    PNeuralNetCNN_Cifar10->printWeights(PNeuralNetCNN_Cifar10, net_layer, 1);
                    break;
                case 2: // Print filter weights
                    PNeuralNetCNN_Cifar10->printWeights(PNeuralNetCNN_Cifar10, net_layer, 2);
                    break;
                }
            }
            else if (PNeuralNetCNN_Cifar100 != NULL && (strcmp(net_name, NET_CIFAR100_NAME) == 0))
            {
                switch (net_io) // Check which weights to print for CIFAR-100
                {
                case 0: // Print output weights
                    PNeuralNetCNN_Cifar100->printWeights(PNeuralNetCNN_Cifar100, net_layer, 0);
                    break;
                case 1: // Print input weights
                    PNeuralNetCNN_Cifar100->printWeights(PNeuralNetCNN_Cifar100, net_layer, 1);
                    break;
                case 2: // Print filter weights
                    PNeuralNetCNN_Cifar100->printWeights(PNeuralNetCNN_Cifar100, net_layer, 2);
                    break;
                }
            }
            else
                LOG("Need three parameters"); // Log an error if the conditions are not met
            break;

        case 2: // Print gradients of the neural network
            if (PNeuralNetCNN_Cifar10 != NULL && (strcmp(net_name, NET_CIFAR10_NAME) == 0))
            {
                switch (net_io) // Check which gradients to print
                {
                case 0: // Print output gradients
                    PNeuralNetCNN_Cifar10->printGradients(PNeuralNetCNN_Cifar10, net_layer, 0);
                    break;
                case 1: // Print input gradients
                    PNeuralNetCNN_Cifar10->printGradients(PNeuralNetCNN_Cifar10, net_layer, 1);
                    break;
                case 2: // Print filter gradients
                    PNeuralNetCNN_Cifar10->printGradients(PNeuralNetCNN_Cifar10, net_layer, 2);
                    break;
                }
            }
            else if (PNeuralNetCNN_Cifar100 != NULL && (strcmp(net_name, NET_CIFAR100_NAME) == 0))
            {
                switch (net_io) // Check which gradients to print for CIFAR-100
                {
                case 0: // Print output gradients
                    PNeuralNetCNN_Cifar100->printGradients(PNeuralNetCNN_Cifar100, net_layer, 0);
                    break;
                case 1: // Print input gradients
                    PNeuralNetCNN_Cifar100->printGradients(PNeuralNetCNN_Cifar100, net_layer, 1);
                    break;
                case 2: // Print filter gradients
                    PNeuralNetCNN_Cifar100->printGradients(PNeuralNetCNN_Cifar100, net_layer, 2);
                    break;
                }
            }
            else
                LOG("Need three parameters"); // Log an error if the conditions are not met
            break;

        case 3: // Print network layer information
            PNeuralNetCNN_Cifar10->printNetLayersInfor(PNeuralNetCNN_Cifar10); // Print layers info for CIFAR-10
            PNeuralNetCNN_Cifar10->printNetLayersInfor(PNeuralNetCNN_Cifar100); // Print layers info for CIFAR-100
            break;

        case 4: // Start training CIFAR-10 (one image at a time)
            LOGINFOR("NeuralNet start trainning..."); // Log the training start
            PNeuralNetCNN_Cifar10->trainning.trainingSaving = false; // Disable training saving
            PNeuralNetCNN_Cifar10->trainning.one_by_one = true; // Enable one-by-one training
            PNeuralNetCNN_Cifar10->trainning.batch_by_batch = false; // Disable batch training
            PNeuralNetCNN_Cifar10->trainning.trainningGoing = true; // Set training state to active
            NeuralNetStartTrainning(PNeuralNetCNN_Cifar10); // Start the training process
            break;

        case 5: // Start training CIFAR-10 (batch)
            LOGINFOR("NeuralNet start trainning..."); // Log the training start
            PNeuralNetCNN_Cifar10->trainning.trainingSaving = false; // Disable training saving
            PNeuralNetCNN_Cifar10->trainning.one_by_one = false; // Disable one-by-one training
            PNeuralNetCNN_Cifar10->trainning.batch_by_batch = true; // Enable batch training
            PNeuralNetCNN_Cifar10->trainning.trainningGoing = true; // Set training state to active
            NeuralNetStartTrainning(PNeuralNetCNN_Cifar10); // Start the training process
            break;

        case 6: // Train on all CIFAR-10 images without saving
            LOGINFOR("NeuralNet start trainning..."); // Log the training start
            PNeuralNetCNN_Cifar10->trainning.trainingSaving = false; // Disable training saving
            PNeuralNetCNN_Cifar10->trainning.batch_by_batch = false; // Disable batch training
            PNeuralNetCNN_Cifar10->trainning.one_by_one = false; // Disable one-by-one training
            PNeuralNetCNN_Cifar10->trainning.trainningGoing = true; // Set training state to active
            NeuralNetStartTrainning(PNeuralNetCNN_Cifar10); // Start the training process
            break;

        case 7: // Train on all CIFAR-10 images and save weights
            LOGINFOR("NeuralNet start trainning..."); // Log the training start
            PNeuralNetCNN_Cifar10->trainning.trainingSaving = true; // Enable training saving
            PNeuralNetCNN_Cifar10->trainning.one_by_one = false; // Disable one-by-one training
            PNeuralNetCNN_Cifar10->trainning.batch_by_batch = false; // Disable batch training
            PNeuralNetCNN_Cifar10->trainning.trainningGoing = true; // Set training state to active
            NeuralNetStartTrainning(PNeuralNetCNN_Cifar10); // Start the training process
            break;

        case 8: // Start training CIFAR-100 (one image at a time)
            LOGINFOR("NeuralNet start trainning..."); // Log the training start
            PNeuralNetCNN_Cifar100->trainning.data_type = Cifar100; // Set data type to Cifar100
            PNeuralNetCNN_Cifar100->trainning.trainingSaving = false; // Disable training saving
            PNeuralNetCNN_Cifar100->trainning.one_by_one = true; // Enable one-by-one training
            PNeuralNetCNN_Cifar100->trainning.batch_by_batch = false; // Disable batch training
            PNeuralNetCNN_Cifar100->trainning.trainningGoing = true; // Set training state to active
            NeuralNetStartTrainning(PNeuralNetCNN_Cifar100); // Start the training process
            break;

        case 9: // Start training CIFAR-100 (batch)
            LOGINFOR("NeuralNet start trainning..."); // Log the training start
            PNeuralNetCNN_Cifar100->trainning.data_type = Cifar100; // Set data type to Cifar100
            PNeuralNetCNN_Cifar100->trainning.trainingSaving = false; // Disable training saving
            PNeuralNetCNN_Cifar100->trainning.one_by_one = false; // Disable one-by-one training
            PNeuralNetCNN_Cifar100->trainning.batch_by_batch = true; // Enable batch training
            PNeuralNetCNN_Cifar100->trainning.trainningGoing = true; // Set training state to active
            NeuralNetStartTrainning(PNeuralNetCNN_Cifar100); // Start the training process
            break;

        case 10: // Train on all CIFAR-100 images without saving
            LOGINFOR("NeuralNet start trainning..."); // Log the training start
            PNeuralNetCNN_Cifar100->trainning.data_type = Cifar100; // Set data type to Cifar100
            PNeuralNetCNN_Cifar100->trainning.trainingSaving = false; // Disable training saving
            PNeuralNetCNN_Cifar100->trainning.batch_by_batch = false; // Disable batch training
            PNeuralNetCNN_Cifar100->trainning.one_by_one = false; // Disable one-by-one training
            PNeuralNetCNN_Cifar100->trainning.trainningGoing = true; // Set training state to active
            NeuralNetStartTrainning(PNeuralNetCNN_Cifar100); // Start the training process
            break;

        case 11: // Train on all CIFAR-100 images and save weights
            LOGINFOR("NeuralNet start trainning..."); // Log the training start
            PNeuralNetCNN_Cifar100->trainning.data_type = Cifar100; // Set data type to Cifar100
            PNeuralNetCNN_Cifar100->trainning.trainingSaving = true; // Enable training saving
            PNeuralNetCNN_Cifar100->trainning.one_by_one = false; // Disable one-by-one training
            PNeuralNetCNN_Cifar100->trainning.batch_by_batch = false; // Disable batch training
            PNeuralNetCNN_Cifar100->trainning.trainningGoing = true; // Set training state to active
            NeuralNetStartTrainning(PNeuralNetCNN_Cifar100); // Start the training process
            break;

        default: // Handle unrecognized commands
            LOG("Unknown command"); // Log an error if the command is unknown
            break;
        }


        return 0;
    }
} //main

void showBanner(void)
{
    char pwd_path[100]; // Buffer to hold each line read from the banner file
    FILE* fp = fopen("../banner.txt", "r"); // Open the banner file for reading

    if (fp != NULL) // Check if the file was opened successfully
    {
        // Read lines from the file until the end is reached
        while (fgets(pwd_path, sizeof(pwd_path), fp) != NULL)
        {
            printf("%s\n", pwd_path); // Print each line read from the banner file
        }
    }

    if (fp != NULL) // Ensure the file pointer is valid before closing
        fclose(fp); // Close the file to free resources

    // Uncomment the following lines if you want to print the current working directory
    // if (getcwd(pwd_path, 512) != NULL)
    //     LOGINFOR("%s\n", pwd_path); // Log the current working directory
}


