/////////////////////////////////////////////////////////////////////////////////////////////
/*
 *  ann-dataset.c
 *  Home Page :http://www.1234998.top
 *  Created on: May 20, 2023
 *  Author: M
 */
/////////////////////////////////////////////////////////////////////////////////////////////

#ifdef PLATFORM_STM32
#include "usart.h"
#include "octopus.h"
#endif

#include "ann-dataset.h"

#define Cifar10FilePathName1 "../cifar-10-batches-bin\\data_batch_1.bin"
#define Cifar10FilePathName2 "../cifar-10-batches-bin\\data_batch_2.bin"
#define Cifar10FilePathName3 "../cifar-10-batches-bin\\data_batch_3.bin"
#define Cifar10FilePathName4 "../cifar-10-batches-bin\\data_batch_4.bin"
#define Cifar10FilePathName5 "../cifar-10-batches-bin\\data_batch_5.bin"
#define Cifar10FilePathName6 "../cifar-10-batches-bin\\test_batch.bin"

#define Cifar100FilePathName_test "../cifar-100-binary\\test.bin"
#define Cifar100FilePathName_train "../cifar-100-binary\\train.bin"
///////////////////////////////////////////////////////////////////////////////////
uint8_t CifarBuffer[CIFAR10_IMAGE_SIZE + 1];
const char LabelNameList[][10] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
char DataSetName[][10] = {"cifar-10", "cifar-10", ""};
FILE *PCifarFile_Trainning = NULL;
FILE *PCifarFile_Testing = NULL;

void CloseTrainningDataset()
{
    if (PCifarFile_Trainning != NULL)
        fclose(PCifarFile_Trainning);
    PCifarFile_Trainning = NULL;
}
void CloseTestingDataset()
{
    if (PCifarFile_Testing != NULL)
        fclose(PCifarFile_Testing);
    PCifarFile_Testing = NULL;
}
char *GetDataSetName(uint16_t DsType)
{
    return DataSetName[DsType];
}

/// @brief ////////////////////////////////////////////////////////////////////////
/// @param TestingIndex
/// @param DataSetType
/// @return
TPPicture Dataset_GetTestingPic(uint32_t TestingIndex, uint16_t DataSetType)
{
    if (PCifarFile_Testing == NULL)
    {
        if (DataSetType == Cifar10)
        {
            PCifarFile_Testing = fopen(Cifar10FilePathName6, "rb");
            LOGINFOR("load testing set from %s", Cifar10FilePathName6);
        }
        else if (DataSetType == Cifar100)
        {
            PCifarFile_Testing = fopen(Cifar100FilePathName_test, "rb");
            LOGINFOR("load testing set from %s", Cifar100FilePathName_test);
        }
    }
    if (PCifarFile_Testing != NULL)
        return Dataset_GetPic(PCifarFile_Testing, TestingIndex, DataSetType);
    else
        return NULL;
}

/// @brief ///////////////////////////////////////////////////////////////////////
/// @param TrainningIndex
/// @param DataSetType
/// @return
TPPicture Dataset_GetTrainningPic(uint32_t TrainningIndex, uint16_t DataSetType)
{
    uint32_t image_index = TrainningIndex;
    if (DataSetType == Cifar10)
    {
        if (TrainningIndex >= CIFAR_TRAINNING_IMAGE_COUNT || TrainningIndex <= 0)
        {
            CloseTrainningDataset();
            PCifarFile_Trainning = fopen(Cifar10FilePathName1, "rb");
            LOGINFOR("load trainning set from %s", Cifar10FilePathName1);
        }
        else if (TrainningIndex == CIFAR10_TRAINNING_IMAGE_BATCH_COUNT)
        {
            CloseTrainningDataset();
            PCifarFile_Trainning = fopen(Cifar10FilePathName2, "rb");
            LOGINFOR("load trainning set from %s", Cifar10FilePathName2);
        }
        else if (TrainningIndex == CIFAR10_TRAINNING_IMAGE_BATCH_COUNT * 2)
        {
            CloseTrainningDataset();
            PCifarFile_Trainning = fopen(Cifar10FilePathName3, "rb");
            LOGINFOR("load trainning set from %s", Cifar10FilePathName3);
        }
        else if (TrainningIndex == CIFAR10_TRAINNING_IMAGE_BATCH_COUNT * 3)
        {
            CloseTrainningDataset();
            PCifarFile_Trainning = fopen(Cifar10FilePathName4, "rb");
            LOGINFOR("load trainning set from %s", Cifar10FilePathName4);
        }
        else if (TrainningIndex == CIFAR10_TRAINNING_IMAGE_BATCH_COUNT * 4)
        {
            CloseTrainningDataset();
            PCifarFile_Trainning = fopen(Cifar10FilePathName5, "rb");
            LOGINFOR("load trainning set from %s", Cifar10FilePathName5);
        }
        image_index = TrainningIndex % CIFAR10_TRAINNING_IMAGE_BATCH_COUNT;
    }
    else if (DataSetType == Cifar100)
    {
        if (PCifarFile_Trainning == NULL)
        {
            PCifarFile_Trainning = fopen(Cifar100FilePathName_train, "rb");
            LOGINFOR("load trainning set from %s", Cifar100FilePathName_train);
        }
    }
    if (PCifarFile_Trainning != NULL)
        return Dataset_GetPic(PCifarFile_Trainning, image_index, DataSetType);
    else
        return NULL;
}
/// @brief /////////////////////////////////////////////////////////////////////////
/// @param PFile
/// @param ImageIndex
/// @param DataSetType
/// @return
TPPicture Dataset_GetPic(FILE *PFile, uint32_t ImageIndex, uint16_t DataSetType)
{
    uint32_t iSize = 0;
    TPPicture pPic = NULL;

    if (DataSetType == Cifar10)
    {
        iSize = Cifar10ReadImage(PFile, CifarBuffer, ImageIndex);
        if (iSize != CIFAR10_IMAGE_SIZE)
            return pPic;
        pPic = malloc(sizeof(TDSImage));
        pPic->data_type = Cifar10;
        pPic->labelIndex = CifarBuffer[0];
        pPic->detailIndex = pPic->labelIndex;
        pPic->volume = MakeVolume(CIFAR_IMAGE_WIDTH, CIFAR_IMAGE_HEIGHT, 3);
        pPic->volume->init(pPic->volume, CIFAR_IMAGE_WIDTH, CIFAR_IMAGE_HEIGHT, 3, 0);
        for (uint16_t y = 0; y < CIFAR_IMAGE_HEIGHT; y++)
        {
            for (uint16_t x = 0; x < CIFAR_IMAGE_WIDTH; x++)
            {
                //// 前1024个条目包含红色通道值，后1024个条目包含绿色通道值，最后1024个条目包含蓝色通道值。
                uint8_t r = CifarBuffer[y * CIFAR_IMAGE_WIDTH + x + 1];
                uint8_t g = CifarBuffer[y * CIFAR_IMAGE_WIDTH + CIFAR_IMAGE_WIDTH * CIFAR_IMAGE_HEIGHT + x + 1];
                uint8_t b = CifarBuffer[y * CIFAR_IMAGE_WIDTH + CIFAR_IMAGE_WIDTH * CIFAR_IMAGE_HEIGHT * 2 + x + 1];
                float32_t fr = r / 255.0 - 0.5; // 对数据进行归一化预处理平移图像
                float32_t fg = g / 255.0 - 0.5;
                float32_t fb = b / 255.0 - 0.5;
                pPic->volume->setValue(pPic->volume, x, y, 0, fr);
                pPic->volume->setValue(pPic->volume, x, y, 1, fg);
                pPic->volume->setValue(pPic->volume, x, y, 2, fb);
            }
        }
    }

    else if (DataSetType == Cifar100)
    {
        iSize = Cifar100ReadImage(PFile, CifarBuffer, ImageIndex);
        if (iSize != CIFAR100_IMAGE_SIZE)
            return pPic;
        pPic = malloc(sizeof(TDSImage));
        // if (pPic == NULL) retrun NULL;
        pPic->data_type = Cifar100;
        pPic->labelIndex = CifarBuffer[0];
        pPic->detailIndex = CifarBuffer[1];
        pPic->volume = MakeVolume(CIFAR_IMAGE_WIDTH, CIFAR_IMAGE_HEIGHT, 3);
        pPic->volume->init(pPic->volume, CIFAR_IMAGE_WIDTH, CIFAR_IMAGE_HEIGHT, 3, 0);
        for (uint16_t y = 0; y < CIFAR_IMAGE_HEIGHT; y++)
        {
            for (uint16_t x = 0; x < CIFAR_IMAGE_WIDTH; x++)
            {
                uint8_t r = CifarBuffer[y * CIFAR_IMAGE_WIDTH + 0000 + x + 2];
                uint8_t g = CifarBuffer[y * CIFAR_IMAGE_WIDTH + 1024 + x + 2];
                uint8_t b = CifarBuffer[y * CIFAR_IMAGE_WIDTH + 2048 + x + 2];
                float32_t fr = r / 255.0;
                float32_t fg = g / 255.0;
                float32_t fb = b / 255.0;
                pPic->volume->setValue(pPic->volume, x, y, 0, fr - 0.5);
                pPic->volume->setValue(pPic->volume, x, y, 1, fg - 0.5);
                pPic->volume->setValue(pPic->volume, x, y, 2, fb - 0.5);
            }
        }
    }
    else
    {
        // LOGINFOR("Read data failed from %s TrainningIndex=%d DataSetType = %d\n", PFile->_tmpfname, ImageIndex, DataSetType);
    }
    return pPic;
}

/// @brief ///////////////////////////////////////////////////////////////////////
/// @param FileName
/// @param Buffer
/// @param ImageIndex
/// @return
uint32_t CifarReadImage(const char *FileName, uint8_t *Buffer, uint32_t ImageIndex)
{
    uint32_t offset = CIFAR10_IMAGE_SIZE * ImageIndex;
    return ReadFileToBuffer(FileName, Buffer, CIFAR10_IMAGE_SIZE, offset);
}
/// @brief ////////////////////////////////////////////////////////////////////////
/// @param PFile
/// @param Buffer
/// @param ImageIndex
/// @return
uint32_t Cifar10ReadImage(FILE *PFile, uint8_t *Buffer, uint32_t ImageIndex)
{
    uint32_t offset = CIFAR10_IMAGE_SIZE * ImageIndex;
    return ReadFileToBuffer2(PFile, Buffer, CIFAR10_IMAGE_SIZE, offset);
}
uint32_t Cifar100ReadImage(FILE *PFile, uint8_t *Buffer, uint32_t ImageIndex)
{
    uint32_t offset = CIFAR100_IMAGE_SIZE * ImageIndex;
    return ReadFileToBuffer2(PFile, Buffer, CIFAR100_IMAGE_SIZE, offset);
}
/// @brief ////////////////////////////////////////////////////////////////////////
/// @param FileName
/// @param Buffer
/// @param ReadSize
/// @return
uint32_t ReadFileToBuffer(const char *FileName, uint8_t *Buffer, uint32_t ReadSize, uint32_t OffSet)
{
    FILE *pFile = fopen(FileName, "rb");
    uint32_t readLength = 0;
    uint32_t fileSize = 0;

    if (pFile != NULL)
    {
        fseek(pFile, 0, SEEK_END);
        fileSize = ftell(pFile);
        if (OffSet + ReadSize > fileSize)
        {
            LOG("out of file size %d > %d", OffSet + ReadSize, fileSize);
            return 0;
        }
        fseek(pFile, OffSet, SEEK_SET);
        memset(Buffer, 0, ReadSize);
        while (!feof(pFile))
        {
            uint32_t len = fread(Buffer, sizeof(uint8_t), ReadSize, pFile);
            // LOG("\nBuffer: %d, len: %d \n", count, len);
            // for (uint32_t i = 0; i < len; i++)
            //{
            //	LOG("%02x,", Buffer[i]);
            //}
            readLength = readLength + len;
            break;
        }
    }
    else
    {
        LOGERROR("Error opening file\n");
    }
    fclose(pFile);
    // LOG("\nreadLength =  %d\n", readLength);
    return (readLength);
}
/// @brief //////////////////////////////////////////////////////////////////////////
/// @param PFile
/// @param Buffer
/// @param ReadSize
/// @param OffSet
/// @return
uint32_t ReadFileToBuffer2(FILE *PFile, uint8_t *Buffer, uint32_t ReadSize, uint32_t OffSet)
{
    uint32_t readLength = 0;
    uint32_t fileSize = 0;

    if (PFile != NULL)
    {
        fseek(PFile, 0, SEEK_END);
        fileSize = ftell(PFile);
        if (OffSet + ReadSize > fileSize)
        {
            LOG("out of file size %d > %d", OffSet + ReadSize, fileSize);
            return 0;
        }
        fseek(PFile, OffSet, SEEK_SET);
        memset(Buffer, 0, ReadSize);
        while (!feof(PFile))
        {
            uint32_t len = fread(Buffer, sizeof(uint8_t), ReadSize, PFile);
            // LOG("\nBuffer: %d, len: %d \n", count, len);
            // for (uint32_t i = 0; i < len; i++)
            //{
            //	LOG("%02x,", Buffer[i]);
            //}
            readLength = readLength + len;
            break;
        }
    }
    else
    {
        LOGERROR("Error file not open!\n");
    }
    // LOG("\nreadLength =  %d\n", readLength);
    return (readLength);
}
/// @brief //////////////////////////////////////////////////////////////////
/// @param FileName
/// @param Buffer
/// @param WriteSize
/// @return /
uint32_t WriteBufferToFile(const char *FileName, float32_t *Buffer, uint32_t WriteSize)
{
    FILE *pFile = fopen(FileName, "wb");
    if (pFile == NULL)
    {
        LOG("Error opening file");
        return 0;
    }
    fwrite(Buffer, sizeof(float32_t), WriteSize, pFile);
    fclose(pFile);
    return 0;
}

// 从BMP文件加载数据到Volume结构体
TPVolume LoadBmpFileToVolume(const char *FileName)
{
    FILE *pFile = fopen(FileName, "rb");
    TPVolume tPVolume = NULL;
    BITMAPFILEHEADER fileHeader; // 定义一个 BMP 文件头的结构体
    BITMAPINFOHEADER infoHeader; // 定义一个 BMP 文件信息结构体
    if (pFile == NULL)
    {
        printf("Error opening file: %s\n", FileName);
        return NULL;
    }
    fseek(pFile, 0, SEEK_SET);
    fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, pFile);
    fread(&infoHeader, sizeof(BITMAPINFOHEADER), 1, pFile);
    // 检查文件类型是否为BMP
    if (fileHeader.bfType != 0x4D42 || (infoHeader.biBitCount != 24 && infoHeader.biBitCount != 32))
    {
        printf("Error: Invalid BMP file or unsupported format \n");
        fclose(pFile);
        return NULL;
    }

    int rowSize = (infoHeader.biWidth * infoHeader.biBitCount + 31) / 32 * 4;
    unsigned char *pixelData = (unsigned char *)malloc(rowSize * infoHeader.biHeight);
    fseek(pFile, fileHeader.bfOffBits, SEEK_SET);
    fread(pixelData, 1, rowSize * infoHeader.biHeight, pFile);

    tPVolume = MakeVolume(CIFAR_IMAGE_WIDTH, CIFAR_IMAGE_HEIGHT, 3);
    tPVolume->init(tPVolume, CIFAR_IMAGE_WIDTH, CIFAR_IMAGE_HEIGHT, 3, 0);
    // 创建Volume结构体并分配内存
    tPVolume->_w = infoHeader.biWidth;
    tPVolume->_h = infoHeader.biHeight;
    tPVolume->_depth = 3; // 两个张量：权重和梯度

    // 将像素数据转换为Volume中的权重和梯度张量数据
    for (uint16_t y = 0; y < tPVolume->_h; y++)
    {
        for (uint16_t x = 0; x < tPVolume->_w; x++)
        {
            uint16_t offset_x = rowSize * y;
            uint16_t offset_y = tPVolume->_h - y - 1; // bottom-up DIB,开始行存储在最下面
            // 0:r,1:g,2:b
            if (infoHeader.biBitCount == 32)
            {
                tPVolume->setValue(tPVolume, x, offset_y, 0, pixelData[offset_x + x * 4 + 2] / 255.0); // R
                tPVolume->setValue(tPVolume, x, offset_y, 1, pixelData[offset_x + x * 4 + 1] / 255.0); // G
                tPVolume->setValue(tPVolume, x, offset_y, 2, pixelData[offset_x + x * 4 + 0] / 255.0); // B
            }
            else if (infoHeader.biBitCount == 24)
            {
                tPVolume->setValue(tPVolume, x, offset_y, 0, pixelData[offset_x + x * 3 + 2] / 255.0); // R
                tPVolume->setValue(tPVolume, x, offset_y, 1, pixelData[offset_x + x * 3 + 1] / 255.0); // G
                tPVolume->setValue(tPVolume, x, offset_y, 2, pixelData[offset_x + x * 3 + 0] / 255.0); // B
            }
        }
    }

    // 释放临时像素数据内存
    free(pixelData);
    fclose(pFile);
    return tPVolume;
}

// Min - Max Normalization 最小 - 最大值标准化：
// 也称为离差标准化，是对原始数据的线性变换，使结果值映射到[0 - 1] 之间。
// 实现方法是将变量值减去最小值并除以最大值和最小值的差。
// 将Volume结构体中的张量保存为BMP文件
void SaveVolumeToBMP(TPVolume PVolume, bool_t Grads, uint16_t Depth, uint16_t biBitCount, const char *FileName)
{
    uint16_t width = PVolume->_w;
    uint16_t height = PVolume->_h;
    float32_t rgb_v = 0;
    TPMaxMin pMaxMin = NULL;
    BITMAPFILEHEADER fileHeader; // 定义一个 BMP 文件头的结构体
    BITMAPINFOHEADER infoHeader; // 定义一个 BMP 文件信息结构体
    // 创建文件并写入文件头信息
    FILE *pFile = CreateBMP(FileName, biBitCount, width, height);

    fseek(pFile, 0, SEEK_SET);
    fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, pFile);
    fread(&infoHeader, sizeof(BITMAPINFOHEADER), 1, pFile);

    int rowSize = (infoHeader.biWidth * infoHeader.biBitCount + 31) / 32 * 4;
    unsigned char *pixelData = (unsigned char *)malloc(rowSize * infoHeader.biHeight);
    fseek(pFile, fileHeader.bfOffBits, SEEK_SET);

    if (Grads)
        pMaxMin = TensorMaxMin(PVolume->grads);
    else
        pMaxMin = TensorMaxMin(PVolume->weight);
    // 将像素数据转换为Volume中的权重和梯度张量数据
    if (Depth >= 0)
    {
        for (uint16_t y = 0; y < height; y++)
        {
            for (uint16_t x = 0; x < width; x++)
            {
                uint16_t offset_y = height - y - 1; // bottom-up DIB,开始行存储在最下面
                uint16_t offset_x = rowSize * offset_y;

                if (Grads)
                    rgb_v = PVolume->getGradValue(PVolume, x, y, Depth);
                else
                    rgb_v = PVolume->getValue(PVolume, x, y, Depth);

                rgb_v = (rgb_v - pMaxMin->min) / (pMaxMin->max - pMaxMin->min) * 255;
                rgb_v = rgb_v > 0 ? rgb_v : 0;
                rgb_v = rgb_v > 255 ? 255 : rgb_v;
                if (infoHeader.biBitCount == 32)
                {
                    pixelData[offset_x + x * 4 + 0] = rgb_v; // B
                    pixelData[offset_x + x * 4 + 1] = rgb_v; // G
                    pixelData[offset_x + x * 4 + 2] = rgb_v; // R
                    pixelData[offset_x + x * 4 + 3] = 255;   //
                }
                else if (infoHeader.biBitCount == 24)
                {
                    pixelData[offset_x + x * 3 + 0] = rgb_v; // B
                    pixelData[offset_x + x * 3 + 1] = rgb_v; // G
                    pixelData[offset_x + x * 3 + 2] = rgb_v; // R
                    // pixelData[offset_x + x * 3 + 3] = 0;     //
                }
            }
        } // for
    }
    else
    {
        for (uint16_t y = 0; y < height; y++)
        {
            for (uint16_t x = 0; x < width; x++)
            {
                uint16_t offset_y = height - y - 1; // bottom-up DIB,开始行存储在最下面
                uint16_t offset_x = rowSize * offset_y;
                float32_t rgb_r, rgb_g, rgb_b;
                if (Grads)
                {
                    rgb_r = PVolume->getGradValue(PVolume, x, y, 0);
                    rgb_g = PVolume->getGradValue(PVolume, x, y, 1);
                    rgb_b = PVolume->getGradValue(PVolume, x, y, 2);
                }
                else
                {
                    rgb_r = PVolume->getValue(PVolume, x, y, 0);
                    rgb_g = PVolume->getValue(PVolume, x, y, 1);
                    rgb_b = PVolume->getValue(PVolume, x, y, 2);
                }
                rgb_r = (rgb_r - pMaxMin->min) / (pMaxMin->max - pMaxMin->min) * 255;
                rgb_r = rgb_r > 0 ? rgb_r : 0;
                rgb_r = rgb_r > 255 ? 255 : rgb_r;
                rgb_g = (rgb_g - pMaxMin->min) / (pMaxMin->max - pMaxMin->min) * 255;
                rgb_g = rgb_g > 0 ? rgb_g : 0;
                rgb_g = rgb_g > 255 ? 255 : rgb_g;
                rgb_b = (rgb_b - pMaxMin->min) / (pMaxMin->max - pMaxMin->min) * 255;
                rgb_b = rgb_b > 0 ? rgb_b : 0;
                rgb_b = rgb_b > 255 ? 255 : rgb_b;
                if (infoHeader.biBitCount == 32)
                {
                    pixelData[offset_x + x * 4 + 0] = rgb_b; // B
                    pixelData[offset_x + x * 4 + 1] = rgb_g; // G
                    pixelData[offset_x + x * 4 + 2] = rgb_r; // R
                    pixelData[offset_x + x * 4 + 3] = 255;   //
                }
                else if (infoHeader.biBitCount == 24)
                {
                    pixelData[offset_x + x * 3 + 0] = rgb_b; // B
                    pixelData[offset_x + x * 3 + 1] = rgb_g; // G
                    pixelData[offset_x + x * 3 + 2] = rgb_r; // R
                    // pixelData[offset_x + x * 3 + 3] = 0;     //
                }
            }
        } // for
    }

    // 写入像素数据
    fwrite(pixelData, 1, infoHeader.biSizeImage, pFile);
    fflush(pFile);
    // 释放临时像素数据内存
    fclose(pFile);
    free(pixelData);
    free(pMaxMin);
}
void PrintBMP(char *BMPFileName)
{
    BITMAPFILEHEADER fileHeader; // 定义一个 BMP 文件头的结构体
    BITMAPINFOHEADER infoHeader; // 定义一个 BMP 文件信息结构体
    FILE *pFile;                 // 定义一个文件指针

    if ((pFile = fopen(BMPFileName, "rb")) == NULL) // fp = 0x00426aa0
    {
        printf("Cann't open the file!\n");
        return 0;
    }
    fseek(pFile, 0, SEEK_SET);
    fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, pFile);
    fread(&infoHeader, sizeof(BITMAPINFOHEADER), 1, pFile);

    printf("\nBMPFileName = %s\n", BMPFileName);
    printf("fileHeader.bfType = 0x%x\n", fileHeader.bfType);
    printf("fileHeader.bfSize = %d\n", fileHeader.bfSize);
    printf("fileHeader.bfReserved1 = %d\n", fileHeader.bfReserved1);
    printf("fileHeader.bfReserved2 = %d\n", fileHeader.bfReserved2);
    printf("fileHeader.bfOffBits = %d\n", fileHeader.bfOffBits);

    printf("infoHeader.biSize = %d \n", infoHeader.biSize);
    printf("infoHeader.biHeight = %d \n", infoHeader.biHeight);
    printf("infoHeader.biWidth = %d \n", infoHeader.biWidth);
    printf("infoHeader.biPlanes = %d \n", infoHeader.biPlanes);
    printf("infoHeader.biBitCount = %d \n", infoHeader.biBitCount);
    printf("infoHeader.biCompression = %d \n", infoHeader.biCompression);
    printf("infoHeader.biSizeImage = %d \n", infoHeader.biSizeImage);
    printf("infoHeader.biXPelsPerMeter = %d \n", infoHeader.biXPelsPerMeter);
    printf("infoHeader.biYPelsPerMeter = %d \n", infoHeader.biYPelsPerMeter);
    printf("infoHeader.biClrUsed = %d \n", infoHeader.biClrUsed);
    printf("infoHeader.biClrImportant = %d \n", infoHeader.biClrImportant);

    int rowSize = (infoHeader.biWidth * infoHeader.biBitCount + 31) / 32 * 4;
    unsigned char *pixelData = (unsigned char *)malloc(rowSize * infoHeader.biHeight);
    fseek(pFile, fileHeader.bfOffBits, SEEK_SET);
    fread(pixelData, 1, rowSize * infoHeader.biHeight, pFile);

    printf("\nPixel Data:\n");
    for (int y = 0; y < infoHeader.biHeight; y++)
    {
        printf("infoHeader.biHeight=%d\n", y);
        for (int x = 0; x < rowSize; x++)
        {
            printf("%02X", pixelData[y * rowSize + x]);
            if ((x + 1) % 32 == 0)
                printf("\n");
        }
    }

    /*
    // Seek to the beginning of pixel data
    fseek(pFile, fileHeader.bfOffBits, SEEK_SET);

    // Calculate the row size in bytes (including padding)
    int rowSize = (infoHeader.biWidth * infoHeader.biBitCount + 31) / 32 * 4;

    // Allocate memory for a row of pixels
    unsigned char* rowBuffer = (uint8_t*)malloc(rowSize);

    // Iterate over each row

    for (int y = 0; y < infoHeader.biHeight; y++)
    {
        // Read a row of pixels
        fread(rowBuffer, 1, rowSize, pFile);

        // Iterate over each pixel in the row
        for (int x = 0; x < infoHeader.biWidth; x++)
        {
            // Calculate the pixel index in the row buffer
            int pixelIndex = x * (infoHeader.biBitCount / 8);

            // Extract RGB values based on the pixel format
            unsigned char red, green, blue;
            if (infoHeader.biBitCount == 24)
            {
                blue = rowBuffer[pixelIndex];
                green = rowBuffer[pixelIndex + 1];
                red = rowBuffer[pixelIndex + 2];
            }
            else if (infoHeader.biBitCount == 32)
            {
                blue = rowBuffer[pixelIndex];
                green = rowBuffer[pixelIndex + 1];
                red = rowBuffer[pixelIndex + 2];
                // Skip the alpha channel
            }

            // Print the RGB values
            printf("Pixel (%d, %d): R = %d, G = %d, B = %d\n", x, y, red, green, blue);
        }
    }*/

    // Clean up
    free(pixelData);
    fclose(pFile);
}

FILE *CreateBMP(char *BMPFileName, uint16_t biBitCount, uint16_t width, uint16_t height)
{
    BITMAPFILEHEADER fileHeader;
    BITMAPINFOHEADER infoHeader;
    FILE *pFile;

    if ((pFile = fopen(BMPFileName, "wb+")) == NULL)
    {
        printf("Cannot create the file!\n");
        return NULL;
    }

    // 设置文件头
    fileHeader.bfType = 0x4D42; // BMP 文件标识符 "BM"
    fileHeader.bfSize = 0;      // 文件大小，暂时设置为 0
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER); // 数据偏移量

    // 设置信息头
    infoHeader.biSize = sizeof(BITMAPINFOHEADER);
    infoHeader.biWidth = width;         // 图像宽度
    infoHeader.biHeight = height;       // 图像高度，正值表示图像自上而下，负值表示自下而上
    infoHeader.biPlanes = 1;            // 平面数，固定为 1
    infoHeader.biBitCount = biBitCount; // 位数
    infoHeader.biCompression = 0;       // 压缩类型，0 表示不压缩
    infoHeader.biSizeImage = 0;         // 图像数据大小，暂时设置为 0
    infoHeader.biXPelsPerMeter = 0;     // 水平分辨率，暂时设置为 0
    infoHeader.biYPelsPerMeter = 0;     // 垂直分辨率，暂时设置为 0
    infoHeader.biClrUsed = 0;           // 颜色表中实际使用的颜色数，暂时设置为 0
    infoHeader.biClrImportant = 0;      // 对图像显示有重要影响的颜色数，暂时设置为 0

    // 计算图像数据的行大小
    int rowSize = (width * biBitCount + 31) / 32 * 4;

    // 计算图像数据大小
    int imageSize = rowSize * height;
    infoHeader.biSizeImage = imageSize;
    // 分配并填充图像数据（默认为黑色）
    unsigned char *imageData = (unsigned char *)calloc(imageSize, sizeof(unsigned char));

    // 写入文件头和信息头
    fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, pFile);
    fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, pFile);

    // 写入图像数据
    fwrite(imageData, sizeof(unsigned char), imageSize, pFile);

    fflush(pFile);
    // 释放图像数据内存
    free(imageData);
    // fclose(pFile);
    return pFile;
}