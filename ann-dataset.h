#ifndef _INC_ANN_CIFAR_H_
#define _INC_ANN_CIFAR_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "ann-cnn.h"

//cifar-10 /cifar-100
#define CIFAR_TRAINNING_IMAGE_COUNT 50000
#define CIFAR_TESTING_IMAGE_COUNT 10000
#define CIFAR_TRAINNING_IMAGE_SAVINT_COUNT 1000

#define CIFAR_IMAGE_WIDTH 32
#define CIFAR_IMAGE_HEIGHT 32
#define CIFAR10_TRAINNING_IMAGE_BATCH_COUNT 10000
//#define CIFAR100_IMAGE_LABEL_NUM 100
#define CIFAR10_IMAGE_SIZE (3072 + 1) // 32X32X3+1
#define CIFAR100_IMAGE_SIZE (3072 + 2) // 32X32X3+2

 // 文件信息头结构体
typedef struct BITMAPFILEHEADER
{
    uint16_t bfType;        // 19778，必须是BM字符串，对应的十六进制为0x4d42，十进制为19778，否则不是BMP格式文件
    uint32_t bfSize;        // 文件大小，以字节为单位（2-5字节）
    uint16_t bfReserved1;   // 保留，必须设置为0（6-7字节）
    uint16_t bfReserved2;   // 保留，必须设置为0（8-9字节）
    uint32_t bfOffBits;     // 从文件头到像素数据的偏移（10-13字节）
} TBITMAPFILEHEADER;

// 图像信息头结构体
typedef struct BITMAPINFOHEADER
{
    uint32_t biSize;          // 此结构体的大小（14-17字节）
    uint32_t biWidth;         // 图像的宽（18-21字节）
    uint32_t biHeight;        // 图像的高（22-25字节）
    uint16_t biPlanes;        // 表示BMP图片的平面属，显然显示器只有一个平面，所以恒等于1（26-27字节）
    uint16_t biBitCount;      // 一像素所占的位数，一般为24（28-29字节）
    uint32_t biCompression;   // 说明图像数据压缩的类型，0为不压缩（30-33字节）
    uint32_t biSizeImage;     // 像素数据所占大小，这个值应该等于上面文件头结构中bfSize - bfOffBits（34-37字节）
    uint32_t biXPelsPerMeter; // 说明水平分辨率，用像素/米表示。一般为0（38-41字节）
    uint32_t biYPelsPerMeter; // 说明垂直分辨率，用像素/米表示。一般为0（42-45字节）
    uint32_t biClrUsed;       // 说明位图实际使用的彩色表中的颜色索引数（设为0的话，则说明使用所有调色板项）（46-49字节）
    uint32_t biClrImportant;  // 说明对图像显示有重要影响的颜色索引的数目，如果是0，表示都重要（50-53字节）
} TBITMAPINFOHEADER;

// 24位图像素信息结构体，即调色板
typedef struct PixelInfo {
    uint8_t rgbBlue;    // 该颜色的蓝色分量（值范围为0-255）
    uint8_t rgbGreen;   // 该颜色的绿色分量（值范围为0-255）
    uint8_t rgbRed;     // 该颜色的红色分量（值范围为0-255）
    uint8_t rgbReserved; // 保留，必须为0
} TPixelInfo;


typedef enum DataSetType
{
	Cifar10,
	Cifar100
} TDataSetType;

typedef struct ANN_CNN_DataSet_Image
{
	TDataSetType data_type;
	uint16_t labelIndex;
	uint16_t detailIndex;
	TPVolume volume;
} TDSImage, *TPPicture;

char *GetDataSetName(uint16_t DsType);

TPPicture Dataset_GetTestingPic(uint32_t TestingIndex, uint16_t DataSetType);
TPPicture Dataset_GetTrainningPic(uint32_t TrainningIndex, uint16_t DataSetType);
TPPicture Dataset_GetPic(FILE *PFile, uint32_t ImageIndex, uint16_t DataSetType);
uint32_t CifarReadImage(const char *FileName, uint8_t *Buffer, uint32_t ImageIndex);
uint32_t Cifar10ReadImage(FILE *PFile, uint8_t *Buffer, uint32_t ImageIndex);
uint32_t Cifar100ReadImage(FILE* PFile, uint8_t* Buffer, uint32_t ImageIndex);
uint32_t ReadFileToBuffer(const char* FileName, uint8_t* Buffer, uint32_t ReadSize, uint32_t OffSet);
uint32_t ReadFileToBuffer2(FILE *PFile, uint8_t *Buffer, uint32_t ReadSize, uint32_t OffSet);

void PrintBMP(char* BMPFileName);
FILE* CreateBMP(char* BMPFileName, uint16_t biBitCount, uint16_t width, uint16_t height);
TPVolume LoadBmpFileToVolume(const char* FileName);
void SaveVolumeToBMP(TPVolume PVolume, bool_t Grads, uint16_t Depth, uint16_t biBitCount, const char* FileName);
void CloseTestingDataset();
void CloseTrainningDataset();
#endif /* _INC_ANN_CNN_H_ */