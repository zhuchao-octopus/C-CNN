#ifndef _INC_ANN_CIFAR_H_
#define _INC_ANN_CIFAR_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "ann-cnn.h"

// cifar-10
#define CIFAR10_IMAGE_NUM_TOTAL 60000
#define CIFAR10_IMAGE_LABEL_NUM 20
#define CIFAR10_IMAGE_WIDTH 32
#define CIFAR10_IMAGE_HEIGHT 32
#define CIFAR10_IMAGE_SIZE (3072 + 1) // 32X32X3+1
#define CIFAR10_TESTING_IMAGE_COUNT 10000
#define CIFAR10_TRAINNING_IMAGE_BATCH_COUNT 10000
#define CIFAR10_TRAINNING_IMAGE_COUNT CIFAR10_TRAINNING_IMAGE_BATCH_COUNT * 5

typedef struct ANN_CNN_DataSet_Image
{
	TDataSetType data_type;
	// char *lableName;
	uint16_t labelIndex;
	uint16_t detailIndex;
	TPVolume volume;
} TDSImage, *TPPicture;

char *GetDataSetName(uint16_t DsType);

TPPicture Dataset_GetTestingPic(uint32_t TestingIndex, uint16_t DataSetType);
TPPicture Dataset_GetTrainningPic(uint32_t TrainningIndex, uint16_t DataSetType);
TPPicture Dataset_GetPic(FILE *PFile, uint32_t ImageIndex, uint16_t DataSetType);
uint32_t CifarReadImage(const char *FileName, uint8_t *Buffer, uint32_t ImageIndex);
uint32_t CifarReadImage2(FILE *PFile, uint8_t *Buffer, uint32_t ImageIndex);

uint32_t ReadFileToBuffer(const char* FileName, uint8_t* Buffer, uint32_t ReadSize, uint32_t OffSet);
uint32_t ReadFileToBuffer2(FILE *PFile, uint8_t *Buffer, uint32_t ReadSize, uint32_t OffSet);


#endif /* _INC_ANN_CNN_H_ */