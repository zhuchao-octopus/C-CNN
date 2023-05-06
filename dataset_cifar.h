#ifndef _INC_ANN_CIFAR_H_
#define _INC_ANN_CIFAR_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "ann-cnn.h"

char *GetDataSetName(uint16_t DsType);

TPPicture Dataset_GetTestingPic(uint32_t TestingIndex, uint16_t DataSetType);
TPPicture Dataset_GetTrainningPic(uint32_t TrainningIndex, uint16_t DataSetType);
TPPicture Dataset_GetPic(FILE *PFile, uint32_t ImageIndex, uint16_t DataSetType);
uint32_t CifarReadImage(const char *FileName, uint8_t *Buffer, uint32_t ImageIndex);
uint32_t CifarReadImage2(FILE *PFile, uint8_t *Buffer, uint32_t ImageIndex);

uint32_t ReadFile2(FILE *PFile, uint8_t *Buffer, uint32_t ReadSize, uint32_t OffSet);
uint32_t ReadFileToBuffer(const char *FileName, uint8_t *Buffer, uint32_t ReadSize, uint32_t OffSet);

#endif /* _INC_ANN_CNN_H_ */