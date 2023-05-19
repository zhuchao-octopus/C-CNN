
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
FILE* PCifarFile_Testing = NULL;

void CloseDataset()
{
    if (PCifarFile_Trainning != NULL)
        fclose(PCifarFile_Trainning);
    if (PCifarFile_Testing != NULL)
        fclose(PCifarFile_Testing);
    PCifarFile_Trainning = NULL;
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
        }
        else  if (DataSetType == Cifar100)
        {
            PCifarFile_Testing = fopen(Cifar100FilePathName_test, "rb");
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
            image_index = TrainningIndex % CIFAR10_TRAINNING_IMAGE_BATCH_COUNT;
            if (PCifarFile_Trainning == NULL)
            {
                PCifarFile_Trainning = fopen(Cifar10FilePathName1, "rb");
            }
            else if (TrainningIndex == CIFAR10_TRAINNING_IMAGE_BATCH_COUNT)
            {
                CloseDataset();
                    PCifarFile_Trainning = fopen(Cifar10FilePathName2, "rb");
            }
            else if (TrainningIndex == CIFAR10_TRAINNING_IMAGE_BATCH_COUNT * 2)
            {
                CloseDataset();
                PCifarFile_Trainning = fopen(Cifar10FilePathName3, "rb");
            }
            else if (TrainningIndex == CIFAR10_TRAINNING_IMAGE_BATCH_COUNT * 3)
            {
                CloseDataset();
                PCifarFile_Trainning = fopen(Cifar10FilePathName4, "rb");
            }
            else if (TrainningIndex == CIFAR10_TRAINNING_IMAGE_BATCH_COUNT * 4)
            {
                CloseDataset();
                PCifarFile_Trainning = fopen(Cifar10FilePathName5, "rb");
            }
    }
    else if (DataSetType == Cifar100)
    {
        if (PCifarFile_Trainning == NULL)
        {
            PCifarFile_Trainning = fopen(Cifar100FilePathName_train, "rb");
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
    uint32_t iSize = CifarReadImage2(PFile, CifarBuffer, ImageIndex);
    TPPicture pPic = NULL;

    if ((DataSetType == Cifar10) && (iSize == CIFAR10_IMAGE_SIZE))
    {
        pPic = malloc(sizeof(TDSImage));
        pPic->data_type = Cifar10;
        pPic->labelIndex = CifarBuffer[0];
        pPic->detailIndex = pPic->labelIndex;
        pPic->volume = MakeVolume(CIFAR10_IMAGE_WIDTH, CIFAR10_IMAGE_HEIGHT, 3);
        pPic->volume->init(pPic->volume, CIFAR10_IMAGE_WIDTH, CIFAR10_IMAGE_HEIGHT, 3, 0);
        for (uint16_t y = 0; y < CIFAR10_IMAGE_HEIGHT; y++)
        {
            for (uint16_t x = 0; x < CIFAR10_IMAGE_WIDTH; x++)
            {
                //// 前1024个条目包含红色通道值，后1024个条目包含绿色通道值，最后1024个条目包含蓝色通道值。
                uint8_t r = CifarBuffer[y * CIFAR10_IMAGE_WIDTH + x + 1];
                uint8_t g = CifarBuffer[y * CIFAR10_IMAGE_WIDTH + CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT + x + 1];
                uint8_t b = CifarBuffer[y * CIFAR10_IMAGE_WIDTH + CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT * 2 + x + 1];
                float32_t fr = r / 255.0;
                float32_t fg = g / 255.0;
                float32_t fb = b / 255.0;
                pPic->volume->setValue(pPic->volume, x, y, 0, fr);
                pPic->volume->setValue(pPic->volume, x, y, 1, fg);
                pPic->volume->setValue(pPic->volume, x, y, 2, fb);
            }
        }
    }

    else if ((DataSetType == Cifar100) && (iSize == (CIFAR10_IMAGE_SIZE + 1)))
    {
        pPic = malloc(sizeof(TDSImage));
        pPic->data_type = Cifar100;
        pPic->labelIndex = CifarBuffer[0];
        pPic->detailIndex = CifarBuffer[1];
        pPic->volume = MakeVolume(CIFAR10_IMAGE_WIDTH, CIFAR10_IMAGE_HEIGHT, 3);
        pPic->volume->init(pPic->volume, CIFAR10_IMAGE_WIDTH, CIFAR10_IMAGE_HEIGHT, 3, 0);
        for (uint16_t y = 0; y < CIFAR10_IMAGE_HEIGHT; y++)
        {
            for (uint16_t x = 0; x < CIFAR10_IMAGE_WIDTH; x++)
            {
                uint8_t r = CifarBuffer[y * CIFAR10_IMAGE_WIDTH + 0000 + x + 2];
                uint8_t g = CifarBuffer[y * CIFAR10_IMAGE_WIDTH + 1024 + x + 2];
                uint8_t b = CifarBuffer[y * CIFAR10_IMAGE_WIDTH + 2048 + x + 2];
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
        // LOGINFO("Read data failed from %s TrainningIndex=%d DataSetType = %d\n", PFile->_tmpfname, ImageIndex, DataSetType);
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
uint32_t CifarReadImage2(FILE *PFile, uint8_t *Buffer, uint32_t ImageIndex)
{
    uint32_t offset = CIFAR10_IMAGE_SIZE * ImageIndex;
    return ReadFileToBuffer2(PFile, Buffer, CIFAR10_IMAGE_SIZE, offset);
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
