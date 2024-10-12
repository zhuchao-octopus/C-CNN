/////////////////////////////////////////////////////////////////////////////////////////////
/*
 *  ann-configuration.h
 *  Home Page :http://www.1234998.top
 *  Created on: June 05, 2023
 *  Author: M
 */
/////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _INC_ANN_CONFIGURATION_H_
#define _INC_ANN_CONFIGURATION_H_

void NeuralNetCreateAndInit_Cifar10(void);
void NeuralNetCreateAndInit_Cifar100(void);

TPNeuralNet NeuralNetInit_C_CNN_16(char* NetName);
TPNeuralNet NeuralNetInit_C_CNN_9(char* NetName);


#endif

