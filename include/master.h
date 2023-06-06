#ifndef __MASTER_PROCESS_H__
#define __MASTER_PROCESS_H__

#include "RayTrace.h"

//This function is the main that only the master process
//will run.
//
//Inputs:
//    data - the ConfigData that holds the scene information.
//
//Outputs: None
void masterMain( ConfigData *data, DynamicProperties* dprops );

//This function will perform ray tracing when no MPI use was
//given.
//
//Inputs:
//    data - the ConfigData that holds the scene information.
//
//Outputs: None
void masterSequential(ConfigData *data, float* pixels);

// Dynamic Partition Method
// Communicates with worker threads to divide up dynamic blocks of given image
void dynamic_master(ConfigData* data, DynamicProperties* dprops, float* pixels, double* commStart);

// A method to copy pixels over from one array to another
// when using dynamic partitioning method
void includePixels(ConfigData* data, DynamicProperties* dprops, float* globalPixels, float* localPixels, int unitNum);

#endif
