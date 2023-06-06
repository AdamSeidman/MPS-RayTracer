#ifndef __SLAVE_PROCESS_H__
#define __SLAVE_PROCESS_H__

#include "RayTrace.h"

// Main Worker Thread
void slaveMain( ConfigData *data, DynamicProperties* dprops );

// Worker thread for dynamic partitioning mode
void dynamic_worker(ConfigData* data, DynamicProperties* dprops, float* pixels);

#endif
