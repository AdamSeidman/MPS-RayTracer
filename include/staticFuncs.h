#ifndef __rt_static_funcs__
#define __rt_static_funcs__

#include "RayTrace.h"

// Functions to shade pixels for static partitioning methods

void staticHorizontal(ConfigData* data, float* pixels);
void staticVertical(ConfigData* data, float* pixels);
void staticBlocks(ConfigData* data, float* pixels);
void staticCyclesHorizontal(ConfigData* data, float* pixels);
void staticCyclesVertical(ConfigData* data, float* pixels);


// Functions to communicate shaded pixels to master thread

void communicateHorizontal(ConfigData* data, float* globalPixels, float* localPixels, int proc);
void communicateVertical(ConfigData* data, float* globalPixels, float* localPixels, int proc);
void communicateBlocks(ConfigData* data, float* globalPixels, float* localPixels, int proc);
void communicateCyclesHorizontal(ConfigData* data, float* globalPixels, float* localPixels, int proc);
void communicateCyclesVertical(ConfigData* data, float* globalPixels, float* localPixels, int proc);

#endif