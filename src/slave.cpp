//This file contains the code that the slave process will execute.

#include <iostream>
#include <mpi.h>
#include "RayTrace.h"
#include "slave.h"
#include "staticFuncs.h"

void slaveMain(ConfigData* data, DynamicProperties* dprops)
{
    double renderTime = 0.0, startTime, stopTime;

    float* localPixels = new float[3 * data->width * data->height];
    //Depending on the partitioning scheme, different things will happen.
    //You should have a different function for each of the required 
    //schemes that returns some values that you need to handle.

    startTime = MPI_Wtime();

    switch (data->partitioningMode)
    {
        // Call paritioning method main function

        case PART_MODE_NONE:
            //The worker will do nothing since this means sequential operation.
            break;
            
        case PART_MODE_STATIC_STRIPS_HORIZONTAL:
            staticHorizontal(data, localPixels);
            break;

        case PART_MODE_STATIC_STRIPS_VERTICAL:
            staticVertical(data, localPixels);
            break;

        case PART_MODE_STATIC_BLOCKS:
            staticBlocks(data, localPixels);
            break;

        case PART_MODE_STATIC_CYCLES_HORIZONTAL:
            staticCyclesHorizontal(data, localPixels);
            break;

        case PART_MODE_STATIC_CYCLES_VERTICAL:
            staticCyclesVertical(data, localPixels);
            break;

        case PART_MODE_DYNAMIC:
            dynamic_worker(data, dprops, localPixels);
            break;

        default:
            // Sequential
            for( int i = 0; i < data->height; ++i )
            {
                if (data->mpi_rank == (i % (data->mpi_procs)))
                {
                    for( int j = 0; j < data->width; ++j )
                    {
                        int row = i;
                        int column = j;

                        //Calculate the index into the array.
                        int baseIndex = 3 * (((row * data->width) / data->mpi_procs) + column);

                        //Call the function to shade the pixel.
                        shadePixel(&(localPixels[baseIndex]),row,j,data);
                    }
                }
            }
            break;
    }

    // Calculate timings
    stopTime = MPI_Wtime();
    renderTime = stopTime - startTime;

    std::cout << "Worker Computation Time: " << renderTime << " seconds" << std::endl << std::endl;

    // Send information about pixels to main thread
    MPI_Send(localPixels, (3 * data->width * data->height), MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
}

// Worker thread for dynamic partitioning mode
void dynamic_worker(ConfigData* data, DynamicProperties* dprops, float* pixels)
{
    int unitNumber;
    int finFlag = 1;

    // Receive initial work unit number
    MPI_Recv(&unitNumber, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    while (unitNumber != -1)
    {
        // Do shading
        int posX = data->dynamicBlockWidth * (unitNumber % dprops->numXUnits);
        int posY = data->dynamicBlockHeight * (unitNumber / dprops->numYUnits);

        for (int x = posX; x < posX + data->dynamicBlockWidth; ++x)
        {
            if (x >= data->width)
            { break; } // Leave if outside of image

            for (int y = posY; y < posY + data->dynamicBlockHeight; ++y)
            {
                if (y >= data->height)
                { break; } // Leave if outside of image

                int row = y;
                int column = x;

                // Calculate the index into the array
                int baseIndex = 3 * ( row * data->width + column );

                // Call the function to shade the pixel
                shadePixel(&(pixels[baseIndex]),row,column,data);
            }
        }

        MPI_Send(&finFlag, 1, MPI_INT, 0, 2, MPI_COMM_WORLD); // Send flag that thread is complete to master
        MPI_Recv(&unitNumber, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive new work unit number
    }
}
