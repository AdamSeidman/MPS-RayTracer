//This file contains the code that the master process will execute.

#include <iostream>
#include <mpi.h>
#include <math.h>

#include "RayTrace.h"
#include "master.h"
#include "staticFuncs.h"

// A method to copy pixels over from one array to another
// when using dynamic partitioning method
void includePixels(ConfigData* data, DynamicProperties* dprops, float* globalPixels, float* localPixels, int unitNum)
{
    int posX = data->dynamicBlockWidth * (unitNum % dprops->numXUnits);
    int posY = data->dynamicBlockHeight * (unitNum / dprops->numYUnits);

    for (int column = posX; column < posX + data->dynamicBlockWidth; ++column)
    {
        if (column >= data->width)
        { break; } // Leave if outside of image

        for (int row = posY; row < posY + data->dynamicBlockHeight; ++row)
        {
            if (row >= data->height)
            { break; } // Leave if outside of image

            // Calculate the index into the array
            int baseIndex = 3 * ( row * data->width + column );

            // Save to globalPixels
            globalPixels[baseIndex] = localPixels[baseIndex];
            globalPixels[baseIndex + 1] = localPixels[baseIndex + 1];
            globalPixels[baseIndex + 2] = localPixels[baseIndex + 2];
        }
    }
}

void masterMain(ConfigData* data, DynamicProperties* dprops)
{
    
    //Depending on the partitioning scheme, different things will happen.
    //You should have a different function for each of the required 
    //schemes that returns some values that you need to handle.
    
    //Allocate space for the image on the master.
    float* localPixels = new float[3 * data->width * data->height];
    float* globalPixels = new float[ 3 * data->width * data->height];
    
    //Execution time will be defined as how long it takes
    //for the given function to execute based on partitioning
    //type.
    double renderTime = 0.0, startTime, stopTime;
    double commStart = 0.0;
    double compTime = 0.0;

	//Add the required partitioning methods here in the case statement.
	//You do not need to handle all cases; the default will catch any
	//statements that are not specified. This switch/case statement is the
	//only place that you should be adding code in this function. Make sure
	//that you update the header files with the new functions that will be
	//called.
	//It is suggested that you use the same parameters to your functions as shown
	//in the sequential example below.

    startTime = MPI_Wtime();

    void (*comm_func)(ConfigData*, float*, float*, int) = NULL;

    switch (data->partitioningMode) // Call the partitioning method and assign communication method
    {
        case PART_MODE_NONE:
            //Call the function that will handle this.
            masterSequential(data, globalPixels);
            commStart = MPI_Wtime();
            break;

        case PART_MODE_STATIC_STRIPS_HORIZONTAL:
            comm_func = communicateHorizontal;
            staticHorizontal(data, globalPixels);
            break;

        case PART_MODE_STATIC_STRIPS_VERTICAL:
            comm_func = communicateVertical;
            staticVertical(data, globalPixels);
            break;

        case PART_MODE_STATIC_BLOCKS:
            comm_func = communicateBlocks;
            staticBlocks(data, globalPixels);
            break;

        case PART_MODE_STATIC_CYCLES_HORIZONTAL:
            comm_func = communicateCyclesHorizontal;
            staticCyclesHorizontal(data, globalPixels);
            break;

        case PART_MODE_STATIC_CYCLES_VERTICAL:
            comm_func = communicateCyclesVertical;
            staticCyclesVertical(data, globalPixels);
            break;

        case PART_MODE_DYNAMIC:
            dynamic_master(data, dprops, globalPixels, &commStart);
            break;

        default:
            break; // Should not reach here
    }

    if (comm_func != NULL)
    {
        // If communication needs to occur, do it for each worker process
        commStart = MPI_Wtime();
        for (int proc = 1; proc < data->mpi_procs; proc++)
        {
            MPI_Recv(localPixels, (3 * data->width * data->height), MPI_FLOAT, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            comm_func(data, globalPixels, localPixels, proc); // Assigned function from staticFuncs
        }
    }

    stopTime = MPI_Wtime();

    // Calculate times
    renderTime = stopTime - startTime;
    compTime = commStart - startTime;
    std::cout << "Execution Time: " << renderTime << " seconds" << std::endl << std::endl;
    std::cout << "Master Computation Time: " << compTime << " seconds" << std::endl << std::endl;

    //After this gets done, save the image.
    std::cout << "Image will be save to: ";

    // Save to file
    std::string file = generateFileName(data);
    std::cout << file << std::endl;
    savePixels(file, globalPixels, data);

    //Delete the pixel data.
    delete[] globalPixels;
    delete[] localPixels;

}

// Dynamic Partition Method
// Communicates with worker threads to divide up dynamic blocks of given image
void dynamic_master(ConfigData* data, DynamicProperties* dprops, float* pixels, double* commStart)
{
    // Variables that store worker thread information
    int workRecords[dprops->numberUnits];
    int dummy_flag;
    int unit = 0;
    MPI_Request requests[(data->mpi_procs) - 1];

    for (int proc = 1; proc < data->mpi_procs; ++proc)
    {
        // Initial assignment
        int unitNum = unit;
        
        // Set up requests
        MPI_Irecv(&dummy_flag, 1, MPI_INT, proc, 2, MPI_COMM_WORLD, &requests[proc - 1]);
        
        // Send unit numbers
        MPI_Send(&unitNum, 1, MPI_INT, proc, 3, MPI_COMM_WORLD);
        workRecords[unit] = proc;

        ++unit;
        if (unit == dprops->numberUnits) // Stop if num procs > work units
        { break; }
    }

    int index; // index of completed request

    while (unit < dprops->numberUnits)
    {
        // Wait for a request to be marked as complete
        MPI_Waitany((data->mpi_procs - 1), requests, &index, MPI_STATUS_IGNORE);
        ++index; // Make index reflect process number

        // Send new work unit
        int unitNum = unit;
        MPI_Send(&unitNum, 1, MPI_INT, index, 3, MPI_COMM_WORLD);
        workRecords[unit] = index;

        // Send request for next flag
        MPI_Irecv(&dummy_flag, 1, MPI_INT, index, 2, MPI_COMM_WORLD, &requests[index - 1]);

        // Get ready for next work unit
        ++unit;
    }

    // Wait for the rest of the flags to finish their work units.
    MPI_Waitall((data->mpi_procs - 1), requests, MPI_STATUSES_IGNORE);

    *commStart = MPI_Wtime(); // Set the communication time

    for (int proc = 1; proc < data->mpi_procs; ++proc)
    {
        // Tell each process it is done
        int endCode = END_CODE;
        MPI_Send(&endCode, 1, MPI_INT, proc, 3, MPI_COMM_WORLD);

        // Get rendered pixels from each process
        float* localPixels = new float[3 * data->width * data->height];
        MPI_Recv(localPixels, (3 * data->width * data->height), MPI_FLOAT, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int record = 0; record < dprops->numberUnits; ++record)
        {
            // Save pixels from each work unit in process
            if (workRecords[record] == proc)
            {
                includePixels(data, dprops, pixels, localPixels, record);
            }
        }
    }
}

void masterSequential(ConfigData* data, float* pixels)
{
    //Start the computation time timer.
    double computationStart = MPI_Wtime();

    //Render the scene.
    for( int row = 0; row < data->height; ++row )
    {
        for( int column = 0; column < data->width; ++column )
        {
            //Calculate the index into the array.
            int baseIndex = 3 * ( row * data->width + column );

            //Call the function to shade the pixel.
            shadePixel(&(pixels[baseIndex]), row, column, data);
        }
    }

    //Stop the comp. timer
    double computationStop = MPI_Wtime();
    double computationTime = computationStop - computationStart;

    //After receiving from all processes, the communication time will
    //be obtained.
    double communicationTime = 0.0;

    //Print the times and the c-to-c ratio
	//This section of printing, IN THIS ORDER, needs to be included in all of the
	//functions that you write at the end of the function.
    std::cout << "Total Computation Time: " << computationTime << " seconds" << std::endl;
    std::cout << "Total Communication Time: " << communicationTime << " seconds" << std::endl;
    double c2cRatio = communicationTime / computationTime;
    std::cout << "C-to-C Ratio: " << c2cRatio << std::endl;
}
