#include <mpi.h>
#include <math.h>
using namespace std;

#include "staticFuncs.h"
#include "float.h"

void staticHorizontal(ConfigData* data, float* pixels)
{
    int rowHeight = data->height / data->mpi_procs;

    int startRow = data->mpi_rank * rowHeight;
    int stopRow = ( data->mpi_rank + 1 ) * rowHeight;

    if (data->mpi_rank + 1 == data->mpi_procs)
    {
        // Stop if too many processes
        stopRow = data->height;
    }

    for( int row = startRow; row < stopRow; row++ )
    {
        for( int column = 0; column < data->width; column++ )
        {

            //Calculate the index into the array.
            int baseIndex = 3 * ( row * data->width + column );

            //Call the function to shade the pixel.
            shadePixel(&(pixels[baseIndex]), row, column, data);
        }
    }
}

void staticVertical(ConfigData* data, float* pixels)
{
    int colWidth = data->width / data->mpi_procs;

    int startCol = data->mpi_rank * colWidth;
    int stopCol = ( data->mpi_rank + 1 ) * colWidth;

    if (data->mpi_rank + 1 == data->mpi_procs)
    {
        // Stop if too many processes
        stopCol = data->width;
    }

    for ( int row = 0; row < data->height; row++ )
    {
        for ( int column = startCol; column < stopCol; column++ )
        {
            //Calculate the index into the array.
            int baseIndex = 3 * ( row * data->width + column );

            //Call the function to shade the pixel.
            shadePixel(&(pixels[baseIndex]), row, column, data);
        }
    }
}

void staticBlocks(ConfigData* data, float* pixels)
{
    //Determine the range in which to shade pixels
    int blockDimensions = (int)(sqrt((double)(data->mpi_procs)));
    int blockWidth = (data->width)/(blockDimensions);
    int blockHeight = (data->height)/(blockDimensions);

    int startRow = (data->mpi_rank / blockDimensions) * blockHeight;
    int stopRow = ((data->mpi_rank / blockDimensions) + 1) * blockHeight;
    
    int startCol = (data->mpi_rank % blockDimensions) * blockWidth;
    int stopCol = ((data->mpi_rank % blockDimensions) + 1) * blockWidth;

    //Cut off the dimensions at the right and bottom of the image
    if ((data->mpi_rank + 1) % blockDimensions == 0)
    {
        stopCol = data->width;
    }

    if ((data->mpi_rank + blockDimensions) >= data->mpi_procs)
    {
        stopRow = data->height;
    }

    //Shade the relevant pixels
    for ( int row = startRow; row < stopRow; row++ )
    {
        for( int column = startCol; column < stopCol; column++ )
        {
            //Calculate the index into the array.
            int baseIndex = 3 * ( row * data->width + column );

            //Call the function to shade the pixel.
            shadePixel(&(pixels[baseIndex]), row, column, data);
        }
    }

}

void staticCyclesHorizontal(ConfigData* data, float* pixels)
{
    for ( int baseRow = (data->mpi_rank * data->cycleSize); baseRow < data->height; baseRow += (data->mpi_procs * data->cycleSize) )
    {
        for (int row = baseRow; (row < baseRow + data->cycleSize) && (row < data->height); ++row)
        {
            for (int column = 0; column < data->width; ++column)
            {                
                //Calculate the index into the array.
                int baseIndex = 3 * ( row * data->width + column );

                //Call the function to shade the pixel.
                shadePixel(&(pixels[baseIndex]), row, column, data);
            }
        }
    }
}

void staticCyclesVertical(ConfigData* data, float* pixels)
{
    for ( int baseCol = (data->mpi_rank * data->cycleSize); baseCol < data->width; baseCol += (data->mpi_procs * data->cycleSize) )
    {
        for (int column = baseCol; (column < baseCol + data->cycleSize) && (column < data->width); ++column)
        {
            // Go through each column
            for (int row = 0; row < data->height; ++row)
            {
                //Calculate the index into the array.
                int baseIndex = 3 * ( row * data->width + column );
    
                //Call the function to shade the pixel.
                shadePixel(&(pixels[baseIndex]), row, column, data);
            }
        }
    }
}


/* COMMUNICATION FUNCTIONS */

void communicateHorizontal(ConfigData* data, float* globalPixels, float* localPixels, int proc)
{
    int startRow = proc * (data->height / data->mpi_procs);
    int stopRow = (proc + 1) * (data->height / data->mpi_procs);

    if (proc + 1 == data->mpi_procs)
    {
        // Stop if too many processes
        stopRow = data->height;
    }
    
    for (int row = startRow; row < stopRow; row++)
    {
        for ( int column = 0; column < data->width; column++ )
        {
            //Calculate the index into the array.
            int baseIndex = 3 * ( row * data->width + column );

            //Call the function to shade the pixel.
            globalPixels[baseIndex] = localPixels[baseIndex];
            globalPixels[baseIndex + 1] = localPixels[baseIndex + 1];
            globalPixels[baseIndex + 2] = localPixels[baseIndex + 2];
        }   
    }
}

void communicateVertical(ConfigData* data, float* globalPixels, float* localPixels, int proc)
{
    int startCol = proc * (data->width / data->mpi_procs);
    int stopCol = (proc + 1) * (data->width / data->mpi_procs);

    if (proc + 1 == data->mpi_procs)
    {
        // Stop if there are too many processes
        stopCol = data->width;
    }

    for ( int row = 0; row < data->height; row++ )
    {
        for ( int column = startCol; column < stopCol; column++ )
        {
            //Calculate the index into the array.
            int baseIndex = 3 * ( row * data->width + column );

            //Call the function to shade the pixel.
            globalPixels[baseIndex] = localPixels[baseIndex];
            globalPixels[baseIndex + 1] = localPixels[baseIndex + 1];
            globalPixels[baseIndex + 2] = localPixels[baseIndex + 2];
        }
    }
}

void communicateBlocks(ConfigData* data, float* globalPixels, float* localPixels, int proc)
{
    //Calculate the grid dimensions and the width/height of each block
    int blockDimensions = (int)(sqrt((double)(data->mpi_procs)));
    int blockWidth = (data->width)/(blockDimensions);
    int blockHeight = (data->height)/(blockDimensions);
    //Determine the square of pixels that the worker calculated
    int startRow = (proc / blockDimensions) * blockHeight;
    int stopRow = ((proc / blockDimensions) + 1) * blockHeight;
    
    int startCol = (proc % blockDimensions) * blockWidth;
    int stopCol = ((proc % blockDimensions) + 1) * blockWidth;
    //Cut off at the right and bottom borders of the image
    if ((proc + 1) % blockDimensions == 0)
    {
        stopCol = data->width;
    }
    if ((proc + blockDimensions) >= data->mpi_procs)
    {
        stopRow = data->height;
    }

    for ( int row = startRow; row < stopRow; row++ )
    {
        for ( int column = startCol; column < stopCol; column++ )
        {
            //Calculate the index into the array.
            int baseIndex = 3 * ( row * data->width + column );

            //Assign the pixels to the global image.
            globalPixels[baseIndex] = localPixels[baseIndex];
            globalPixels[baseIndex + 1] = localPixels[baseIndex + 1];
            globalPixels[baseIndex + 2] = localPixels[baseIndex + 2];
        }
    }
}

void communicateCyclesHorizontal(ConfigData* data, float* globalPixels, float* localPixels, int proc)
{
    for ( int baseRow = (proc * data->cycleSize); baseRow < data->height; baseRow += (data->mpi_procs * data->cycleSize) )
    {
        for (int row = baseRow; (row < baseRow + data->cycleSize) && (row < data->height); ++row)
        {
            for (int column = 0; column < data->width; ++column)
            {
                //Calculate the index into the array.
                int baseIndex = 3 * ( row * data->width + column );
    
                //Call the function to shade the pixel.
                globalPixels[baseIndex] = localPixels[baseIndex];
                globalPixels[baseIndex + 1] = localPixels[baseIndex + 1];
                globalPixels[baseIndex + 2] = localPixels[baseIndex + 2];
            }
        }
    }
}

void communicateCyclesVertical(ConfigData* data, float* globalPixels, float* localPixels, int proc)
{
    for ( int baseCol = (proc * data->cycleSize); baseCol < data->width; baseCol += (data->mpi_procs * data->cycleSize) )
    {
        for (int column = baseCol; (column < baseCol + data->cycleSize) && (column < data->width); ++column)
        {
            for (int row = 0; (row < data->height); ++row)
            {
                //Calculate the index into the array.
                int baseIndex = 3 * ( row * data->width + column );

                //Call the function to shade the pixel.
                globalPixels[baseIndex] = localPixels[baseIndex];
                globalPixels[baseIndex + 1] = localPixels[baseIndex + 1];
                globalPixels[baseIndex + 2] = localPixels[baseIndex + 2];
            }
        }
    }
}
