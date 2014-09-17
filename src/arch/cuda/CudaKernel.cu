/*
 * CudaKernel.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#include "CudaKernel.hpp"

#include <assert.h>
#include "cuda_util.hpp"
#include <sys/stat.h>


namespace PVCuda {

CudaKernel::CudaKernel(CudaDevice* inDevice){
   argsSet = false;
   dimsSet = false;
   this->device = inDevice;
}

CudaKernel::CudaKernel()
{
   argsSet = false;
   dimsSet = false;
   device = NULL;
}

CudaKernel::~CudaKernel()
{
}

//All global work sizes are total global
int CudaKernel::run(int global_work_size)
{
   setDims(1, 1, global_work_size, 1, 1, 1);
   return do_run();
}

int CudaKernel::run(int global_work_size, int local_work_size)
{
   setDims(1, 1, global_work_size, 1, 1, local_work_size);
   return do_run();
}

int CudaKernel::run(int gWorkSizeX, int gWorkSizeY, int lWorkSizeX, int lWorkSizeY)
{
   setDims(gWorkSizeY, 1, gWorkSizeX, lWorkSizeY, 1, lWorkSizeX);
   return do_run();
}

int CudaKernel::run(int gWorkSizeX, int gWorkSizeY, int gWorkSizeF,
                  int lWorkSizeX, int lWorkSizeY, int lWorkSizeF)
{
   setDims(gWorkSizeX, gWorkSizeY, gWorkSizeF, lWorkSizeX, lWorkSizeY, lWorkSizeF);
   return do_run();
}

void CudaKernel::setDims(int gWorkSizeX, int gWorkSizeY, int gWorkSizeF, int lWorkSizeX, int lWorkSizeY, int lWorkSizeF){
   assert(device);
   assert(gWorkSizeF % lWorkSizeF == 0);
   assert(gWorkSizeX % lWorkSizeX == 0);
   assert(gWorkSizeY % lWorkSizeY == 0);
   int gridSizeF = gWorkSizeF / lWorkSizeF;
   int gridSizeX = gWorkSizeX / lWorkSizeX;
   int gridSizeY = gWorkSizeY / lWorkSizeY;

   int max_grid_size_f = device->get_max_grid_size_dimension(0);
   if(gWorkSizeF > max_grid_size_f){
      printf("run: global work size f %d is bigger than allowed grid size f %d\n", gridSizeF, max_grid_size_f);
      exit(-1);
   }
   int max_grid_size_x = device->get_max_grid_size_dimension(1);
   if(gridSizeX > max_grid_size_x){
      printf("run: global work size x %d is bigger than allowed grid size x %d\n", gridSizeX, max_grid_size_x);
      exit(-1);
   }
   int max_grid_size_y = device->get_max_grid_size_dimension(2);
   if(gridSizeY > max_grid_size_y){
      printf("run: global work size y %d is bigger than allowed grid size y %d\n", gridSizeY, max_grid_size_y);
      exit(-1);
   }

   int max_threads = device->get_max_threads();
   int local_work_size = lWorkSizeX * lWorkSizeY * lWorkSizeF;
   if (local_work_size > max_threads) {
      printf("run: local_work_size %d is bigger than allowed thread size %d\n", local_work_size, max_threads);
      exit(-1);
   }

   int max_threads_f = device->get_max_block_size_dimension(0);
   if (lWorkSizeF > max_threads_f) {
      printf("run: local_work_size_f %d is bigger than allowed thread size f %d\n", lWorkSizeF, max_threads_f);
      exit(-1);
   }

   int max_threads_x = device->get_max_block_size_dimension(1);
   if (lWorkSizeX > max_threads_x) {
      printf("run: local_work_size_x %d is bigger than allowed thread size x %d\n", lWorkSizeX, max_threads_x);
      exit(-1);
   }

   int max_threads_y = device->get_max_block_size_dimension(2);
   if (lWorkSizeY > max_threads_y) {
      printf("run: local_work_size_y %d is bigger than allowed thread size y %d\n", lWorkSizeY, max_threads_y);
      exit(-1);
   }

   grid_size.x = gridSizeF;
   grid_size.y = gridSizeX;
   grid_size.z = gridSizeY;
   block_size.x = lWorkSizeF;
   block_size.y = lWorkSizeX;
   block_size.z = lWorkSizeY;
   dimsSet = true;
}

}
