/*
 * CudaKernel.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#include "CudaKernel.hpp"

#include "cuda_util.hpp"
#include <assert.h>
#include <cmath>
#include <sys/stat.h>

namespace PVCuda {

CudaKernel::CudaKernel(CudaDevice *inDevice) {
   argsSet     = false;
   dimsSet     = false;
   mDevice     = inDevice;
   mKernelName = nullptr;
}

CudaKernel::CudaKernel() {
   argsSet     = false;
   dimsSet     = false;
   mDevice     = nullptr;
   mKernelName = nullptr;
}

CudaKernel::~CudaKernel() {}

int CudaKernel::run() {
   int runResult = do_run();
   handleCallError(mKernelName);
   return runResult;
}

// All global work sizes are total global
int CudaKernel::run(long global_work_size) {
   setDims(global_work_size, 1, 1, 1, 1, 1);
   int runResult = do_run();
   handleCallError(mKernelName);
   return runResult;
}

int CudaKernel::run(long global_work_size, long local_work_size) {
   setDims(global_work_size, 1, 1, local_work_size, 1, 1);
   int runResult = do_run();
   handleCallError(mKernelName);
   return runResult;
}

int CudaKernel::run_nocheck(long global_work_size, long local_work_size) {
   setDims(global_work_size, 1, 1, local_work_size, 1, 1, false);
   int runResult = do_run();
   handleCallError(mKernelName);
   return runResult;
}

int CudaKernel::run(long gWorkSizeX, long gWorkSizeY, long lWorkSizeX, long lWorkSizeY) {
   setDims(gWorkSizeX, gWorkSizeY, 1, lWorkSizeX, lWorkSizeY, 1);
   int runResult = do_run();
   handleCallError(mKernelName);
   return runResult;
}

int CudaKernel::run_nocheck(long gWorkSizeX, long gWorkSizeY, long lWorkSizeX, long lWorkSizeY) {
   setDims(gWorkSizeX, gWorkSizeY, 1, lWorkSizeX, lWorkSizeY, 1, false);
   int runResult = do_run();
   handleCallError(mKernelName);
   return runResult;
}

int CudaKernel::run(
      long gWorkSizeX,
      long gWorkSizeY,
      long gWorkSizeF,
      long lWorkSizeX,
      long lWorkSizeY,
      long lWorkSizeF) {
   setDims(gWorkSizeF, gWorkSizeX, gWorkSizeY, lWorkSizeF, lWorkSizeX, lWorkSizeY);
   int runResult = do_run();
   handleCallError(mKernelName);
   return runResult;
}

// These dims are based on gpu dimensions, not PV dimensions
void CudaKernel::setDims(
      long gWorkSizeX,
      long gWorkSizeY,
      long gWorkSizeZ,
      long lWorkSizeX,
      long lWorkSizeY,
      long lWorkSizeZ,
      bool error) {
   assert(mDevice);
   if (error) {
      assert(gWorkSizeX % lWorkSizeX == 0);
      assert(gWorkSizeY % lWorkSizeY == 0);
      assert(gWorkSizeZ % lWorkSizeZ == 0);
   }
   unsigned int gridSizeX = (unsigned int)std::ceil((float)gWorkSizeX / (float)lWorkSizeX);
   unsigned int gridSizeY = (unsigned int)std::ceil((float)gWorkSizeY / (float)lWorkSizeY);
   unsigned int gridSizeZ = (unsigned int)std::ceil((float)gWorkSizeZ / (float)lWorkSizeZ);

   unsigned int max_grid_size_x = mDevice->get_max_grid_size_dimension(0);
   if (gridSizeX > max_grid_size_x) {
      Fatal().printf(
            "run: global work size x %ld is bigger than allowed grid size x %d\n",
            gridSizeX,
            max_grid_size_x);
   }
   int max_grid_size_y = mDevice->get_max_grid_size_dimension(1);
   if (gridSizeY > max_grid_size_y) {
      Fatal().printf(
            "run: global work size y %ld is bigger than allowed grid size y %d\n",
            gridSizeY,
            max_grid_size_y);
   }

   int max_grid_size_z = mDevice->get_max_grid_size_dimension(2);
   if (gWorkSizeZ > max_grid_size_z) {
      Fatal().printf(
            "run: global work size f %ld is bigger than allowed grid size f %d\n",
            gridSizeZ,
            max_grid_size_z);
   }

   int max_threads      = mDevice->get_max_threads();
   long local_work_size = lWorkSizeX * lWorkSizeY * lWorkSizeZ;
   if (local_work_size > max_threads) {
      Fatal().printf(
            "run: local_work_size %ld is bigger than allowed thread size %d\n",
            local_work_size,
            max_threads);
   }

   int max_threads_x = mDevice->get_max_block_size_dimension(0);
   if (lWorkSizeX > max_threads_x) {
      Fatal().printf(
            "run: local_work_size_x %ld is bigger than allowed thread size x %d\n",
            lWorkSizeX,
            max_threads_x);
   }

   int max_threads_y = mDevice->get_max_block_size_dimension(1);
   if (lWorkSizeY > max_threads_y) {
      Fatal().printf(
            "run: local_work_size_y %ld is bigger than allowed thread size y %d\n",
            lWorkSizeY,
            max_threads_y);
   }

   int max_threads_z = mDevice->get_max_block_size_dimension(2);
   if (lWorkSizeZ > max_threads_z) {
      Fatal().printf(
            "run: local_work_size_f %ld is bigger than allowed thread size f %d\n",
            lWorkSizeZ,
            max_threads_z);
   }

   grid_size.x  = gridSizeX;
   grid_size.y  = gridSizeY;
   grid_size.z  = gridSizeZ;
   block_size.x = (unsigned int)lWorkSizeX;
   block_size.y = (unsigned int)lWorkSizeY;
   block_size.z = (unsigned int)lWorkSizeZ;
   dimsSet      = true;
}

void CudaKernel::cudnnHandleError(cudnnStatus_t status, const char *errStr) {
   if (status != CUDNN_STATUS_SUCCESS) {
      Fatal() << "CUDNN " << errStr << ": " << cudnnGetErrorString(status) << "\n";
      return;
   }
}

} // end namespace PVCuda
