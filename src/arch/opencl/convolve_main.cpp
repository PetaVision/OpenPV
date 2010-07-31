#include "CLDevice.hpp"
#include "convolve.h"

#include "Timer.hpp"

#include <stdio.h>
#include <stdlib.h>

#define DEVICE 0

void init_random_data(float * idata, int nxGlobal, int nyGlobal, int nPad);
void init_test_data(float * idata, int nxGlobal, int nyGlobal, int nPad);
void convolve_c(float * idata, float * odata, int nxGlobal, int nyGlobal, int nPad);
void check_results(float * results1, float * results2, int nxGlobal, int nyGlobal, int nPad);
void validate_results(float * results1, float * results2, int nxGlobal, int nyGlobal, int nPad);

int main(int argc, char * argv[])
{
   unsigned int nxl, nyl;

   PV::Timer timer;
   
   int status = 0;
   int argid  = 0;
   int query  = 0;
   int device = DEVICE;
   
   if (argc > 1) {
      device = atoi(argv[1]);
   }
	
   PV::CLDevice * cld = new PV::CLDevice(device);
   
   // query and print information about the devices found
   //
   if (query) cld->query_device_info();
   
   if (device == 0) {
      nxl = NXL;
      nyl = NYL;
      cld->createKernel("convolve.cl", "convolve");
   }
   else {
      nxl = 1;
      nyl = 1;
      cld->createKernel("convolve_cpu.cl", "convolve_cpu");
   }
   
   size_t global;                      // global domain size for our calculation
   size_t local;                       // local domain size for our calculation
   
   cl_mem input;                         // device memory used for the input array
   cl_mem output;                        // device memory used for the output array

   const unsigned int size_ex  = SIZE_EX;
   const unsigned int size_img = SIZE_IMG;
   
   const unsigned int nxGlobal = NXGLOBAL;
   const unsigned int nyGlobal = NYGLOBAL;
   
   const unsigned int nPad  = NPAD;
   const unsigned int nPad2 = NPAD2;
   
   const unsigned int sx = 1;
   const unsigned int sy = nxGlobal + nPad2;
   
   float * data     = (float *) malloc(size_ex * sizeof(float));    // original data set given to device
   float * results_d = (float *) malloc(size_img * sizeof(float));  // results returned from device
   float * results_l = (float *) malloc(size_img * sizeof(float));  // results returned from local thread
   unsigned char * activity = (unsigned char *) malloc(size_ex * sizeof(unsigned char));
	
   assert(data != NULL);
   assert(results_d != NULL);
   assert(results_l != NULL);
   assert(activity != NULL);
   
   bzero(data,      size_ex*sizeof(float));
   bzero(results_d, size_img*sizeof(float));
   bzero(results_l, size_img*sizeof(float));
   bzero(activity,  size_ex*sizeof(unsigned char));
	
   size_t local_size_ex = (nxl + nPad2) * (nyl + nPad2) * sizeof(float); // padded image patch
   
   init_random_data(data, nxGlobal, nyGlobal, nPad);

   // time running kernel locally
   //
   timer.start();
   convolve_c(data, results_l, nxGlobal, nyGlobal, nPad);
   timer.stop();
   printf("Executing on local:  "); timer.elapsed_time();

#ifdef USE_ACTIVITY_BYTES
   input  = cld->addReadBuffer (argid++, activity, size_ex*sizeof(unsigned char));
#else
   input  = cld->addReadBuffer (argid++, data,     size_ex*sizeof(float));
#endif
   output = cld->addWriteBuffer(argid++, size_img*sizeof(float));
   status = cld->addKernelArg  (argid++, nxGlobal);
   status = cld->addKernelArg  (argid++, nyGlobal);
   status = cld->addKernelArg  (argid++, nPad);
   status = cld->addLocalArg   (argid++, local_size_ex);
   
   timer.start();
#ifdef USE_ACTIVITY_BYTES
   cld->run(nxGlobal/4, nyGlobal, nxl, nyl);
#else
   cld->run(nxGlobal, nyGlobal, nxl, nyl);
#endif
   timer.stop();
   printf("Executing on device: "); timer.elapsed_time();
   printf("Elapsed time on device:            device time == %f \n", ((float)cld->get_execution_time())/1.0e6);
   
   cld->copyResultsBuffer(output, results_d, size_img*sizeof(float));
   
   // Check results for accuracy
   //
   check_results(results_d, results_l, nxGlobal, nyGlobal, nPad);

   // Shutdown and cleanup
   //
   clReleaseMemObject(input);
   clReleaseMemObject(output);
   delete cld;
   
   printf("Finished...\n");
   
   return status;
}

/**
 * run the convolution on a single CPU core
 */
void convolve_c(float * idata, float * odata, int nxGlobal, int nyGlobal, int nPad)
{
   const int sy = nxGlobal + 2*nPad;

   for (int j = nPad; j < nyGlobal + nPad; j++) {
      for (int i = nPad; i < nxGlobal + nPad; i++) {
         float sum = 0.0f;
         int k = (i-nPad) + (j-nPad)*nxGlobal;
         int kex = i + j*(nxGlobal + 2*nPad);
         
         for (int ii = 0; ii < 3; ii++) {
            int offset = (ii-1)*sy;
            sum += 1.1f*idata[kex+offset-1] + 2.1f*idata[kex+offset] + 3.1f*idata[kex+offset+1];
         }
         odata[k] = sum;         
      }
   }
}

void check_results(float * results1, float * results2, int nxGlobal, int nyGlobal, int nPad)
{
   int status = 0;
   for (int k = 0; k < nxGlobal*nyGlobal; k++) {
      if (fabs(results1[k] - results2[k]) > .00001) {
         printf("check_results: results differ at k==%d, results1==%f, results2==%f\n",
                k, results1[k], results2[k]);
         return;
      }
   }
}

//
// Validate our results
//
void validate_results(float * data, float * results, int nxGlobal, int nyGlobal, int nPad)
{
   unsigned int correct = 0;               // number of correct results returned
   int incorrect = 0;
	
   for (int j = nPad; j < nyGlobal + nPad; j++) {
      for (int i = nPad; i < nxGlobal + nPad; i++) {
         int k = (i-nPad) + (j-nPad)*nxGlobal;
         int kex = i + j*(nxGlobal + 2*nPad);
         if (results[k] == data[kex]) {
            correct++;
            //printf("        correct value! results[%d]==%f, data[%d]==%f\n",
            //         k, results[k], kex, data[kex]);
         } else {
            incorrect += 1;
            if (incorrect < 10) {
               printf("Error incorrect value! results[%d]==%f, data[%d]==%f\n",
                      k, results[k], kex, data[kex]);
            }
         }
      }
   }

   // Print a brief summary detailing the results
   //
   printf("Computed '%d/%d' correct values! results[0]==%f, %f\n",
          correct, nxGlobal*nyGlobal, results[0], results[1]);
}	


#define PV_RANDOM_MAX       0x7fffffff
#define PV_INV_RANDOM_MAX   (1.0 / (double) PV_RANDOM_MAX)

static inline double pv_random_prob()
{
   return (double) random() * PV_INV_RANDOM_MAX;
}

void init_test_data(float * data, int nxGlobal, int nyGlobal, int nPad)
{
   // fill in the interior
   //
   int ii = 0;
   for (int j = nPad; j < nyGlobal + nPad; j++) {
      for (int i = nPad; i < nxGlobal + nPad; i++) {
         data[i + j*(nxGlobal + 2*nPad)] = ii;
      }
   }
}  

void init_random_data(float * data, int nxGlobal, int nyGlobal, int nPad)
{
   // initialize the interior
   //
   int ii = 0;
   for (int j = nPad; j < nyGlobal + nPad; j++) {
      for (int i = nPad; i < nxGlobal + nPad; i++) {
         data[i + j*(nxGlobal + 2*nPad)] = pv_random_prob();
      }
   }
}
