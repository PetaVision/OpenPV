/**
 * This file tests weight initialization to a 2D Gaussian with sigma = 1.0 and normalized to 1.0
 * Test compares HyPerConn to KernelConn,
 * assumes kernelConn produces correct 2D Gaussian weights
 *
 */

#undef DEBUG_PRINT

#include <layers/HyPerLayer.hpp>
#include <connections/HyPerConn.hpp>
#include <io/io.h>
#include <assert.h>

#include "Example.hpp"

using namespace PV;

int check_kernel_vs_hyper(HyPerConn * cHyPer, HyPerConn * cKernel, int kPre,
		int axonID);

int main(int argc, char * argv[])
{
   PV_Init* initObj = new PV_Init(&argc, &argv);
   if (pv_getopt(argc, argv, "-p", NULL)==0) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank==0) {
         fprintf(stderr, "%s does not take a -p argument; the necessary param file is hardcoded.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   int const cl_argc = argc+2;
   char ** cl_argv = (char **) calloc((size_t) (cl_argc+1), sizeof(char *));
   assert(cl_argv);
   int arg = 0;
   for (arg=0; arg<argc; arg++) {
      cl_argv[arg] = strdup(argv[arg]);
      assert(cl_argv);
   }
   cl_argv[arg++] = strdup("-p");
   cl_argv[arg++] = strdup("input/test_gauss2d.params");
   assert(arg==cl_argc);
   cl_argv[arg] = NULL;
   const char * pre_layer_name = "test_gauss2d pre";
   const char * post_layer_name = "test_gauss2d post";
   const char * pre2_layer_name = "test_gauss2d pre 2";
   const char * post2_layer_name = "test_gauss2d post 2";

   initObj->initialize(cl_argc, cl_argv);
   PV::HyPerCol * hc = new PV::HyPerCol("test_gauss2d column", cl_argc, cl_argv, initObj);
   PV::Example * pre = new PV::Example(pre_layer_name, hc);
   assert(pre);
   PV::Example * post = new PV::Example(post_layer_name, hc);
   assert(post);

   
   PV::HyPerConn * cHyPer = new HyPerConn("test_gauss2d hyperconn", hc);

   PV::HyPerConn * cKernel = new HyPerConn("test_gauss2d kernelconn", hc);

   PV::Example * pre2 = new PV::Example(pre2_layer_name, hc);
   assert(pre2);
   PV::Example * post2 = new PV::Example(post2_layer_name, hc);
   assert(post2);

   PV::HyPerConn * cHyPer1to2 =
         new HyPerConn("test_gauss2d hyperconn 1 to 2", hc);
   assert(cHyPer1to2);

   PV::HyPerConn * cKernel1to2 =
         new HyPerConn("test_gauss2d kernelconn 1 to 2", hc);
   assert(cKernel1to2);

   PV::HyPerConn * cHyPer2to1 =
         new HyPerConn("test_gauss2d hyperconn 2 to 1", hc);
   assert(cHyPer2to1);

   PV::HyPerConn * cKernel2to1 =
         new HyPerConn("test_gauss2d kernelconn 2 to 1", hc);
   assert(cKernel2to1);
   
   int status = 0;

   for (int l=0; l<hc->numberOfLayers(); l++) {
      status = hc->getLayer(l)->communicateInitInfo();
      assert(status==PV_SUCCESS);
   }
   for (int c=0; c<hc->numberOfConnections(); c++) {
      status = hc->getConnection(c)->communicateInitInfo();
      assert(status==PV_SUCCESS);
   }
   for (int l=0; l<hc->numberOfLayers(); l++) {
      status = hc->getLayer(l)->allocateDataStructures();
      assert(status==PV_SUCCESS);
   }
   for( int c=0; c<hc->numberOfConnections(); c++ ) {
      BaseConnection * baseConn = hc->getConnection(c);
      HyPerConn * conn = dynamic_cast<HyPerConn *>(baseConn);
      conn->allocateDataStructures();
      conn->writeWeights(0, true);
   }

   const int axonID = 0;
   int num_pre_extended = pre->clayer->numExtended;
   assert(num_pre_extended == cHyPer->getNumWeightPatches());

   for (int kPre = 0; kPre < num_pre_extended; kPre++) {
     //printf("testing testing 1 2 3...\n");
     status = check_kernel_vs_hyper(cHyPer, cKernel, kPre, axonID);
     assert(status==0);
     status = check_kernel_vs_hyper(cHyPer1to2, cKernel1to2, kPre, axonID);
     assert(status==0);
     status = check_kernel_vs_hyper(cHyPer2to1, cKernel2to1, kPre, axonID);
     assert(status==0);
   }

   delete hc;
   for (int arg=0; arg<cl_argc; arg++) {
      free(cl_argv[arg]);
   }
   free(cl_argv);
   delete initObj;
   return 0;
}

int check_kernel_vs_hyper(HyPerConn * cHyPer, HyPerConn * cKernel, int kPre, int axonID)
{
   assert(cKernel->usingSharedWeights()==true);
   assert(cHyPer->usingSharedWeights()==false);
   int status = 0;
   PVPatch * hyperPatch = cHyPer->getWeights(kPre, axonID);
   PVPatch * kernelPatch = cKernel->getWeights(kPre, axonID);
   int hyPerDataIndex = cHyPer->patchIndexToDataIndex(kPre);
   int kernelDataIndex = cKernel->patchIndexToDataIndex(kPre);

   int nk = cHyPer->fPatchSize() * (int) hyperPatch->nx; // hyperPatch->nf * hyperPatch->nx;
   assert(nk == (cKernel->fPatchSize() * (int) kernelPatch->nx));// assert(nk == (kernelPatch->nf * kernelPatch->nx));
   int ny = hyperPatch->ny;
   assert(ny == kernelPatch->ny);
   int sy = cHyPer->yPatchStride(); // hyperPatch->sy;
   assert(sy == cKernel->yPatchStride()); // assert(sy == kernelPatch->sy);
   pvwdata_t * hyperWeights = cHyPer->get_wData(axonID, hyPerDataIndex); // hyperPatch->data;
   pvwdata_t * kernelWeights = cKernel->get_wDataHead(axonID, kernelDataIndex)+hyperPatch->offset; // kernelPatch->data;
   float test_cond = 0.0f;
   for (int y = 0; y < ny; y++) {
      for (int k = 0; k < nk; k++) {
         test_cond = kernelWeights[k] - hyperWeights[k];
         if (fabs(test_cond) > 0.001f) {
            printf("y %d\n", y);
            printf("k %d\n", k);
            printf("kernelweight %f\n", kernelWeights[k]);
            printf("hyperWeights %f\n", hyperWeights[k]);
            const char * cHyper_filename = "gauss2d_hyper.txt";
            cHyPer->writeTextWeights(cHyper_filename, kPre);
            const char * cKernel_filename = "gauss2d_kernel.txt";
            cKernel->writeTextWeights(cKernel_filename, kPre);
            status=1;
            //exit(EXIT_FAILURE);
         }
      }
      // advance pointers in y
      hyperWeights += sy;
      kernelWeights += sy;
   }
   return status;
}

