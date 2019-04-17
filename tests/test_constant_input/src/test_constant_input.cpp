/*
 * test_constant_input.cpp
 *
 * This is a full system test with a retina and a spatially constant
 * image.  The retina is non-spiking and connected to layer l1.
 * Using mirror boundary conditions the output of l1 should also be
 * spatially constant.
 *
 *  Created on: Mar 19, 2010
 *      Author: Craig Rasmussen
 */

#include "TestImage.hpp"

#include "columns/ComponentBasedObject.hpp"
#include "columns/HyPerCol.hpp"
#include "components/PatchSize.hpp"
#include "layers/Retina.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <arch/mpi/mpi.h>

#undef DEBUG_OUTPUT

using namespace PV;

int checkLoc(HyPerCol *hc, const PVLayerLoc *loc);

int checkInput(const PVLayerLoc *loc, const float *data, float val, bool extended);

int main(int argc, char *argv[]) {
   int status = 0;

   PV_Init *initObj = new PV_Init(&argc, &argv, false /*allowUnrecognizedArguments*/);
   initObj->registerKeyword("TestImage", Factory::create<TestImage>);

   // create the managing hypercolumn
   //
   HyPerCol *hc = new HyPerCol(initObj);

   TestImage *image = dynamic_cast<TestImage *>(
         dynamic_cast<HyPerLayer *>(hc->getObjectFromName("test_constant_input_image")));

   HyPerLayer *retina = dynamic_cast<HyPerLayer *>(
         dynamic_cast<HyPerLayer *>(hc->getObjectFromName("test_constant_input_retina")));

   ComponentBasedObject *conn = dynamic_cast<ComponentBasedObject *>(
         hc->getObjectFromName("test_constant_input_connection"));

   hc->allocateColumn();

   auto *patchSize             = conn->getComponentByType<PatchSize>();
   const int nxp               = patchSize->getPatchSizeX();
   const int nyp               = patchSize->getPatchSizeY();
   const PVLayerLoc *imageLoc  = image->getLayerLoc();
   const PVLayerLoc *retinaLoc = retina->getLayerLoc();
   const int nfPre             = imageLoc->nf;

   float sumOfWeights = (float)(nxp * nyp * nfPre);
   if (imageLoc->nx > retinaLoc->nx) {
      sumOfWeights *= imageLoc->nx / retinaLoc->nx;
   }
   if (imageLoc->ny > retinaLoc->ny) {
      sumOfWeights *= imageLoc->ny / retinaLoc->ny;
   }

   hc->run();

   const int rank = hc->columnId();
#ifdef DEBUG_OUTPUT
   DebugLog().printf("[%d]: column: ", rank);
   printLoc(hc->getImageLoc());
   DebugLog().printf("[%d]: image : ", rank);
   printLoc(image->getImageLoc());
   DebugLog().printf("[%d]: retina: ", rank);
   printLoc(*retina->getLayerLoc());
   DebugLog().printf("[%d]: l1    : ", rank);
   printLoc(*l1->getLayerLoc());
#endif

   status = checkLoc(hc, image->getLayerLoc());
   if (status != PV_SUCCESS) {
      Fatal().printf("[%d]: test_constant_input: ERROR in image loc\n", rank);
   }

   status = checkLoc(hc, retina->getLayerLoc());
   if (status != PV_SUCCESS) {
      Fatal().printf("[%d]: test_constant_input: ERROR in retina loc\n", rank);
   }

   float const *imageActivity = image->getComponentByType<ActivityComponent>()->getActivity();
   status = checkInput(image->getLayerLoc(), imageActivity, image->getConstantVal(), true);
   if (status != PV_SUCCESS) {
      Fatal().printf("[%d]: test_constant_input: ERROR in image data\n", rank);
   }

   ActivityComponent *retinaActivity = retina->getComponentByType<ActivityComponent>();
   float retinaVal                   = sumOfWeights * image->getConstantVal();

   status =
         checkInput(retinaActivity->getLayerLoc(), retinaActivity->getActivity(), retinaVal, false);
   if (status != 0) {
      Fatal().printf("[%d]: test_constant_input: ERROR in retina data\n", rank);
   }

   BasePublisherComponent *retinaPublisher = retina->getComponentByType<BasePublisherComponent>();
   status                                  = checkInput(
         retinaPublisher->getLayerLoc(), retinaPublisher->getLayerData(), retinaVal, true);
   if (status != 0) {
      Fatal().printf("[%d]: test_constant_input: ERROR in retina data\n", rank);
   }

   delete hc;
   delete initObj;

   return status;
}

int checkInput(PVLayerLoc const *loc, const float *data, float val, bool extended) {
   int status = 0;

   const PVHalo *halo = &loc->halo;
   const int nk       = (loc->nx + halo->lt + halo->rt) * (loc->ny + halo->dn + halo->up) * loc->nf;

   for (int k = 0; k < nk; k++) {
      if (data[k] != val) {
         return -1;
      }
   }

   return status;
}

int createTestFile(const char *filename, int nTotal, float *buf) {
   int i, err = 0;
   size_t result;

   FILE *fd = fopen(filename, "wb");

   for (i = 0; i < nTotal; i++) {
      buf[i] = (float)i;
   }

   result = fwrite(buf, sizeof(float), nTotal, fd);
   fclose(fd);
   if ((int)result != nTotal) {
      ErrorLog().printf("[ ]: createTestFile: failure to write to file %s\n", filename);
   }

   fd     = fopen(filename, "rb");
   result = fread(buf, sizeof(float), nTotal, fd);
   fclose(fd);
   if ((int)result != nTotal) {
      ErrorLog().printf("[ ]: createTestFile: unable to read from file %s\n", filename);
   }

   err = 0;
   for (i = 0; i < nTotal; i++) {
      if (buf[i] != (float)i) {
         err = 1;
         ErrorLog().printf("%s file is incorrect at %d\n", filename, i);
      }
   }

   return err;
}

int printLoc(const PVLayerLoc *loc) {
   InfoLog().printf(
         "nxGlobal==%d nyGlobal==%d nx==%d ny==%d kx0==%d ky0==%d halo==(%d,%d,%d,%d) nf==%d\n",
         loc->nxGlobal,
         loc->nyGlobal,
         loc->nx,
         loc->ny,
         loc->kx0,
         loc->ky0,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         loc->nf);
   InfoLog().flush();
   return 0;
}

int checkLoc(HyPerCol *hc, const PVLayerLoc *loc) {
   int status = PV_SUCCESS;

   const int cols = hc->numCommColumns();
   const int rows = hc->numCommRows();

   if (loc->nxGlobal != loc->nx * cols) {
      status = PV_FAILURE;
   }
   if (loc->nyGlobal != loc->ny * rows) {
      status = PV_FAILURE;
   }

   if (loc->kx0 != loc->nx * hc->commColumn()) {
      status = PV_FAILURE;
   }
   if (loc->ky0 != loc->ny * hc->commRow()) {
      status = PV_FAILURE;
   }

   return status;
}
