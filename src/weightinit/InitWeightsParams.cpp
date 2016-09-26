/*
 * InitWeightsParams.cpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#include "InitWeightsParams.hpp"

#include <stdlib.h>

#include "include/default_params.h"
#include "io/io.hpp"
#include "io/fileio.hpp"
#include "utils/conversions.h"
#include "columns/Communicator.hpp"

namespace PV {

InitWeightsParams::InitWeightsParams()
{
   initialize_base();
}
InitWeightsParams::InitWeightsParams(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

InitWeightsParams::~InitWeightsParams()
{
   free(this->name);
   free(this->filename);
}

int InitWeightsParams::initialize_base() {
   this->parent = NULL;
   this->pre = NULL;
   this->post = NULL;
   this->channel = CHANNEL_EXC;
   this->name = strdup("Unknown");
   this->filename = NULL;
   this->useListOfArborFiles = false;
   this->combineWeightFiles = false;
   return PV_SUCCESS;
}

int InitWeightsParams::getnfPatch() {
   return parentConn->fPatchSize();
}

int InitWeightsParams::getnyPatch() {
   return parentConn->yPatchSize();
}

int InitWeightsParams::getnxPatch() {
   return parentConn->xPatchSize();
}

int InitWeightsParams::getPatchSize() {
   return parentConn->fPatchSize()*parentConn->xPatchSize()*parentConn->yPatchSize();
}

int InitWeightsParams::getsx() {
   return parentConn->xPatchStride();
}

int InitWeightsParams::getsy() {
   return parentConn->yPatchStride();
}

int InitWeightsParams::getsf() {
   return parentConn->fPatchStride();
}

float InitWeightsParams::getWMin() {
   return parentConn->getWMin();
}

float InitWeightsParams::getWMax() {
   return parentConn->getWMax();
}

int InitWeightsParams::initialize(char const * name, HyPerCol * hc) {
   int status = PV_SUCCESS;

   this->parentConn = NULL;
   this->parent = hc;
   this->setName(name);

   return status;
}

int InitWeightsParams::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   // Read/write any params from the params file, typically
   // parent->parameters()->ioParamValue(ioFlag, name, "param_name", &param, default_value);
   parentConn = dynamic_cast<HyPerConn *>(parent->getConnFromName(name));
   ioParam_initWeightsFile(ioFlag);
   ioParam_useListOfArborFiles(ioFlag);
   ioParam_combineWeightFiles(ioFlag);
   ioParam_numWeightFiles(ioFlag);
   return PV_SUCCESS;
}

void InitWeightsParams::ioParam_initWeightsFile(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(ioFlag, name, "initWeightsFile", &filename, NULL, false/*warnIfAbsent*/);
}

void InitWeightsParams::ioParam_useListOfArborFiles(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "initWeightsFile"));
   if (filename!=NULL) {
      parent->parameters()->ioParamValue(ioFlag, name, "useListOfArborFiles", &useListOfArborFiles, false/*default*/, true/*warnIfAbsent*/);
   }
}

void InitWeightsParams::ioParam_combineWeightFiles(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "initWeightsFile"));
   if (filename!=NULL) {
      parent->parameters()->ioParamValue(ioFlag, name, "combineWeightFiles", &combineWeightFiles, false/*default*/, true/*warnIfAbsent*/);
   }
}

void InitWeightsParams::ioParam_numWeightFiles(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "initWeightsFile"));
   if (filename!=NULL) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "combineWeightFiles"));
      if (combineWeightFiles) {
         int max_weight_files = 1;  // arbitrary limit...
         parent->parameters()->ioParamValue(ioFlag, name, "numWeightFiles", &numWeightFiles, max_weight_files, true/*warnIfAbsent*/);
      }
   }
}

int InitWeightsParams::communicateParamsInfo() {
   // to be called during communicateInitInfo stage;
   // set any member variables that depend on other objects
   // having been initialized or communicateInitInfo'd
   assert(parentConn != NULL);
   this->pre = parentConn->getPre();
   this->post = parentConn->getPost();
   assert(this->pre && this->post);
   this->channel = parentConn->getChannel();

   return PV_SUCCESS;
}

void InitWeightsParams::calcOtherParams(int dataPatchIndex) {
   this->getcheckdimensionsandstrides();
   kernelIndexCalculations(dataPatchIndex);
}

void InitWeightsParams::getcheckdimensionsandstrides() {
}

int InitWeightsParams::kernelIndexCalculations(int dataPatchIndex) {
   //kernel index stuff:
   int kxKernelIndex;
   int kyKernelIndex;
   int kfKernelIndex;
   parentConn->dataIndexToUnitCellIndex(dataPatchIndex, &kxKernelIndex, &kyKernelIndex, &kfKernelIndex);
   const int kxPre_tmp = kxKernelIndex;
   const int kyPre_tmp = kyKernelIndex;
   const int kfPre_tmp = kfKernelIndex;

   // get distances to nearest neighbor in post synaptic layer (meaured relative to pre-synatpic cell)
   float xDistNNPreUnits;
   float xDistNNPostUnits;
   dist2NearestCell(kxPre_tmp, pre->getXScale(), post->getXScale(),
         &xDistNNPreUnits, &xDistNNPostUnits);
   float yDistNNPreUnits;
   float yDistNNPostUnits;
   dist2NearestCell(kyPre_tmp, pre->getYScale(), post->getYScale(),
         &yDistNNPreUnits, &yDistNNPostUnits);

   // get indices of nearest neighbor
   int kxNN;
   int kyNN;
   kxNN = nearby_neighbor( kxPre_tmp, pre->getXScale(), post->getXScale());
   kyNN = nearby_neighbor( kyPre_tmp, pre->getYScale(), post->getYScale());

   // get indices of patch head
   int kxHead;
   int kyHead;
   kxHead = zPatchHead(kxPre_tmp, parentConn->xPatchSize(), pre->getXScale(), post->getXScale());
   kyHead = zPatchHead(kyPre_tmp, parentConn->yPatchSize(), pre->getYScale(), post->getYScale());

   // get distance to patch head (measured relative to pre-synaptic cell)
   float xDistHeadPostUnits;
   xDistHeadPostUnits = xDistNNPostUnits + (kxHead - kxNN);
   float yDistHeadPostUnits;
   yDistHeadPostUnits = yDistNNPostUnits + (kyHead - kyNN);
   float xRelativeScale = xDistNNPreUnits == xDistNNPostUnits ? 1.0f : xDistNNPreUnits
         / xDistNNPostUnits;
   xDistHeadPreUnits = xDistHeadPostUnits * xRelativeScale;
   float yRelativeScale = yDistNNPreUnits == yDistNNPostUnits ? 1.0f : yDistNNPreUnits
         / yDistNNPostUnits;
   yDistHeadPreUnits = yDistHeadPostUnits * yRelativeScale;


   // sigma is in units of pre-synaptic layer
   dxPost = xRelativeScale;
   dyPost = yRelativeScale;

   return kfPre_tmp;
}

float InitWeightsParams::calcYDelta(int jPost) {
   return calcDelta(jPost, dyPost, yDistHeadPreUnits);
}

float InitWeightsParams::calcXDelta(int iPost) {
   return calcDelta(iPost, dxPost, xDistHeadPreUnits);
}

float InitWeightsParams::calcDelta(int post, float dPost, float distHeadPreUnits) {
   return distHeadPreUnits + post * dPost;
}

} /* namespace PV */
