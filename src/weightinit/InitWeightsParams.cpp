/*
 * InitWeightsParams.cpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#include "InitWeightsParams.hpp"

#include <stdlib.h>

#include "../include/default_params.h"
#include "../io/io.h"
#include "../io/fileio.hpp"
#include "../utils/conversions.h"
#include "../columns/InterColComm.hpp"

namespace PV {

InitWeightsParams::InitWeightsParams()
{
   initialize_base();
}
InitWeightsParams::InitWeightsParams(HyPerConn * pConn) {
   initialize_base();
   initialize(pConn);
}

InitWeightsParams::~InitWeightsParams()
{
   free(this->name);
}


int InitWeightsParams::initialize_base() {
   this->parent = NULL;
   this->pre = NULL;
   this->post = NULL;
   this->channel = CHANNEL_EXC;
   this->name = strdup("Unknown");
   return PV_SUCCESS;
}
int InitWeightsParams::getnfPatch_tmp()        {return parentConn->fPatchSize();}
int InitWeightsParams::getnyPatch_tmp() {
   return parentConn->yPatchSize();
}
int InitWeightsParams::getnxPatch_tmp()        {return parentConn->xPatchSize();}
int InitWeightsParams::getPatchSize_tmp()      {return parentConn->fPatchSize()*
      parentConn->xPatchSize()*parentConn->yPatchSize();}
int InitWeightsParams::getsx_tmp()        {return parentConn->xPatchStride();}
int InitWeightsParams::getsy_tmp()        {return parentConn->yPatchStride();}
int InitWeightsParams::getsf_tmp()        {return parentConn->fPatchStride();}

int InitWeightsParams::initialize(HyPerConn * pConn) {
   int status = PV_SUCCESS;

   this->parentConn = pConn;
   this->parent = parentConn->getParent();
   this->pre = parentConn->getPre();
   this->post = parentConn->getPost();
   this->channel = parentConn->getChannel();
   this->setName(parentConn->getName());

   return status;
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
