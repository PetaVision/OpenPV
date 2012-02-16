/*
 * InitWeightsParams.cpp
 *
 *  Created on: Aug 10, 2011
 *      Author: kpeterson
 */

#include "InitWeightsParams.hpp"

#include <stdlib.h>

#include "../include/default_params.h"
#include "../io/ConnectionProbe.hpp"
#include "../io/io.h"
#include "../io/fileio.hpp"
#include "../utils/conversions.h"
#include "../utils/pv_random.h"
#include "../columns/InterColComm.hpp"

namespace PV {

InitWeightsParams::InitWeightsParams()
{
   initialize_base();
}
InitWeightsParams::InitWeightsParams(HyPerConn * parentConn) {
   initialize_base();
   initialize(parentConn);
}

InitWeightsParams::~InitWeightsParams()
{
   // TODO Auto-generated destructor stub
}


int InitWeightsParams::initialize_base() {
   this->parent = NULL;
   this->pre = NULL;
   this->post = NULL;
   this->channel = CHANNEL_EXC;
   this->name = strdup("Unknown");

   deltaTheta=0;

   return 1;
}
int InitWeightsParams::initialize(HyPerConn * parentConn) {
   int status = PV_SUCCESS;

   this->parentConn = parentConn;
   this->parent = parentConn->getParent();
   this->pre = parentConn->getPre();
   this->post = parentConn->getPost();
   this->channel = parentConn->getChannel();
   this->setName(parentConn->getName());

   return status;

}

void InitWeightsParams::getcheckdimensionsandstrides(PVPatch * patch) {
   // get/check dimensions and strides of full sized temporary patch
   nxPatch_tmp = patch->nx;
   nyPatch_tmp = patch->ny;
   nfPatch_tmp = parentConn->fPatchSize(); //patch->nf;

   sx_tmp = parentConn->xPatchStride(); //patch->sx;
   assert(sx_tmp == parentConn->fPatchSize()); // patch->nf);
   sy_tmp = parentConn->yPatchStride(); //patch->sy;
   assert(sy_tmp == parentConn->fPatchSize()*parentConn->xPatchSize()); //patch->nf * patch->nx);
   sf_tmp = parentConn->fPatchStride(); //patch->sf;
   assert(sf_tmp == 1);
}

int InitWeightsParams::kernelIndexCalculations(PVPatch * patch, int patchIndex) {
   //kernel index stuff:
   int kxKernelIndex;
   int kyKernelIndex;
   int kfKernelIndex;
   parentConn->patchIndexToKernelIndex(patchIndex, &kxKernelIndex, &kyKernelIndex, &kfKernelIndex);
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
   kxHead = zPatchHead(kxPre_tmp, nxPatch_tmp, pre->getXScale(), post->getXScale());
   kyHead = zPatchHead(kyPre_tmp, nyPatch_tmp, pre->getYScale(), post->getYScale());

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
   dxPost = xRelativeScale; //powf(2, (float) post->getXScale());
   dyPost = yRelativeScale; //powf(2, (float) post->getYScale());

   return kfPre_tmp;
}

void InitWeightsParams::calculateThetas(int kfPre_tmp, int patchIndex) {
   noPost = post->getLayerLoc()->nf;
   dthPost = PI*thetaMax / (float) noPost;
   th0Post = rotate * dthPost / 2.0f;
   noPre = pre->getLayerLoc()->nf;
   const float dthPre = calcDthPre();
   const float th0Pre = calcTh0Pre(dthPre);
   fPre = patchIndex % pre->getLayerLoc()->nf;
   assert(fPre == kfPre_tmp);
   const int iThPre = patchIndex % noPre;
   thPre = th0Pre + iThPre * dthPre;
}

float InitWeightsParams::calcDthPre() {
   return PI*thetaMax / (float) noPre;
}
float InitWeightsParams::calcTh0Pre(float dthPre) {
   return rotate * dthPre / 2.0f;
}

float InitWeightsParams::calcThPost(int fPost) {
   int oPost = fPost % noPost;
   float thPost = th0Post + oPost * dthPost;
   if (noPost == 1 && noPre > 1) {
      thPost = thPre;
   }
   return thPost;
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

bool InitWeightsParams::checkTheta(float thPost) {
   //      float deltaTheta = fabsf(thetaPre - thPost);
   //      deltaTheta = (deltaTheta <= PI / 2.0) ? deltaTheta : PI - deltaTheta;
  if ((deltaTheta = fabs(thPre - thPost)) > deltaThetaMax) {
     //the following is obviously not ideal. But cocirc needs this deltaTheta:
     deltaTheta = (deltaTheta <= PI / 2.0) ? deltaTheta : PI - deltaTheta;
      return true;
   }
  deltaTheta = (deltaTheta <= PI / 2.0) ? deltaTheta : PI - deltaTheta;
   return false;
}


} /* namespace PV */
