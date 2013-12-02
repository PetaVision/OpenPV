/*
 * ReciprocalConn.cpp
 *
 *  For two connections with transpose geometries (the pre of one has the
 *  same geometry as the post of another and vice versa).  They are not
 *  forced to be each other's transpose, but there is a penalty term
 *  on the L2 norm of the difference between it and its reciprocal's
 *  transpose, of the amount
 *  r/2 * || W/f_W - (W_recip)'/f_{W_recip} ||^2.
 *
 *  r is the parameter reciprocalFidelityCoeff
 *  W is the weights and W_recip is the weights of the reciprocal connection
 *  f_W and f_{W_recip} are the number of features of each connection.
 *
 *  The connection forces the normalization method to be that for each feature,
 *  the sum across kernels (number of features of reciprocal connection) is unity.
 *
 *  Dividing by f_W and f_{W_recip} therefore makes the sum of all matrix
 *  entries of each connection 1.
 *
 *  This is still under the assumption that nxp = nyp = 1.
 *
 *  There is also the ability to specify an additional penalty of the form
 *  dW/dt += slownessPost * slownessPre', by setting slownessFlag to true
 *  and specifying slownessPre and slownessPost layers.
 *
 *  The name comes from the motivation of including a slowness term on
 *  ReciprocalConns, but the slowness interpretation is not within ReciprocalConn.
 *  IncrementLayer was written so that setting A to an IncrementLayer, and
 *  having a layer B be the post of a clone of the ReciprocalConn, setting
 *  slownessPre to A and slownessPost to B would implement a slowness term.
 *
 *  Created on: Feb 16, 2012
 *      Author: pschultz
 */

#include "ReciprocalConn.hpp"

namespace PV {

ReciprocalConn::ReciprocalConn() {
   initialize_base();

}

ReciprocalConn::ReciprocalConn(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, InitWeights * weightInit) {
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
}

int ReciprocalConn::initialize_base() {
   updateRulePreName = NULL;
   updateRulePostName = NULL;
   updateRulePre = NULL;
   updateRulePost = NULL;
   reciprocalWgtsName = NULL;
   reciprocalWgts = NULL;
   slownessPreName = NULL;
   slownessPostName = NULL;
   slownessPre = NULL;
   slownessPost = NULL;
   return PV_SUCCESS;
}

int ReciprocalConn::initialize(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, InitWeights * weightInit) {
   int status = KernelConn::initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);

   return status;
}

int ReciprocalConn::setParams(PVParams * params) {
   int status = KernelConn::setParams(params);
   readRelaxationRate(params);
   readReciprocalFidelityCoeff(params);
   status = readUpdateRulePre(params)==PV_SUCCESS ? status : PV_FAILURE;
   status = readUpdateRulePost(params)==PV_SUCCESS ? status : PV_FAILURE;
   readSlownessFlag(params);
   status = readSlownessPre(params)==PV_SUCCESS ? status : PV_FAILURE;
   status = readSlownessPost(params)==PV_SUCCESS ? status : PV_FAILURE;
   status = readReciprocalWgts(params)==PV_SUCCESS ? status : PV_FAILURE;
   MPI_Barrier(parent->icCommunicator()->communicator());
   if (status != PV_SUCCESS) exit(EXIT_FAILURE);
   if( status != PV_SUCCESS ) abort();
   return status;
}

void ReciprocalConn::readRelaxationRate(PVParams * params) {
   relaxationRate = params->value(name, "relaxationRate", 1.0f);
}

void ReciprocalConn::readReciprocalFidelityCoeff(PVParams * params) {
   reciprocalFidelityCoeff = params->value(name, "reciprocalFidelityCoeff", 1.0f);
}

int ReciprocalConn::readUpdateRulePre(PVParams * params) {
   return getLayerName(params, "updateRulePre", &updateRulePreName, preLayerName);
}

int ReciprocalConn::readUpdateRulePost(PVParams * params) {
   return getLayerName(params, "updateRulePost", &updateRulePostName, postLayerName);
}

void ReciprocalConn::readSlownessFlag(PVParams * params) {
   slownessFlag = params->value(name, "slownessFlag", false)!=0;
}

int ReciprocalConn::readSlownessPre(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "slownessFlag"));
   return getLayerName(params, "slownessPre", &slownessPreName, NULL);
}

int ReciprocalConn::readSlownessPost(PVParams * params) {
   int status = PV_SUCCESS;
   assert(!params->presentAndNotBeenRead(name, "slownessFlag"));
   return getLayerName(params, "slownessPost", &slownessPostName, NULL);
   return status;
}

int ReciprocalConn::getLayerName(PVParams * params, const char * parameter_name, char ** layer_name_ptr, const char * default_name) {
   int status = PV_SUCCESS;
   assert(*layer_name_ptr==NULL);
   if (slownessFlag) {
      const char * slowness_pre_name = params->stringValue(name, parameter_name);
      if (slowness_pre_name==NULL) {
         if (default_name != NULL) {
            *layer_name_ptr = strdup(default_name);
            if (slownessPreName==NULL) {
               fprintf(stderr, "ReciprocalConn \"%s\" error: unable to allocate memory for name of %s layer.\n", name, parameter_name);
               status = PV_FAILURE;
            }
         }
         else {
            fprintf(stderr, "ReciprocalConn \"%s\" error: parameter \"%s\" must be defined.\n", name, parameter_name);
            status = PV_FAILURE;
         }
      }
      else {
         *layer_name_ptr = strdup(slowness_pre_name);
         if (slownessPreName==NULL) {
            fprintf(stderr, "ReciprocalConn \"%s\" error: unable to allocate memory for name of %s layer.\n", name, parameter_name);
            status = PV_FAILURE;
         }
      }
   }
   return status;
}

int ReciprocalConn::readReciprocalWgts(PVParams * params) {
   reciprocalWgtsName = params->stringValue(name, "reciprocalWgts");
   int status = PV_SUCCESS;
   if( reciprocalWgtsName == NULL || reciprocalWgtsName[0] == '0') {
      fprintf(stderr, "ReciprocalConn \"%s\": reciprocalWgts must be defined.\n", name);
      status = PV_FAILURE;
   }
   return status;
}

int ReciprocalConn::communicateInitInfo() {
   int status = KernelConn::communicateInitInfo();
   status = setParameterLayer("updateRulePre", updateRulePreName, &updateRulePre)==PV_SUCCESS ? status : PV_FAILURE;
   status = setParameterLayer("updateRulePost", updateRulePostName, &updateRulePost)==PV_SUCCESS ? status : PV_FAILURE;
   status = setParameterLayer("slownessPre", slownessPreName, &slownessPre)==PV_SUCCESS ? status : PV_FAILURE;
   status = setParameterLayer("slownessPost", slownessPostName, &slownessPost)==PV_SUCCESS ? status : PV_FAILURE;
   status = setReciprocalWgts(reciprocalWgtsName)==PV_SUCCESS ? status : PV_FAILURE;
   return status;
}

int ReciprocalConn::setParameterLayer(const char * paramname, const char * layername, HyPerLayer ** layerPtr) {
   int status = PV_SUCCESS;
   assert(layername);
   *layerPtr = parent->getLayerFromName(layername);
   if( *layerPtr == NULL ) {
      fprintf(stderr, "ReciprocalConn \"%s\" error: parameter \"%s\" has value \"%s\", but this not the name of a layer.\n", name, paramname, layername);
      status = PV_FAILURE;
   }
   return status;
}

int ReciprocalConn::initNormalize() {
   // PVParams * params = parent->parameters();
   nxUnitCellPost = zUnitCellSize(postSynapticLayer()->getXScale(), preSynapticLayer()->getXScale());
   nyUnitCellPost = zUnitCellSize(postSynapticLayer()->getYScale(), preSynapticLayer()->getYScale());
   nfUnitCellPost = fPatchSize();
   sizeUnitCellPost = nxUnitCellPost*nyUnitCellPost*nfUnitCellPost;

   return PV_SUCCESS;
}

int ReciprocalConn::setReciprocalWgts(const char * recipName) {
   int status = PV_SUCCESS;
   if( status == PV_SUCCESS && recipName == NULL ) {
      status = PV_FAILURE;
      fprintf(stderr, "ReciprocalConn \"%s\": setReciprocalWgts called with null argument.\n", name);
   }

   HyPerConn * c;
   if( status == PV_SUCCESS ) {
      c = parent->getConnFromName(recipName);
      if( c == NULL) {
         status = PV_FAILURE;
         fprintf(stderr, "ReciprocalConn \"%s\": reciprocalWgts \"%s\" could not be found.\n", name, recipName);
      }
   }
   if( status == PV_SUCCESS && reciprocalWgts != NULL) {
      if(c != reciprocalWgts) {
         fprintf(stderr, "ReciprocalConn \"%s\": setReciprocalWgts called with reciprocalWgts already set.\n", name);
         status = PV_FAILURE;
      }
   }
   if( status == PV_SUCCESS && reciprocalWgts == NULL ) {
      reciprocalWgts = dynamic_cast<ReciprocalConn *>(parent->getConnFromName(recipName));
      if( reciprocalWgts == NULL ) {
         fprintf(stderr, "ReciprocalConn \"%s\": reciprocalWgts \"%s\" is not a ReciprocalConn.\n", name, recipName);
         status = PV_FAILURE;
      }
   }
   return status;
}

int ReciprocalConn::update_dW(int axonID) {
   int nExt = preSynapticLayer()->getNumExtended();
   int numKernelIndices = getNumDataPatches();
   int delay = getDelay(axonID);
   const pvdata_t * preactbuf = updateRulePre->getLayerData(delay);
   const pvdata_t * postactbuf = updateRulePost->getLayerData(delay);
   const pvdata_t * slownessprebuf = getSlownessFlag() ? slownessPre->getLayerData(delay) : NULL;
   const pvdata_t * slownesspostbuf = getSlownessFlag() ? slownessPost->getLayerData(delay) : NULL;

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));
   for(int kExt=0; kExt<nExt;kExt++) {
      PVPatch * weights = getWeights(kExt,axonID);
      size_t offset = getAPostOffset(kExt, axonID);
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      pvdata_t preact = preactbuf[kExt];
      const pvdata_t * postactRef = &postactbuf[offset];
      pvdata_t * dwdata = get_dwData(axonID, kExt);
      int lineoffsetw = 0;
      int lineoffseta = 0;
      for( int y=0; y<ny; y++ ) {
         for( int k=0; k<nk; k++ ) {
            dwdata[lineoffsetw + k] += updateRule_dW(preact, postactRef[lineoffseta+k]);
         }
         lineoffsetw += syp;
         lineoffseta += sya;
      }
      if( slownessFlag ) {
         preact = slownessprebuf[kExt];
         postactRef = &slownesspostbuf[offset];

         int lineoffsetw = 0;
         int lineoffseta = 0;
         for( int y=0; y<ny; y++ ) {
            for( int k=0; k<nk; k++ ) {
               dwdata[lineoffsetw + k] += updateRule_dW(preact, postactRef[lineoffseta+k]);
            }
            lineoffsetw += syp;
            lineoffseta += sya;
         }
      }
   }
   if( reciprocalFidelityCoeff ) {
      for( int k=0; k<numKernelIndices; k++) {
         const pvdata_t * wdata = get_wDataHead(axonID, k);
         pvdata_t * dwdata = get_dwDataHead(axonID, k);
         int n=0;
         for( int y=0; y<nyp; y++ ) { for( int x=0; x<nxp; x++) { for( int f=0; f<nfp; f++ ) {
            int xRecip, yRecip, fRecip, kRecip;
            getReciprocalWgtCoordinates(x, y, f, k, &xRecip, &yRecip, &fRecip, &kRecip);
            const pvdata_t * recipwdata = reciprocalWgts->get_wDataHead(axonID, kRecip);
            int nRecip = kIndex(xRecip, yRecip, fRecip, reciprocalWgts->xPatchSize(), reciprocalWgts->yPatchSize(), reciprocalWgts->fPatchSize());
            dwdata[n] -= reciprocalFidelityCoeff/nfp*(wdata[n]/nfp-recipwdata[nRecip]/reciprocalWgts->fPatchSize());
            n++;
         }}}
      }
   }

   // Divide by (numNeurons/numKernels)
   int divisor = pre->getNumNeurons()/numKernelIndices;
   assert( divisor*numKernelIndices == pre->getNumNeurons() );
   for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
      int numpatchitems = nxp * nyp * nfp;
      pvdata_t * dwpatchdata = get_dwDataHead(axonID,kernelindex);
      for( int n=0; n<numpatchitems; n++ ) {
         dwpatchdata[n] /= divisor;
      }
   }

   lastUpdateTime = parent->simulationTime();
   return PV_SUCCESS;
}

int ReciprocalConn::updateWeights(int arborID) {
   lastUpdateTime = parent->simulationTime();
   // add dw to w
   for( int k=0; k<nxp*nyp*nfp*getNumDataPatches(); k++ ) {
      get_wDataStart(arborID)[k] += relaxationRate*parent->getDeltaTime()*get_dwDataStart(arborID)[k];
   }
   int status = PV_SUCCESS;
   pvdata_t * arborstart = get_wDataStart(arborID);
   for( int k=0; k<nxp*nyp*nfp*getNumDataPatches(); k++ ) {
      if( arborstart[k] < 0 ) arborstart[k] = 0;
   }
   return status;
}

ReciprocalConn::~ReciprocalConn() {
   free(updateRulePreName);  updateRulePreName = NULL;
   free(updateRulePostName); updateRulePostName = NULL;
   free(slownessPreName);    slownessPreName = NULL;
   free(slownessPostName);   slownessPostName = NULL;
}

} /* namespace PV */
