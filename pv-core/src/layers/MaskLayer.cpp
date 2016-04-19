
/*
 * MaskLayer.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: slundquist 
 */

#include "MaskLayer.hpp"

namespace PV {

MaskLayer::MaskLayer(const char * name, HyPerCol * hc){
   initialize_base();
   initialize(name, hc);
}

MaskLayer::MaskLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

MaskLayer::~MaskLayer(){
   if(maskLayerName){
      free(maskLayerName);
   }
   if(features){
      free(features);
   }
   if(maskMethod){
      free(maskMethod);
   }
}

int MaskLayer::initialize_base(){
   maskLayerName = NULL;
   maskLayer = NULL;
   maskMethod = NULL;
   features = NULL;

   return PV_SUCCESS;
}

int MaskLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_maskMethod(ioFlag);
   ioParam_maskLayerName(ioFlag);
   ioParam_featureIdxs(ioFlag);
   return status;
}

void MaskLayer::ioParam_maskMethod(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "maskMethod", &maskMethod);
   //Check valid methods
   if(strcmp(maskMethod, "layer") == 0){
   }
   else if(strcmp(maskMethod, "maskFeatures") == 0){
   }
   else if(strcmp(maskMethod, "noMaskFeatures") == 0){
   }
   else{
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: \"%s\" is not a valid maskMethod. Options are \"layer\", \"maskFeatures\", or \"noMaskFeatures\".\n",
                 getKeyword(), name, maskMethod);
      }
      exit(-1);
   }
}

void MaskLayer::ioParam_maskLayerName(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "maskMethod"));
   if(strcmp(maskMethod, "layer") == 0){
      parent->ioParamStringRequired(ioFlag, name, "maskLayerName", &maskLayerName);
   }
}

void MaskLayer::ioParam_featureIdxs(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "maskMethod"));
   if(strcmp(maskMethod, "maskFeatures") == 0 || strcmp(maskMethod, "noMaskFeatures") == 0){
      parent->ioParamArray(ioFlag, name, "featureIdxs", &features, &numSpecifiedFeatures);
      if(numSpecifiedFeatures == 0){
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: MaskLayer must specify at least one feature for maskMethod \"%s\".\n",
                    getKeyword(), name, maskMethod);
         }
         exit(-1);
      }
   }
}

int MaskLayer::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();
   if(strcmp(maskMethod, "layer") == 0){
      maskLayer = parent->getLayerFromName(maskLayerName);
      if (maskLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" is not a layer in the HyPerCol.\n",
                    getKeyword(), name, maskLayerName);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      const PVLayerLoc * maskLoc = maskLayer->getLayerLoc();
      const PVLayerLoc * loc = getLayerLoc();
      assert(maskLoc != NULL && loc != NULL);
      if (maskLoc->nxGlobal != loc->nxGlobal || maskLoc->nyGlobal != loc->nyGlobal) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" does not have the same x and y dimensions.\n",
                    getKeyword(), name, maskLayerName);
            fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                    maskLoc->nxGlobal, maskLoc->nyGlobal, maskLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      if(maskLoc->nf != 1 && maskLoc->nf != loc->nf){
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" must either have the same number of features as this layer, or one feature.\n",
                    getKeyword(), name, maskLayerName);
            fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                    maskLoc->nxGlobal, maskLoc->nyGlobal, maskLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      assert(maskLoc->nx==loc->nx && maskLoc->ny==loc->ny);
   }
   else{
      //Check for in bounds featureIdxs
      assert(features);
      const PVLayerLoc * loc = getLayerLoc();
      for(int f = 0; f < numSpecifiedFeatures; f++){
         if(features[f] < 0 || features[f] >= loc->nf){
            std::cout << "Specified feature " << features[f] << "out of bounds\n"; 
            exit(-1);
         }
         
      }
   }

   return status;
}

int MaskLayer::updateState(double time, double dt)
{
   ANNLayer::updateState(time, dt);
   const PVLayerLoc * loc = getLayerLoc();
   pvdata_t * A = getActivity();

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int nbatch = loc->nbatch;

   for(int b = 0; b < nbatch; b++){
      pvdata_t * ABatch = A + b * getNumExtended();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for(int ni = 0; ni < num_neurons; ni++){
         int kThisRes = ni;
         int kThisExt = kIndexExtended(ni, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         float maskVal = 1;
         if(strcmp(maskMethod, "layer") == 0){
            const PVLayerLoc * maskLoc = maskLayer->getLayerLoc();
            pvdata_t * maskActivity = maskLayer->getActivity();
            pvdata_t * maskActivityBatch = maskActivity + b * maskLayer->getNumExtended();
            int kMaskRes;
            if(maskLoc->nf == 1){
               kMaskRes = ni/nf;
            }
            else{
               kMaskRes = ni;
            }
            int kMaskExt = kIndexExtended(kMaskRes, nx, ny, maskLoc->nf, maskLoc->halo.lt, maskLoc->halo.rt, maskLoc->halo.dn, maskLoc->halo.up);
            maskVal = maskActivityBatch[kMaskExt];
         }
         else if(strcmp(maskMethod, "maskFeatures") == 0){
            //Calculate feature index of ni
            int featureNum = featureIndex(ni, nx, ny, nf);
            maskVal = 1; //If nothing specified, copy everything
            for(int specF = 0; specF < numSpecifiedFeatures; specF++){ 
               if(featureNum == features[specF]){
                  maskVal = 0;
                  break;
               }
            }
         }
         else if(strcmp(maskMethod, "noMaskFeatures") == 0){
            //Calculate feature index of ni
            int featureNum = featureIndex(ni, nx, ny, nf);
            maskVal = 0; //If nothing specified, copy nothing 
            for(int specF = 0; specF < numSpecifiedFeatures; specF++){ 
               if(featureNum == features[specF]){
                  maskVal = 1;
                  break;
               }
            }
         }

         //Set value to 0, otherwise, updateState from ANNLayer should have taken care of it
         if(maskVal == 0){
            ABatch[kThisExt] = 0;
         }
      }
   }
   return PV_SUCCESS;
}

BaseObject * createMaskLayer(char const * name, HyPerCol * hc) {
   return hc ? new MaskLayer(name, hc) : NULL;
}

} /* namespace PV */
