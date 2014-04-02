/*
 * LayerProbe.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "LayerProbe.hpp"
#include "../layers/HyPerLayer.hpp"

namespace PV {

LayerProbe::LayerProbe()
{
   initLayerProbe_base();
   // Derived classes of LayerProbe should call LayerProbe::initialize themselves.
}

/**
 * @filename
 */
LayerProbe::LayerProbe(const char * probeName, HyPerCol * hc)
{
   initLayerProbe_base();
   initLayerProbe(probeName, hc);
}

LayerProbe::~LayerProbe()
{
   if (outputstream != NULL) {
      PV_fclose(outputstream); outputstream = NULL;
   }
   free(targetLayerName); targetLayerName = NULL;
   free(msgparams); msgparams = NULL;
   free(msgstring); msgstring = NULL;
   free(probeOutputFilename); probeOutputFilename = NULL;
   if(triggerLayerName){
      free(triggerLayerName);
      triggerLayerName = NULL;
   }
}

int LayerProbe::initLayerProbe_base() {
   probeName = NULL;
   parentCol = NULL;
   outputstream = NULL;
   targetLayerName = NULL;
   targetLayer = NULL;
   msgparams = NULL;
   msgstring = NULL;
   probeOutputFilename = NULL;
   triggerFlag = false;
   triggerLayerName = NULL;
   triggerLayer = NULL;
   triggerOffset = 0;
   return PV_SUCCESS;
}

/**
 * @filename
 * @layer
 */
int LayerProbe::initLayerProbe(const char * probeName, HyPerCol * hc)
{
   setParentCol(hc);
   setProbeName(probeName);
   ioParams(PARAMS_IO_READ);
   parentCol->addLayerProbe(this); // Can't call HyPerLayer::insertProbe yet because HyPerLayer is not known to be instantiated until the communicateInitInfo stage
   return PV_SUCCESS;
}

int LayerProbe::setProbeName(const char * probeName) {
   assert(this->probeName == NULL);
   this->probeName = strdup(probeName);
   if (this->probeName == NULL) {
      assert(parentCol!=NULL);
      fprintf(stderr,"LayerProbe \"%s\" unable to set probeName on rank %d: %s\n",
            probeName, parentCol->columnId(), strerror(errno));
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int LayerProbe::ioParams(enum ParamsIOFlag ioFlag) {
   parentCol->ioParamsStartGroup(ioFlag, probeName);
   ioParamsFillGroup(ioFlag);
   parentCol->ioParamsFinishGroup(ioFlag);
   return PV_SUCCESS;
}

int LayerProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_targetLayer(ioFlag);
   ioParam_message(ioFlag);
   ioParam_probeOutputFile(ioFlag);
   ioParam_triggerFlag(ioFlag);
   ioParam_triggerLayerName(ioFlag);
   ioParam_triggerOffset(ioFlag);
   return PV_SUCCESS;
}

void LayerProbe::ioParam_targetLayer(enum ParamsIOFlag ioFlag) {
   parentCol->ioParamStringRequired(ioFlag, probeName, "targetLayer", &targetLayerName);
}

void LayerProbe::ioParam_message(enum ParamsIOFlag ioFlag) {
   parentCol->ioParamString(ioFlag, probeName, "message", &msgparams, NULL, false/*warnIfAbsent*/);
   if (ioFlag == PARAMS_IO_READ) {
      initMessage(msgparams);
   }
}

void LayerProbe::ioParam_probeOutputFile(enum ParamsIOFlag ioFlag) {
   parentCol->ioParamString(ioFlag, probeName, "probeOutputFile", &probeOutputFilename, NULL, false/*warnIfAbsent*/);
}

void LayerProbe::ioParam_triggerFlag(enum ParamsIOFlag ioFlag) {
   parentCol->ioParamValue(ioFlag, probeName, "triggerFlag", &triggerFlag, triggerFlag);
}

void LayerProbe::ioParam_triggerLayerName(enum ParamsIOFlag ioFlag) {
   assert(!parentCol->parameters()->presentAndNotBeenRead(probeName, "triggerFlag"));
   if (triggerFlag) {
      parentCol->ioParamStringRequired(ioFlag, probeName, "triggerLayerName", &triggerLayerName);
   }
}

void LayerProbe::ioParam_triggerOffset(enum ParamsIOFlag ioFlag) {
   assert(!parentCol->parameters()->presentAndNotBeenRead(probeName, "triggerFlag"));
   if (triggerFlag) {
      parentCol->ioParamValue(ioFlag, probeName, "triggerOffset", &triggerOffset, triggerOffset);
      if(triggerOffset < 0){
         fprintf(stderr, "%s \"%s\" error in rank %d process: TriggerOffset (%f) must be positive\n", parentCol->parameters()->groupKeywordFromName(probeName), probeName, parentCol->columnId(), triggerOffset);
         exit(EXIT_FAILURE);
      }
   }
}

int LayerProbe::initOutputStream(const char * filename) {
   if( parentCol->columnId()==0 ) {
      if( filename != NULL ) {
         char * outputdir = parentCol->getOutputPath();
         char * path = (char *) malloc(strlen(outputdir)+1+strlen(filename)+1);
         sprintf(path, "%s/%s", outputdir, filename);
         bool append = parentCol->getCheckpointReadFlag();
         const char * fopenstring = append ? "a" : "w";
         outputstream = PV_fopen(path, fopenstring);
         if( !outputstream ) {
            fprintf(stderr, "LayerProbe error opening \"%s\" for writing: %s\n", path, strerror(errno));
            exit(EXIT_FAILURE);
         }
         free(path);
      }
      else {
         outputstream = PV_stdout();
      }
   }
   else {
      outputstream = NULL; // Only root process writes; if other processes need something written it should be sent to root.
                           // Derived classes for which it makes sense for a different process to do the file i/o should override initOutputStream
   }
   return PV_SUCCESS;
}

int LayerProbe::communicateInitInfo() {
   int status = setTargetLayer(targetLayerName);
   if (status == PV_SUCCESS) {
      status = initOutputStream(probeOutputFilename);
   }
   if (status == PV_SUCCESS) {
      targetLayer->insertProbe(this);
   }
   if(triggerFlag){
      triggerLayer = parentCol->getLayerFromName(triggerLayerName);
      if (triggerLayer==NULL) {
         if (parentCol->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: triggerLayer \"%s\" is not a layer in the HyPerCol.\n",
                  parentCol->parameters()->groupKeywordFromName(probeName), probeName, triggerLayerName);
         }
#if PV_USE_MPI
         MPI_Barrier(parentCol->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }
   }
   return status;
}

int LayerProbe::setTargetLayer(const char * layerName) {
   targetLayer = parentCol->getLayerFromName(layerName);
   if (targetLayer==NULL) {
      if (parentCol->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: targetLayer \"%s\" is not a layer in the column.\n",
               parentCol->parameters()->groupKeywordFromName(probeName), probeName, targetLayerName);
      }
#if PV_USE_MPI
      MPI_Barrier(parentCol->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int LayerProbe::initMessage(const char * msg) {
   assert(msgstring==NULL);
   int status = PV_SUCCESS;
   if( msg != NULL && msg[0] != '\0' ) {
      size_t msglen = strlen(msg);
      this->msgstring = (char *) calloc(msglen+2, sizeof(char)); // Allocate room for colon plus null terminator
      if(this->msgstring) {
         memcpy(this->msgstring, msg, msglen);
         this->msgstring[msglen] = ':';
         this->msgstring[msglen+1] = '\0';
      }
   }
   else {
      this->msgstring = (char *) calloc(1, sizeof(char));
      if(this->msgstring) {
         this->msgstring[0] = '\0';
      }
   }
   if( !this->msgstring ) {
      fprintf(stderr, "%s \"%s\": Unable to allocate memory for probe's message.\n",
            parentCol->parameters()->groupKeywordFromName(probeName), probeName);
      status = PV_FAILURE;
   }
   assert(status == PV_SUCCESS);
   return status;
}

bool LayerProbe::needUpdate(double time, double dt){
   if(triggerFlag){
      assert(triggerLayer);

      //Need to use choose between lastUpdateTime or nextUpdateTime, since nextUpdateTime will already be updated if offset is 0
      double updateTime;
      if(fabs(triggerLayer->getLastUpdateTime() - time) <= dt/2){
         updateTime = triggerLayer->getLastUpdateTime();
      }
      else{
         updateTime = triggerLayer->getNextUpdateTime();
      }
      //never update flag
      if(updateTime == -1){
         return false;
      }
      //Check for equality
      if(fabs(time - (updateTime - triggerOffset)) < (dt/2)){
         return true;
      }
      //If it gets to this point, don't update
      return false;
   }
   //If no trigger, weightUpdateTime
   else{
      return true;
   }
}

int LayerProbe::outputStateWrapper(double timef, double dt){
   int status = PV_SUCCESS;
   if(needUpdate(timef, dt)){
      status = outputState(timef);
   }
   return status;
}

/**
 * @time
 */
int LayerProbe::outputState(double timef)
{
   return 0;
}

} // namespace PV
