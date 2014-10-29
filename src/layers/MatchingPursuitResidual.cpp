/*
 * MatchingPursuitResidual.cpp
 *
 *  Created on: Aug 13, 2013
 *      Author: pschultz
 */

#include "MatchingPursuitResidual.hpp"

namespace PV {

MatchingPursuitResidual::MatchingPursuitResidual(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

MatchingPursuitResidual::MatchingPursuitResidual() {
   initialize_base();
}

int MatchingPursuitResidual::initialize_base() {
   syncedMovieName = NULL;
   syncedMovie = NULL;
   gSynInited = false;
   refreshPeriod = 0.0;
   nextRefreshTime = 0.0;
   excNeedsUpdate = false;
   return PV_SUCCESS;
}

int MatchingPursuitResidual::initialize(const char * name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);
   if (refreshPeriod>=0) nextRefreshTime = parent->simulationTime();
   return status;
}

int MatchingPursuitResidual::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   PVParams * params = parent->parameters();
   ioParam_syncedMovie(ioFlag);
   ioParam_refreshPeriod(ioFlag);
   return status;
}

void MatchingPursuitResidual::ioParam_syncedMovie(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "syncedMovie", &syncedMovieName, NULL);
}

void MatchingPursuitResidual::ioParam_refreshPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "syncedMovie"));
   if (syncedMovieName==NULL || syncedMovieName[0]=='\0') {
      free(syncedMovieName);
      syncedMovieName = NULL;
      parent->ioParamValue(ioFlag, name, "refreshPeriod", &refreshPeriod, parent->getDeltaTime());
   }
}

int MatchingPursuitResidual::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();

   if (syncedMovieName != NULL) {
      assert(syncedMovieName[0]);
      HyPerLayer * syncedLayer = parent->getLayerFromName(syncedMovieName);
      if (syncedLayer==NULL) {
         fprintf(stderr, "MatchingPursuitLayer \"%s\" error: syncedMovie \"%s\" is not a layer in the HyPerCol.\n",
               name, syncedMovieName);
         return(EXIT_FAILURE);
      }
      syncedMovie = dynamic_cast<Movie *>(syncedLayer);
      if (syncedMovie==NULL) {
         fprintf(stderr, "MatchingPursuitLayer \"%s\" error: syncedMovie \"%s\" is not a Movie or Movie-derived layer in the HyPerCol.\n",
               name, syncedMovieName);
         return(EXIT_FAILURE);
      }
   }

   return status;
}

int MatchingPursuitResidual::resetGSynBuffers(double timed, double dt) {
   int status = PV_SUCCESS;
   bool resetGSynFlag;
   if (syncedMovie) {
      resetGSynFlag = syncedMovie->getLastUpdateTime()>lastUpdateTime;
   }
   else {
      if (refreshPeriod >= 0 && timed >= nextRefreshTime) {
         resetGSynFlag = true;
         nextRefreshTime += refreshPeriod;
      }
      else {
         resetGSynFlag = false;
      }
   }
   if (resetGSynFlag) {
      status = ANNLayer::resetGSynBuffers(timed, dt);
      lastUpdateTime = parent->simulationTime();
      excNeedsUpdate = true;
   }
   return status;
}

int MatchingPursuitResidual::recvAllSynapticInput() {
   int status = ANNLayer::recvAllSynapticInput();
   gSynInited = true;
   excNeedsUpdate = false;
   return status;
}

int MatchingPursuitResidual::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int arborID) {
   int status = PV_SUCCESS;
   if (updateGSynFlag((BaseConnection *) conn)) {
      status = ANNLayer::recvSynapticInput(conn, activity, arborID);
   }
   return status;
}

int MatchingPursuitResidual::recvSynapticInputFromPost(HyPerConn * conn, const PVLayerCube * activity, int arborID) {
   int status = PV_SUCCESS;
   if (updateGSynFlag((BaseConnection *) conn)) {
      status = ANNLayer::recvSynapticInputFromPost(conn, activity, arborID);
   }
   return status;
}

bool MatchingPursuitResidual::updateGSynFlag(BaseConnection * conn) {
   return !gSynInited || conn->getChannel()!=CHANNEL_EXC || excNeedsUpdate;
}

// bool MatchingPursuitResidual::getNewImageFlag() {
//    return syncedMovie && syncedMovie->getLastUpdateTime()>lastUpdateTime;
// }

MatchingPursuitResidual::~MatchingPursuitResidual() {
}

} /* namespace PV */
