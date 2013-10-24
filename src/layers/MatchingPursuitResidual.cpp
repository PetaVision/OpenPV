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
   return ANNLayer::initialize(name, hc);
}

int MatchingPursuitResidual::setParams(PVParams * params) {
   int status = HyPerLayer::setParams(params);
   readSyncedMovie(params);
   readRefreshPeriod(params);
   return status;
}

void MatchingPursuitResidual::readSyncedMovie(PVParams * params) {
   const char * synced_movie_name = params->stringValue(name, "syncedMovie", true);
   if (synced_movie_name && synced_movie_name[0]) {
      syncedMovieName = strdup(synced_movie_name);
      if (syncedMovieName==NULL) {
         fprintf(stderr, "MatchingPursuitLayer \"%s\" error: rank %d process unable to copy syncedMovie param: %s\n", name, parent->columnId(), strerror(errno));
         abort();
      }
   }
}

void MatchingPursuitResidual::readRefreshPeriod(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "syncedMovie"));
   if (syncedMovieName==NULL || syncedMovieName[0]=='\0') {
      refreshPeriod = params->value(name, "refreshPeriod", parent->getDeltaTime());
      if (refreshPeriod>=0) nextRefreshTime = parent->simulationTime();
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
   if (updateGSynFlag(conn)) {
      status = ANNLayer::recvSynapticInput(conn, activity, arborID);
   }
   return status;
}

int MatchingPursuitResidual::recvSynapticInputFromPost(HyPerConn * conn, const PVLayerCube * activity, int arborID) {
   int status = PV_SUCCESS;
   if (updateGSynFlag(conn)) {
      status = ANNLayer::recvSynapticInputFromPost(conn, activity, arborID);
   }
   return status;
}

bool MatchingPursuitResidual::updateGSynFlag(HyPerConn * conn) {
   return !gSynInited || conn->getChannel()!=CHANNEL_EXC || excNeedsUpdate;
}

// bool MatchingPursuitResidual::getNewImageFlag() {
//    return syncedMovie && syncedMovie->getLastUpdateTime()>lastUpdateTime;
// }

MatchingPursuitResidual::~MatchingPursuitResidual() {
}

} /* namespace PV */
