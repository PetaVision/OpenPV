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
   return PV_SUCCESS;
}

int MatchingPursuitResidual::initialize(const char * name, HyPerCol * hc) {
   return ANNLayer::initialize(name, hc);
}

int MatchingPursuitResidual::setParams(PVParams * params) {
   int status = HyPerLayer::setParams(params);
   readSyncedMovie(params);
   return status;
}

void MatchingPursuitResidual::readSyncedMovie(PVParams * params) {
   const char * synced_movie_name = params->stringValue(name, "syncedMovie", true);
   if (synced_movie_name && synced_movie_name[0]) {
      syncedMovieName = strdup(synced_movie_name);
      if (syncedMovieName==NULL) {
         fprintf(stderr, "MatchingPursuitLayer2 \"%s\" error: rank %d process unable to copy syncedMovie param: %s\n", name, parent->columnId(), strerror(errno));
         abort();
      }
   }
}

int MatchingPursuitResidual::communicateInitInfo() {
   int status = ANNLayer::communicateInitInfo();

   if (syncedMovieName != NULL) {
      assert(syncedMovieName[0]);
      HyPerLayer * syncedLayer = parent->getLayerFromName(syncedMovieName);
      if (syncedLayer==NULL) {
         fprintf(stderr, "MatchingPursuitLayer2 \"%s\" error: syncedMovie \"%s\" is not a layer in the HyPerCol.\n",
               name, syncedMovieName);
         return(EXIT_FAILURE);
      }
      syncedMovie = dynamic_cast<Movie *>(syncedLayer);
      if (syncedMovie==NULL) {
         fprintf(stderr, "MatchingPursuitLayer2 \"%s\" error: syncedMovie \"%s\" is not a Movie or Movie-derived layer in the HyPerCol.\n",
               name, syncedMovieName);
         return(EXIT_FAILURE);
      }
   }

   return status;
}

int MatchingPursuitResidual::resetGSynBuffers(double timed, double dt) {
   int status = PV_SUCCESS;
   if (syncedMovie && syncedMovie->getNewImageFlag()) {
      status = ANNLayer::resetGSynBuffers(timed, dt);
   }
   return status;
}

int MatchingPursuitResidual::recvAllSynapticInput() {
   int status = ANNLayer::recvAllSynapticInput();
   gSynInited = true;
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
   return !gSynInited || conn->getChannel()!=CHANNEL_EXC || (syncedMovie && syncedMovie->getNewImageFlag());
}

bool MatchingPursuitResidual::getNewImageFlag() {
   return syncedMovie && syncedMovie->getNewImageFlag();
}

MatchingPursuitResidual::~MatchingPursuitResidual() {
}

} /* namespace PV */
