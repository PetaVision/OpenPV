/*
 * MatchingPursuitResidual.hpp
 *
 *  Created on: Aug 13, 2013
 *      Author: pschultz
 */

#ifndef MATCHINGPURSUITRESIDUAL_HPP_
#define MATCHINGPURSUITRESIDUAL_HPP_

#include "ANNLayer.hpp"
#include "Movie.hpp"

namespace PV {

class MatchingPursuitResidual: public PV::ANNLayer {
public:
   MatchingPursuitResidual(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int resetGSynBuffers(double timed, double dt);
   virtual int recvAllSynapticInput();
   virtual int recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity, int arborID);
   virtual int recvSynapticInputFromPost(HyPerConn * conn, const PVLayerCube * activity, int arborID);
   virtual ~MatchingPursuitResidual();

protected:
   MatchingPursuitResidual();
   int initialize(const char * name, HyPerCol * hc);
   virtual int setParams(PVParams * params);
   virtual void readSyncedMovie(PVParams * params);

   inline bool updateGSynFlag(HyPerConn * conn);
   inline bool getNewImageFlag();

private:
   int initialize_base();

protected:
   char * syncedMovieName;        // If set to the name of a Movie layer, activity resets every time the movie's getNewImageFlag() returns true
   Movie * syncedMovie;           // The layer whose name is syncedMovieName

   bool gSynInited;               // Initially false; it is set to true the first time recvAllSynapticInput is called.
};

} /* namespace PV */
#endif /* MATCHINGPURSUITRESIDUAL_HPP_ */
