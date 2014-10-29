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
#ifdef OBSOLETE
   //Obsolete Jan 15th, 2014 by slundquist
   //getLastUpdateTime in HyPerLayer no longer updates lastUpdateTime, so no longer need to override
   virtual double getLastUpdateTime() {return lastUpdateTime;}
#endif // OBSOLETE

protected:
   MatchingPursuitResidual();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_syncedMovie(enum ParamsIOFlag ioFlag);
   virtual void ioParam_refreshPeriod(enum ParamsIOFlag ioFlag);

   inline bool updateGSynFlag(HyPerConn * conn);
   // inline bool getNewImageFlag();

private:
   int initialize_base();

protected:
   char * syncedMovieName;        // If set to the name of a Movie layer, activity resets every time the movie's getNewImageFlag() returns true
   Movie * syncedMovie;           // The layer whose name is syncedMovieName
   double refreshPeriod;          // If no syncedMovieName, activity resets every refreshPeriod.  Negative value means never refresh
   double nextRefreshTime;        // The next time activity is reset.
   bool excNeedsUpdate;

   bool gSynInited;               // Initially false; it is set to true the first time recvAllSynapticInput is called.
};

} /* namespace PV */
#endif /* MATCHINGPURSUITRESIDUAL_HPP_ */
