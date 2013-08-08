/*
 * MatchingPursuitLayer.hpp
 *
 *  Created on: Jul 31, 2013
 *      Author: pschultz
 */

#ifndef MATCHINGPURSUITLAYER_HPP_
#define MATCHINGPURSUITLAYER_HPP_

#include "HyPerLayer.hpp"
#include "Movie.hpp"

struct matchingpursuit_mpi_data { pvdata_t maxval; int maxloc; int mpirank;};

namespace PV {

class MatchingPursuitLayer: public HyPerLayer {
public:
   MatchingPursuitLayer(const char * name, HyPerCol * hc);
   virtual ~MatchingPursuitLayer();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int updateState(double timed, double dt);
   virtual int outputState(double timed, bool last=false);

   pvdata_t getActivationThreshold() {return activationThreshold;}

protected:
   MatchingPursuitLayer();
   int initialize(const char * name, HyPerCol * hc);
   int openPursuitFile();
   virtual int setParams(PVParams * params);
   virtual void readActivationThreshold(PVParams * params);
   virtual void readSyncedMovie(PVParams * params);
   virtual void readTracePursuit(PVParams * params);
   virtual void readPursuitFile(PVParams * params);

   inline void updateMaxinfo(pvdata_t gsyn, int k);

private:
   int initialize_base();

// Member variables
protected:
   pvdata_t activationThreshold;  // Activities below this value in absolute value are treated as zero
   char * syncedMovieName;        // If set to the name of a Movie layer, activity resets every time the movie's getNewImageFlag() returns true
   Movie * syncedMovie;           // The layer whose name is syncedMovieName
   bool tracePursuit;             // If true, print the neuron whose activity is changed (global restricted index) and the change in activity.
   char * traceFileName;          // If tracePursuit is true, holds the file the trace is output to.  If null or empty, use standard output.
   PV_Stream * traceFile;         // The PV_Stream corresponding to traceFileName

   struct matchingpursuit_mpi_data maxinfo; // Contains the neuron index with the biggest change in activity. argmax |<R,g_i>| in matching pursuit algorithm
};

} /* namespace PV */
#endif /* MATCHINGPURSUITLAYER_HPP_ */
