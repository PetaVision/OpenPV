/*
 * MatchingPursuitLayer.hpp
 *
 *  Created on: Jul 31, 2013
 *      Author: pschultz
 */

#ifndef MATCHINGPURSUITLAYER_HPP_
#define MATCHINGPURSUITLAYER_HPP_

#include <layers/HyPerLayer.hpp>
#include <layers/Movie.hpp>

struct matchingpursuit_mpi_data { pvdata_t maxval; int maxloc; int mpirank;};

namespace PVMatchingPursuit {

class MatchingPursuitLayer: public PV::HyPerLayer {
public:
   MatchingPursuitLayer(const char * name, PV::HyPerCol * hc);
   virtual ~MatchingPursuitLayer();
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual bool activityIsSpiking() { return false; }
   virtual int updateState(double timed, double dt);
   virtual int outputState(double timed, bool last=false);

   pvdata_t getActivationThreshold() {return activationThreshold;}

protected:
   MatchingPursuitLayer();
   int initialize(const char * name, PV::HyPerCol * hc);
   int openPursuitFile();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_activationThreshold(enum ParamsIOFlag ioFlag);
   virtual void ioParam_syncedMovie(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tracePursuit(enum ParamsIOFlag ioFlag);
   virtual void ioParam_traceFile(enum ParamsIOFlag ioFlag);

   inline void initializeMaxinfo(int rank=-1);
   inline void updateMaxinfo(pvdata_t gsyn, int k);

   inline bool inWindowGlobalRes(int neuronIdxRes, const PVLayerLoc * loc);


private:
   int initialize_base();

// Member variables
protected:
   pvdata_t activationThreshold;  // Activities below this value in absolute value are treated as zero
   char * syncedMovieName;        // If set to the name of a Movie layer, activity resets every time the movie's getNewImageFlag() returns true
   PV::Movie * syncedMovie;           // The layer whose name is syncedMovieName
   bool tracePursuit;             // If true, print the neuron whose activity is changed (global restricted index) and the change in activity.
   char * traceFileName;          // If tracePursuit is true, holds the file the trace is output to.  If null or empty, use standard output.
   PV_Stream * traceFile;         // The PV_Stream corresponding to traceFileName

   struct matchingpursuit_mpi_data maxinfo; // Contains the neuron index with the biggest change in activity. argmax |<R,g_i>| in matching pursuit algorithm
   bool useWindowedSynapticInput; // If all connections to this layer update GSyn from the postsynaptic perspective, this is set to true.
                                  // If true, then we only need to update activity in a window around the most recently changed activity.
   int xWindowSize;               // The distance in the x-direction from the changed activity to the edge of the window where the GSyn changes.
   int yWindowSize;               // The distance in the y-direction from the changed activity to the edge of the window where the GSyn changes.

   int activeIndex;
}; // end class MatchingPursuitLayer

PV::BaseObject * createMatchingPursuitLayer(char const * name, PV::HyPerCol * hc);

} /* namespace PVMatchingPursuit */
#endif /* MATCHINGPURSUITLAYER_HPP_ */
