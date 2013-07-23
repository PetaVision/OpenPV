/*
 * BIDSMovieClone.hpp
 *
 *  Created on: Aug 29, 2012
 *      Author: slundquist
 */

#ifndef BIDSMOVIECLONE_HPP_
#define BIDSMOVIECLONE_HPP_

#include "HyPerLayer.hpp"
#include "Movie.hpp"

typedef struct _BIDSCoords{
   int xCoord;
   int yCoord;
} BIDSCoords;

namespace PV{

class BIDSMovieCloneMap : public PV::HyPerLayer{
public:
   BIDSMovieCloneMap(const char * name, HyPerCol * hc, int numChannels);
   BIDSMovieCloneMap(const char * name, HyPerCol * hc);
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc, int numChannels=MAX_CHANNELS);
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   void setCoords(int jitter, float nxScale, float nyScale, int HyPerColx, int HyPerColy);
   BIDSCoords* getCoords();
   int getNumNodes();
   int updateState(double timef, double dt);
   ~BIDSMovieCloneMap();
   
protected:
   BIDSMovieCloneMap();
   char * originalMovieName;
   HyPerLayer * originalMovie;
   BIDSCoords* coords;
   int getNbOrig() {return originalMovie->getLayerLoc()->nb;}
   int getNxOrig() {return originalMovie->getLayerLoc()->nx;}
   int getNyOrig() {return originalMovie->getLayerLoc()->ny;}
   int nxPost;
   int nyPost;
   int getNfOrig() {return originalMovie->getLayerLoc()->nf;}
   int jitter;
};

}
#endif /* BIDSMOVIECLONE_HPP_ */
