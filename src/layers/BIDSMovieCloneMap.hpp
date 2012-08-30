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
   void setCoords(int jitter, float nxScale, float nyScale, int HyPerColx, int HyPerColy);
   BIDSCoords* getCoords();
   int getNumNodes();
   int updateState(float timef, float dt);
   
protected:
   BIDSMovieCloneMap();
   HyPerLayer * originalMovie;
   BIDSCoords* coords;
   int nbPre;
   int nxPre;
   int nyPre;
   int nxPost;
   int nyPost;
   int nf;
};

}
#endif /* BIDSMOVIECLONE_HPP_ */
