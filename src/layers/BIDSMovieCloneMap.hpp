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
   int initialize(const char * name, HyPerCol * hc, int numChannels);
   int numNodes;
   void setCoords(int numNodes, BIDSCoords ** coords, int jitter, float nxScale, float nyScale, int HyPerColx, int HyPerColy);
   BIDSCoords BIDSMovieCloneMap::getCoords(int x, int y);
   int BIDSMovieCloneMap::updateState(float timef, float dt);
   
protected:
   BIDSMovieCloneMap();
   Movie * originalMovie;
   BIDSCoords** coords;
   int nbPre;
   int nxPre;
   int nyPre;
   int nxPost;
   int nyPost;
   int nf;

};

}
#endif /* BIDSMOVIECLONE_HPP_ */
