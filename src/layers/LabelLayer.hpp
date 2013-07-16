/*
 * LabelLayer.hpp
 *
 *  Created on: Jul 9, 2013
 *      Author: bcrocker
 */

#ifndef LABELLAYER_HPP_
#define LABELLAYER_HPP_

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"
#include "Movie.hpp"

namespace PV{

class LabelLayer : public HyPerLayer {

protected:
   LabelLayer();
   int initialize(const char * name, HyPerCol * hc, const char * movieLayerName);
   int initClayer(PVParams * params);
   Movie * movie;
   pvdata_t * labelData;
   int stepSize;
   const char * filename;
   int currentLabel;
   int maxLabel;
   int beginLabel;
   int lenLabel;
   PVLayerLoc labelLoc;


public:
   LabelLayer(const char * name, HyPerCol * hc, const char * movieLayerName);
   virtual int updateState(double time, double dt);
   virtual int outputState(double time, bool last);
   virtual ~LabelLayer();

private:
   int initialize_base();

};

}


#endif /* LABELLAYER_HPP_ */
