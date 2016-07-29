/*
 * MovieTestLayer.hpp
 * Author: slundquist
 */

#ifndef MOVIETESTLAYER_HPP_
#define MOVIETESTLAYER_HPP_
#include <layers/Movie.hpp>

namespace PV{

class MovieTestLayer : public PV::Movie{
public:
   MovieTestLayer(const char * name, HyPerCol * hc);
#ifdef PV_USE_GDAL
   virtual int updateState(double time, double dt);
#endif // PV_USE_GDAL
};


}
#endif /* IMAGETESTLAYER_HPP */
