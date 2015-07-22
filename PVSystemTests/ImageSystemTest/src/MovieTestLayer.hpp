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
   virtual int updateStateWrapper(double time, double dt);
};

}
#endif /* IMAGETESTLAYER_HPP */
