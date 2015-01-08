#ifndef DISPARITYMOVIE_HPP_ 
#define DISPARITYMOVIE_HPP_

#include <layers/Movie.hpp>

namespace PV {

class DisparityMovie: public PV::Movie{
public:
	DisparityMovie(const char* name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);



protected:
   bool updateImage(double timef, double dt);
   void ioParam_numDisparityPeriod(enum ParamsIOFlag ioFlag);
   void ioParam_dPixelDisparity(enum ParamsIOFlag ioFlag);
   void ioParam_moveMethod(enum ParamsIOFlag ioFlag);

private:
   int initialize_base();
   int numDisparity;
   int dPixelDisparity;
   int disparityIndex;
   char* moveMethod;
   int frameOffset;
   int frameCount;
};

} /* namespace PV */
#endif
