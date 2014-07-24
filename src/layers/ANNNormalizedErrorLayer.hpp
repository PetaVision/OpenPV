/*
 * ANNNormalizedErrorLayer.hpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#ifndef ANNNORMALIZEDERRORLAYER_HPP_
#define ANNNORMALIZEDERRORLAYER_HPP_

#include "ANNErrorLayer.hpp"
#include <fstream>
namespace PV {

class ANNNormalizedErrorLayer: public PV::ANNErrorLayer {
public:
   ANNNormalizedErrorLayer(const char * name, HyPerCol * hc);
   virtual ~ANNNormalizedErrorLayer();
   virtual double calcTimeScale();
   virtual double getTimeScale();
protected:
   ANNNormalizedErrorLayer();
   int initialize(const char * name, HyPerCol * hc);
private:
   int initialize_base();
   double timeScale;
   std::ofstream timeScaleStream;
};

} /* namespace PV */
#endif /* ANNNORMALIZEDERRORLAYER_HPP_ */
