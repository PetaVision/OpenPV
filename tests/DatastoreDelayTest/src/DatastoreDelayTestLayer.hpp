/*
 * DatastoreDelayTest.hpp
 *
 *  Created on: Nov 2, 2011
 *      Author: pschultz
 */

#ifndef DATASTOREDELAYTEST_HPP_
#define DATASTOREDELAYTEST_HPP_

#include <layers/ANNLayer.hpp>

namespace PV {

class DatastoreDelayTestLayer : public ANNLayer {

  public:
   DatastoreDelayTestLayer(const char *name, HyPerCol *hc);
   virtual ~DatastoreDelayTestLayer();

   virtual Response::Status updateState(double timed, double dt) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);
   void updateState(
         double timed,
         double dt,
         int numNeurons,
         float *V,
         float *A,
         int nx,
         int ny,
         int nf,
         int lt,
         int rt,
         int dn,
         int up);

   static int
   updateV_DatastoreDelayTestLayer(const PVLayerLoc *loc, bool *inited, float *V, int period);

  protected:
   bool inited;
   int period; // The periodicity of the V buffer, in pixels.

}; // end of class DatastoreDelayTestLayer block

} // end of namespace PV block

#endif /* DATASTOREDELAYTEST_HPP_ */
