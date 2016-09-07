/*
 * Example.hpp
 *
 *  Created on: Oct 19, 2008
 *      Author: rasmussn
 */

#ifndef EXAMPLE_HPP_
#define EXAMPLE_HPP_

#include <layers/HyPerLayer.hpp>

namespace PV
{

class Example: public PV::HyPerLayer
{
public:
   Example(const char* name, HyPerCol * hc);
   virtual bool activityIsSpiking() { return false; }

   virtual int updateState(double time, double dt);

   virtual int initFinish(int colId, int colRow, int colCol);

   virtual int outputState(double timef, bool last=false);
};

}

#endif /* EXAMPLE_HPP_ */
