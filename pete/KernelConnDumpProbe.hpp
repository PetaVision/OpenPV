/*
 * KernelConnProbe.hpp
 *
 *  Created on: Jan 10, 2011
 *      Author: pschultz
 */

#ifndef KERNELCONNDUMPPROBE_HPP_
#define KERNELCONNDUMPPROBE_HPP_

#include "../PetaVision/src/io/ConnectionProbe.hpp"
#include "../PetaVision/src/connections/HyPerConn.hpp"
#include "../PetaVision/src/connections/KernelConn.hpp"

namespace PV {

class KernelConnDumpProbe : public ConnectionProbe {
public:
   KernelConnDumpProbe();
   KernelConnDumpProbe(const char * filenameformatstr);
   KernelConnDumpProbe(const char * filenameformatstr, bool textNotBinary);
   virtual ~KernelConnDumpProbe();

   virtual int outputState(float time, HyPerConn * c);
   bool getTextNotBinary() {return textNotBinaryFlag;}

protected:
   bool textNotBinaryFlag;
   bool stdoutNotFileFlag;
   char * filenameFormatString;

   int validateFormatString(const char * filenameformatstr);

   int initializeProbe();
}; // end class KernelConnProbe

}  // end namespace PV

#endif /* KERNELCONNDUMPPROBE_HPP_ */
