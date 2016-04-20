/*
 * NormalizeGap.cpp
 *
 *  Created on: Feb 28, 2014
 *      Author: pschultz
 */

// Note: NormalizeGap was deprecated on August 11, 2014.  It is included for backward compatibility.
// Previously, GapConn required that normalizeMethod be set to normalizeSum and that normalizeFromPostPerspective be set to true.
// Neither of these requirements are still present, but for now we want the defaults be what the requirements used to be.

#include "NormalizeGap.hpp"

namespace PV {

NormalizeGap::NormalizeGap() {
   initialize_base();
}

NormalizeGap::NormalizeGap(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

NormalizeGap::~NormalizeGap() {
}

int NormalizeGap::initialize_base() {
   return PV_SUCCESS;
}

int NormalizeGap::initialize(const char * name, HyPerCol * hc) {
   int status = NormalizeSum::initialize(name, hc);
   return status;
}

void NormalizeGap::ioParam_normalizeFromPostPerspective(enum ParamsIOFlag ioFlag) {
   // Default of true for normalizeFromPostPerspective was deprecated Aug 11, 2014.
   // This default was chosen for backwards compatibility because GapConn used to require normalizeMethod be normalizeSum,
   // and that normalizeFromPostPerspective be true.
   // Now GapConn can be normalized using any method, so eventually the default will be removed and the parameter required as is for other HyPerConns.
   if (ioFlag == PARAMS_IO_READ) {
      normalizeFromPostPerspective = true;
      if (parent->parameters()->present(name, "normalizeFromPostPerspective") && parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" warning: normalizeFromPostPerspective default is true for GapConns with normalizeMethod of normalizeSum, but the default for this parameter may change to false in a future release, to be consistent with other normalizerMethods.\n", parent->parameters()->groupKeywordFromName(name), name);
      }
   }
   parent->ioParamValue(ioFlag, name, "normalizeFromPostPerspective", &normalizeFromPostPerspective, true/*default*/, true/*warnIfAbsent*/);
}

BaseObject * createNormalizeGap(char const * name, HyPerCol * hc) {
   return hc ? new NormalizeGap(name, hc) : NULL;
}

} /* namespace PV */
