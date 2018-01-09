/*
 * GapConn.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: garkenyon
 */

#include "GapConn.hpp"

namespace PV {

// GapConn was made obsolete Sept 25, 2017. All the gap-specific functionality is in LIFGap,
// so using a HyPerConn with channelCode set to 3 gives equivalent behavior; although
// there are no warnings if you set sharedFlag to false or normalizeWeights to something
// other than normalizeSum.
GapConn::GapConn(const char *name, HyPerCol *hc) {
   Fatal() << "GapConn has been eliminated. Use a HyPerConn with channelCode set to 3.\n";
}

GapConn::~GapConn() {}

} /* namespace PV */
