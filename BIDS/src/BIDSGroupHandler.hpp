/*
 * BIDSGroupHandler.hpp
 *
 *  Created on: Mar 4, 2015
 *      Author: Pete Schultz
 */

#ifndef BIDSGROUPHANDLER_HPP_
#define BIDSGROUPHANDLER_HPP_

#include <io/ParamGroupHandler.hpp>

using namespace PV;

namespace PVBIDS {

class BIDSGroupHandler : public ParamGroupHandler {
public:
   BIDSGroupHandler();
   virtual ~BIDSGroupHandler();

   virtual ParamGroupType getGroupType(char const * keyword);

   virtual HyPerLayer * createLayer(char const * keyword, char const * name, HyPerCol * hc);

   virtual BaseConnection * createConnection(char const * keyword, char const * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer);

   // virtual BaseProbe * createProbe(char const * keyword, char const * name, HyPerCol * hc); // BIDS has no custom probes

   virtual InitWeights * createWeightInitializer(char const * keyword, char const * name, HyPerCol * hc);

   // virtual NormalizeBase * createWeightNormalizer(char const * keyword, char const * name, HyPerCol * hc) // BIDS has no custom weight normalizers
}; // class BIDSGroupHandler

}  // namespace PVBIDS

#endif // BIDSGROUPHANDLER_HPP_
