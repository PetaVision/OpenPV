/*
 * CustomGroupHandler.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: pschultz
 */

#include <io/ParamGroupHandler.hpp>

namespace PV {

class CustomGroupHandler : public ParamGroupHandler {
public:
   CustomGroupHandler();

   virtual ~CustomGroupHandler();

   virtual ParamGroupType getGroupType(char const * keyword);

   virtual NormalizeBase * createWeightNormalizer(char const * keyword, char const * name, HyPerCol * hc);
}; /* class CustomGroupHandler */

}  // namespace PV
