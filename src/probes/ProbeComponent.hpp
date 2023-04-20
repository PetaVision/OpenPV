#ifndef PROBECOMPONENT_HPP_
#define PROBECOMPONENT_HPP_

#include "io/PVParams.hpp"
#include <string>

namespace PV {

class ProbeComponent {
  public:
   ProbeComponent(char const *objName, PVParams *params);
   virtual ~ProbeComponent() {}

   std::string const &getName() const { return mName; }
   char const *getName_c() const { return mName.c_str(); }

  protected:
   ProbeComponent();
   void initialize(char const *objName, PVParams *params);
   PVParams *getParams() { return mParams; }

  private:
   std::string mName;
   PVParams *mParams = nullptr;
};

} // namespace PV

#endif // PROBECOMPONENT_HPP_
