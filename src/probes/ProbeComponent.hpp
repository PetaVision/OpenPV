#ifndef PROBECOMPONENT_HPP_
#define PROBECOMPONENT_HPP_

#include "io/FileStream.hpp"
#include "params/ParamsIO.hpp"
#include <memory>
#include <string>

namespace PV {

class ProbeComponent {
  public:
   ProbeComponent(std::shared_ptr<ParamsIO> paramsIO);
   virtual ~ProbeComponent() {}

   void setPrintParamsStream(FileStream *stream);
   void setPrintLuaStream(FileStream *stream);

   std::string const &getName() const { return mParamsIO->getName(); }
   std::string const &getKeyword() const { return mParamsIO->getKeyword(); }
   char const *getName_c() const { return mParamsIO->getName().c_str(); }
   char const *getKeyword_c() const { return mParamsIO->getKeyword().c_str(); }

  protected:
   ProbeComponent();
   void initialize(std::shared_ptr<ParamsIO> paramsIO);

  protected:
   std::shared_ptr<ParamsIO> mParamsIO;
};

} // namespace PV

#endif // PROBECOMPONENT_HPP_
