#ifndef __PRINTSTREAM_HPP_
#define __PRINTSTREAM_HPP_

extern "C" {
#include <unistd.h>
}

#include <cstdio>
#include <cstring>
#include <cstdarg>

#include "utils/PVLog.hpp"
#include "utils/PVAssert.hpp"
#include "io/io.hpp"


namespace PV {

class PrintStream {
   public:
      PrintStream(std::ostream& stream) {
         setOutStream(stream);
      }
      
      int printf(const char *fmt, ...) {
         va_list args1, args2;
         va_start(args1, fmt);
         va_copy(args2, args1);
         char c;
         int chars_needed = vsnprintf(&c, 1, fmt, args1);
         chars_needed++;
         char output_string[chars_needed];
         int chars_printed = vsnprintf(output_string, chars_needed, fmt, args2);
         pvAssert(chars_printed+1==chars_needed);
         (*mOutStream) << std::string(output_string);
         va_end(args1);
         va_end(args2);
         return chars_needed;
      }
      
      void flush() {
         mOutStream->flush();
      }

      // Operator overloads to allow using << like cout
      template <typename T>
      PrintStream& operator<< (const T &x) {
         (*mOutStream) << x;
         return *this;
      }
      PrintStream& operator<< (std::ostream& (*f)(std::ostream &)) {
         f(std::cout);
         return *this;
      }
      PrintStream& operator<< (std::ostream& (*f)(std::ios &)) {
         f(std::cout);
         return *this;
      }
      PrintStream& operator<< (std::ostream& (*f)(std::ios_base &)) {
         f(std::cout);
         return *this;
      }
   protected:
      PrintStream() {}
      void setOutStream(std::ostream& stream) {
         mOutStream = &stream;
      }

   private:
      std::ostream *mOutStream = nullptr;
};
} /* namespace PV */

#endif 
