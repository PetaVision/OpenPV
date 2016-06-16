#include "PVAssert.hpp"
#include <stdio.h>
#include <stdarg.h>

namespace PV {

void pv_abort_message(const char *file, int line, const char *fmt, ...) {
   getOutputStream().flush();
   std::ostream& st = getErrorStream();
   static int buf_size = 1024;
   char msg[buf_size];
   va_list args;
   va_start(args, fmt);
   vsnprintf(msg, buf_size, fmt, args);
   va_end(args);
   st << "assert failed";
   if (file) {
      st << " <" << basename((char*)file) << ":" << line << ">";
   }
   st << ": " << msg;
   st.flush();
   abort();
}

void pv_assert_failed(const char *file, int line, const char *condition) {
   pv_abort_message(file, line, "%s\n", condition);
}

void pv_assert_failed_message(const char *file, int line, const char *condition, const char *fmt, ...) {
   /* Build up custom error string */
   va_list args;
   va_start(args, fmt);
   static int buf_size = 1024;
   char msg[buf_size];
   vsnprintf(msg, buf_size, fmt, args);
   va_end(args);

   pv_abort_message(file, line, "%s: %s\n", condition, msg);
}

}
