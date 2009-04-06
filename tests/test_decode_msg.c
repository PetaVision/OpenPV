#include "../src/arch/pthreads/pv_thread.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>  /* or <inttypes.h> (<sys/types.h> doesn't work on Darwin) */
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
   uint32_t i;
   char code_in, code;
   void * addr, * buf;

   uint32_t msg  __attribute__ ((aligned(128)));
   uint32_t msg1 __attribute__ ((aligned(128)));

   for (i = 0; i < 1000; i++) {
      msg = (uint32_t) &msg1 + 16*i;
      if ((msg & ADDR_MASK) != msg) {
         printf("FAILED: mask failure at i=%d (%p,%p)\n",
                i, (void*) msg, (void*) (msg & ADDR_MASK));
         exit(1);
      }
   }

   for (i = 0; i < 16; i++) {
      buf = &msg1 + i*16;
      code_in = i;
      msg = encode_msg(buf, code_in);
      code = decode_msg(msg, &addr);
      if ( code != code_in || addr != buf ) {
         printf("FAILED:test_decode_msg: (code_in !=%d, code=%d)\n", code_in, code);
         exit(1);
      }
   }

  return 0;
}
