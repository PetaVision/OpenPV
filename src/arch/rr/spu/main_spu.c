#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>

/*#include <spu_internals.h>*/
#include <spu_mfcio.h>
//#include <spu_timer.h>

#include "pv.h"
#include "pv_harness.h"

// macro for waiting to completion of DMA group related to input tag.
// does so by first setting tag mask and then read the status which is
// blocked until all tags DMA are completed.
#define tag_wait(t) mfc_write_tag_mask(1<<t); mfc_read_tag_status_all();

#define NUM_BUFS 3

volatile data_chunk_t data_buf[NUM_BUFS] __attribute__ ((aligned(128)));

#define MAX_DMA_SIZE_BYTES 16384

volatile mfc_list_element_t dma_list[4] __attribute__ ((aligned(128)));
uint64_t ppu_data_offset[3];

volatile int32_t mboard_from_prev __attribute__ ((aligned(128))) = -1;
volatile int32_t mboard_from_next __attribute__ ((aligned(128))) = -1;
int32_t msg_to_prev __attribute__ ((aligned(16)));
int32_t msg_to_next __attribute__ ((aligned(16)));

float x[NSPU/NTH] __attribute__ ((aligned(128)));
float y[NSPU/NTH] __attribute__ ((aligned(128)));
float f[NSPU] __attribute__ ((aligned(128)));

int myspu;
spu_param_t spu_param __attribute__ ((aligned(16)));
spu_timing_t spu_timing __attribute__ ((aligned(16)));

uint64_t prev_mboard_ea;
uint64_t next_mboard_ea;
uint64_t next_chunk_ea;

/**
 * @tag_id
 * @c
 * @curr_buff
 */
void get_ppu_data(uint32_t tag_id, int c, int curr_buf)
{
   int k;
   uint64_t t_start;

   //t_start = spu_clock_read();
   for (k = 0; k < 3; k++) {
      dma_list[k].size   = CHUNK_SIZE_BYTES / 4;
      dma_list[k].eal    = ppu_data_offset[k] + c * (CHUNK_SIZE_BYTES / 4);
      dma_list[k].notify = 0;
   }
   mfc_getl(&data_buf[curr_buf], spu_param.ppu_phdata_ea, (void*) dma_list,
            sizeof(mfc_list_element_t) * 4,
            tag_id, 0, 0);
}


void put_spu_data(uint32_t tag_id, int c, int curr_buf)
{
   uint64_t t_start;

   //t_start = spu_clock_read();
   mfc_put(&data_buf[curr_buf],
           next_chunk_ea + curr_buf * CHUNK_SIZE_BYTES,
           CHUNK_SIZE_BYTES, tag_id, 0, 0);

   /* notify next SPU that chunk c is ready */
   msg_to_next = c;
   mfc_putf(&msg_to_next, next_mboard_ea, sizeof(int32_t), tag_id, 0, 0);
   /* notify previous SPU that we are ready for chunk (c + NUM_BUFS) */
   if (myspu > 0) {
      msg_to_prev = c + NUM_BUFS;
      mfc_putf(&msg_to_prev, prev_mboard_ea, sizeof(int32_t), tag_id, 0, 0);
   }
}


void put_ppu_data(uint32_t tag_id, int c, int curr_buf)
{
   uint64_t t_start;
   uint64_t addr;

   //t_start = spu_clock_read();
   addr = spu_param.ppu_phdata_ea + (c * CHUNK_SIZE_BYTES / 4);
   mfc_put(data_buf[curr_buf].ph, addr, CHUNK_SIZE_BYTES / 4, tag_id, 0, 0);

   /* notify PPU that chunk c is ready */
   msg_to_next = c;
   mfc_putf(&msg_to_next, next_mboard_ea, sizeof(int32_t), tag_id, 0, 0);
   /* notify previous SPU that we are ready for chunk (c + NUM_BUFS) */
   msg_to_prev = c + NUM_BUFS;
   mfc_putf(&msg_to_prev, prev_mboard_ea, sizeof(int32_t), tag_id, 0, 0);
}


void compute_data(int curr_buf, float xpoffset, float ypoffset)
{
   uint64_t t_start;

   //t_start = spu_clock_read();
   update_phi(CHUNK_SIZE, NSPU,
              (float *) data_buf[curr_buf].ph,
              (float *) data_buf[curr_buf].x,
              (float *) data_buf[curr_buf].y,
              x, xpoffset, y, ypoffset, f);
   //spu_timing.cyc_compute += (spu_clock_read() - t_start);
}


void transpose(float *array)
{
   float tmp[4*NTH];
   int k, i, j;

   for (k = 0; k < NSPU; k+=(4*NTH)) {
      memcpy(tmp, &array[k], (4*NTH) * sizeof(float));
      for (i = 0; i < 4; i++) {
         for (j = 0; j < NTH; j++) {
            array[k+j*4+i] = tmp[i*NTH+j];
         }
      }
   }
}


void compress(float *array)
{
   int k, i;

   for (k = NTH, i = 1; k < NSPU; k+=NTH, i+=1) {
      memcpy(&array[i], &array[k], 1 * sizeof(float));
   }
}


int main(uint64_t speid, uint64_t argp)
{
   int step;
   int c;
   int k;
   uint64_t addr;
   uint32_t tag_id;
   uint32_t mbox_msg;
   uint32_t total_size, size, off, ppu_off;
   int curr_buf, next_buf, prev_buf;
   int nbr;
   float xpoffset, ypoffset;

   uint64_t t_start;

   tag_id = mfc_tag_reserve();
   if (tag_id == MFC_TAG_INVALID) {
      printf("SPE: ERROR can't allocate tag ID\n");
      return -1;
   }

   mfc_get(&spu_param, argp, sizeof(spu_param), tag_id, 0, 0);
   tag_wait(tag_id);

   myspu = spu_param.myspu;
   ppu_data_offset[0] = mfc_ea2l(spu_param.ppu_phdata_ea);
   ppu_data_offset[1] = mfc_ea2l(spu_param.ppu_xdata_ea);
   ppu_data_offset[2] = mfc_ea2l(spu_param.ppu_ydata_ea);

   if (myspu == 0) {
      prev_mboard_ea = 0;
   } else {
      prev_mboard_ea = spu_param.prev_ls_base_ea + (uint32_t) (&mboard_from_next);
   }

   if (myspu == NUM_SPUS - 1) {
      next_mboard_ea = spu_param.ppu_mboard_ea;
      next_chunk_ea = 0;
   } else {
      next_mboard_ea = spu_param.next_ls_base_ea + (uint32_t) (&mboard_from_prev);
      next_chunk_ea = spu_param.next_ls_base_ea + (uint32_t) data_buf;
   }

   /* transfer x, y data from PPU */
   /* DMA lists might be faster, but the code would be messier */
   mbox_msg = spu_read_in_mbox();
   total_size = NSPU / NTH * sizeof(float);
   ppu_off = myspu * total_size;
   off = 0;
   while (off < total_size) {
      size = total_size - off;
      if (size > MAX_DMA_SIZE_BYTES) size = MAX_DMA_SIZE_BYTES;
      mfc_get((void *)x + off, spu_param.ppu_xpdata_ea + ppu_off, size, tag_id, 0, 0);
      mfc_get((void *)y + off, spu_param.ppu_ypdata_ea + ppu_off, size, tag_id, 0, 0);
      ppu_off += size;
      off += size;
   }
   tag_wait(tag_id);

   /* setup for timing */
   memset(&spu_timing, 0, sizeof(spu_timing));
   //spu_slih_register(MFC_DECREMENTER_EVENT, spu_clock_slih);
   //spu_clock_start();

   for (step = 0; step < spu_param.num_steps; step++) {

      for (nbr = 0; nbr < NUM_NBRS; nbr++) {

         xpoffset = nbr_offsets[nbr][0] * NX * DX;
         ypoffset = nbr_offsets[nbr][1] * NY * DY;

         /* wait for message that PPE data is ready */
         mbox_msg = spu_read_in_mbox();

         /* calculate activity address */
         /* get activity */

         /* calculate phi address */
         /* get phi */

         /* forall kPre get task/weights */

         /*  */

         for (int y = 0; y < ny; y++) {
            pvpatch_accumulate(nk, phi->data + y*sy, a, weights->data + y*syw);
         }


         /* read in new f's */
         total_size = NSPU * sizeof(float);
         ppu_off = myspu * total_size + nbr * N * sizeof(float);
         off = 0;
         while (off < total_size) {
            size = total_size - off;
            if (size > MAX_DMA_SIZE_BYTES) size = MAX_DMA_SIZE_BYTES;
            mfc_get((void *)f + off, spu_param.ppu_fdata_ea + ppu_off, size, tag_id, 0, 0);
            ppu_off += size;
            off += size;
         }
         tag_wait(tag_id);
#ifdef SPU_DATA_OPTIMIZE
         transpose(f);
#endif

         /* SPU 0 only:  transfer chunk 0 data in from PPU */
         next_buf = 0;
         if (myspu == 0) {
            get_ppu_data(tag_id, 0, next_buf);
            tag_wait(tag_id);
         }

         /* main processing loop for chunk c */
         for (c = 0; c < NUM_CHUNKS; c++) {
            curr_buf = c % NUM_BUFS;
            next_buf = (c + 1) % NUM_BUFS;
            prev_buf = (c + (NUM_BUFS - 1)) % NUM_BUFS;

            /* wait for message that chunk c-1 transfer out can start */
            if (c - 1 >= 0) {
               if (myspu < NUM_SPUS - 1) {
                  while (mboard_from_next < c - 1) /* wait */ ;
               }
            }

            /* wait for message that chunk c transfer in is done */
            if (myspu > 0) {
               while (mboard_from_prev < c) /* wait */ ;
            }

            spu_dsync();

            /* transfer chunk c-1 out */
            if (c - 1 >= 0) {
               if (myspu == (NUM_SPUS - 1)) {
                  put_ppu_data(tag_id, c-1, prev_buf);
               } else {
                  put_spu_data(tag_id, c-1, prev_buf);
               }
            }

            /* SPU 0 only:  transfer chunk c+1 data in from PPU */
            if (c + 1 < NUM_CHUNKS) {
               if (myspu == 0) {
                  get_ppu_data(tag_id, c+1, next_buf);
               }
            }

            /* compute on chunk c */
            compute_data(curr_buf, xpoffset, ypoffset);

            /* wait for dma to complete */
            tag_wait(tag_id);

         } // for c

         /* wait for message that chunk N-1 transfer out can start */
         if (myspu < NUM_SPUS - 1) {
            while (mboard_from_next < NUM_CHUNKS - 1) /* wait */ ;
         }

         spu_dsync();

         /* transfer chunk N-1 out */
         prev_buf = (NUM_CHUNKS + (NUM_BUFS - 1)) % NUM_BUFS;
         if (myspu == (NUM_SPUS - 1)) {
            put_ppu_data(tag_id, NUM_CHUNKS-1, prev_buf);
         } else {
            put_spu_data(tag_id, NUM_CHUNKS-1, prev_buf);
         }

         /* wait for dma to complete */
         tag_wait(tag_id);

      }  // for nbr

   }  // for step

   //spu_clock_stop();

   /* return timing info to PPU */
   mfc_put(&spu_timing, spu_param.ppu_timing_ea,
           sizeof(spu_timing), tag_id, 0, 0);
   tag_wait(tag_id);
   spu_write_out_mbox(0);

   mfc_tag_release(tag_id);

}



