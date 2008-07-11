#include "pv.h"
#include <columns/PVHyperCol.h>
#include <layers/PVLayer.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h> // for options parsing

#define PARTIAL_OUTPUT 1

/* declaration of internal functions (should these be static) */

int output_state(PVHyperCol* hc, int time_step);
int output_partial_state(PVHyperCol* hc, int time_step);
int output_final_state(PVHyperCol* hc, int time_step);
void pv_output_events_circle(int step, float f[], float h[]);
void pv_output_on_circle(int step, const char* name, float max, float buf[]);
static void parse_options(int argc, char* argv[], char* input_filename, int* n_time_steps);


static void print_result(int length, int cycles, double time)
  {
    double bandwidth, clock_prec;

    if (time < 0.000001)
      return;

    clock_prec = MPI_Wtick();
    bandwidth = (length * clock_prec * cycles) / (1024.0 * 1024.0) / (time
        * clock_prec);
    printf("%8d\t%.6f\t%.4f MB/s\n", length, time / cycles, bandwidth);
  }

static void parse_options(int argc, char* argv[], char* input_filename, int* n_time_steps)
{
    int index;
    int c;
     
    opterr = 0;
     
    while ((c = getopt (argc, argv, "i:n:")) != -1)
    switch (c) {
        case 'i':
	    strcpy(input_filename, optarg);
            break;
        case 'n':
            *n_time_steps = atoi(optarg);
	    break;
        default:
            printf("Unrecognized option '%c'. Aborting.", c);
            exit(-1);
    }
}

int main(int argc, char* argv[])
  {
    int ihc, t;
    int hc_id, comm_id, comm_size;
    double tstart, tend;
    PVHyperCol* hc;
    char input_filename[MAX_FILENAME];
    int n_time_steps = 1;

    input_filename[0] = 0; // clear so we know if user set 

    comm_init(&argc, &argv, &comm_id, &comm_size);

    if (argc == 2)
      {
        n_time_steps = atoi(argv[1]);
      }
    else {
	// Parse the input options.
	parse_options(argc, argv, input_filename, &n_time_steps);
    }

    hc = pv_new_hypercol(comm_id, comm_size, n_time_steps, input_filename);

    // TODO - add initial output
/*     sprintf(filename, "f%d", hc->comm_id); */
/*     pv_output(filename, 0.5, hc->x0, hc->y0, l->x, l->y, l->o, f); */
    char filename[75];
    sprintf(filename, "input_%d", hc->comm_id);
    PVLayer* layer_ptr = (PVLayer*) hc->layer[0];
    pv_output(filename, 0.5, 
	      hc->x0, hc->y0, 
	      layer_ptr->x, layer_ptr->y, layer_ptr->o, 
	      layer_ptr->f);
    
    /* time loop */

    tstart = MPI_Wtime();

    for (t = 0; t < n_time_steps; t++)
      {
        for (ihc = 0; ihc <= hc->n_neighbors; ihc++)
          {
            pv_hypercol_begin_update(hc, ihc, t);
          }
        pv_hypercol_finish_update(hc, t);

        // pv_hypercol_send_layer(hc, i, req);           send_state(hc, req);

        /* update with local events first */
        //           update_partial_state(hc, 0);

        /* loop over neighboring columns */
        //           for (c = 0; c < hc->n_neighbors; c++)
        //             {
        //             recv_state(hc, req, &hc_id); /* hc_id is the index of neighbor */
        //           update_partial_state(hc, hc_id + 1); /* 0 is local index, 1 first neighbor */
        //       }

#ifdef PARTIAL_OUTPUT
        //     output_partial_state(hc, t);
#endif

        /* complete update (new V and local event mask) */
        //     update_state(hc);

#ifdef PARTIAL_OUTPUT
             output_state(hc, t);
#endif
      }

    tend = MPI_Wtime();
    if (hc->comm_id == 0)
      {
        printf("[0] ");
        print_result(NUM_MASK_EVENTS, n_time_steps, tend - tstart);
        printf("[0] mpi_wait_time = %lf\n", hc->mpi_wait_time);
      }

    output_final_state(hc, n_time_steps);
    comm_finalize();

    return 0;
  }

int output_partial_state(PVHyperCol* hc, int time_step)
  {
    int i;
    char filename[64];
    float phimax = -100000.0;
    
    // TODO - fix struct PVLayer declaration in PVHyperCol (circular include reference)
    PVLayer* l = (PVLayer*) hc->layer[1];
    float* phi = l->phi;

    //sprintf(filename, "phi%d", time_step);
    sprintf(filename, "phi");
    for (i = 0; i < N; i++)
      {
        if (phimax < phi[i])
          phimax = phi[i];
      }

    pv_output(filename, phimax/2., hc->x0, hc->y0, l->x, l->y, l->o, phi);
    pv_output_on_circle(time_step, "phi", 1.0, phi);

    return 0;
  }

int output_state(PVHyperCol* hc, int time_step)
  {
    int i;
    char filename[64];
    float fave = 0.0;
    float Vave = 0.0;
    /*     float Vmax = -100000.0; */
    float phimax = -100000.0;

    PVLayer* l = (PVLayer*) hc->layer[1];
    
    float* phi = l->phi;
    float* f = l->f;
    float* V = l->V;
    //float* h = l->h;
    //float* H = l->H;

    /* save event mask */

    // TODO - fix storage of events
    //int offset = time_step * N/8;
    //compress_float_mask(N, f, &s->event_store[offset]);

    /* graphics output */

    for (i = 0; i < N; i++)
      {
        if (phimax < phi[i])
          phimax = phi[i];
      }
    for (i = 0; i < N; i++)
      {
        Vave += V[i];
      }
    for (i = 0; i < N; i++)
      {
        fave += f[i];
      }

    //pv_output_on_circle(time_step, "V  ", 0.6, V);
    //pv_output_events_on_circle(time_step, f, h);

    //sprintf(filename, "f%d_%d", time_step, hc->comm_id);
    sprintf(filename, "f%d", hc->comm_id);
    pv_output(filename, 0.5, hc->x0, hc->y0, l->x, l->y, l->o, f);

    //sprintf(filename, "V%d_%d", time_step, hc->comm_id);
    sprintf(filename, "V%d", hc->comm_id);
    pv_output(filename, -1000., hc->x0, hc->y0, l->x, l->y, l->o, V);

    //sprintf(filename, "h%d_%d", time_step, hc->comm_id);
    //sprintf(filename, "h%d", hc->comm_id);
    //pv_output(filename, 0.5, hc->x0, hc->y0, l->x, l->y, l->o, h);

    //sprintf(filename, "Vinh%d_%d", time_step, hc->comm_id);
    //sprintf(filename, "Vinh%d", hc->comm_id);
    //pv_output(filename, -1000., hc->x0, hc->y0, l->x, l->y, l->o, H);

    /*     //    sprintf(filename, "phi%d_%d", time_loop, hc->comm_id); */
    /*     sprintf(filename, "./output/phi%d", hc->comm_id); */
    /*     pv_output(filename, -1000., hc->loc.x0, hc->loc.y0, hc->loc.x, hc->loc.y, hc->loc.o, phi); */

    printf("loop=%d:  fave=%f, Vave=%f\n", time_step, 1000*fave/N, Vave/N);
  }

int output_final_state(PVHyperCol* hc, int nsteps)
  {
    int i;
    size_t count, size;
    char filename[64];
    float fmax = 1.0;
    unsigned char* recv_buf;
    FILE* fp= NULL;

    PVLayer* l = (PVLayer*) hc->layer[1];
    float* f = l->f;

    //    sprintf(filename, "f%d_%d", time_loop, hc->comm_id);
    sprintf(filename, "f%d", hc->comm_id);
    pv_output(filename, fmax/2., hc->x0, hc->y0, l->x, l->y, l->o, f);

    /* gather event masks and dump them one time step at a time */

    size = N/8;

    if (hc->comm_id == 0)
      {
        sprintf(filename, "%s/events.bin", OUTPUT_PATH);
        fp = fopen(filename, "w");
        if (fp == NULL)
          {
            fprintf(stderr, "ERROR:output_final_state: error opening events.bin\n");
            return 1;
          }

        /* allocate memory for ALL processors (one bit per event) */
        recv_buf = (unsigned char*) malloc(size*hc->comm_size);
        if (recv_buf == NULL)
          {
            fprintf(stderr, "ERROR:output_final_state: error malloc of output buffer of size %ld\n", size);
            return 1;
          }
      }

    for (i = 0; i < nsteps; i++)
      {
        if (DEBUG)
          {
            printf("[%d] output_final_state: gather of size %d\n", hc->comm_id,
                (int) size);
          }
        // MPI_Gather(&hc->event_store[i*size], size, MPI_CHAR, recv_buf, size, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (hc->comm_id == 0)
          {
            count = fwrite(recv_buf, sizeof(unsigned char), size*hc->comm_size,
                fp);
            if (count != size*hc->comm_size)
              {
                fprintf(stderr, "ERROR:output_final_state: error writing output buffer of size %ld\n",
                size*hc->comm_size);
              }
          }
      }

    if (hc->comm_id == 0)
      {
        fclose(fp);
      }
    return 0;
  }
