#include <src/columns/PVHyperCol.h>
#include <mpi.h>
#include <stdio.h>


int comm_init(int* pargc, char*** pargv, int* rank, int* size)
{
     int err = MPI_Init(pargc, pargv);
     err = MPI_Comm_rank(MPI_COMM_WORLD, rank);
     err = MPI_Comm_size(MPI_COMM_WORLD, size);
     return err;
}


int comm_finalize()
{
     return MPI_Finalize();
}


/**
 *  Send event state to each connected column
 */

int send_columns(PVHyperCol* hc, MPI_Request* req)
{
    int c;
    int err = 0;
    int tag = 33;

    /* post the receives as soon as possible to provide a recv buffer */
    for (c = 0; c < hc->n_neighbors; c++) {
        eventmask* emk = column_event(hc, c);
	if (DEBUG) {
	  printf("[%d] send_state: recv eventmask is %p, neighbor index %d, neighbor rank is %d\n",
		 hc->comm_id, emk->event, c, hc->neighbors[c]);
	}
	err = MPI_Irecv(emk, NUM_MASK_EVENTS, EVENT_TYPE_MPI,
			hc->neighbors[c], tag, MPI_COMM_WORLD, &req[c]);
	if (err) error_msg(hc, err, "send_state:recv");
    }

    /* send my event state to all connected columns */
    for (c = 0; c < hc->n_neighbors; c++) {
        if (DEBUG) {
	  printf("[%d] send_state: send eventmask is %p, neighbor index %d, neighbor rank is %d\n",
		 hc->comm_id, &(hc->remote_events[0].event[0]), c, hc->neighbors[c]);
	}
	err = MPI_Send(&(hc->remote_events[0]), NUM_MASK_EVENTS, EVENT_TYPE_MPI,
		       hc->neighbors[c], tag, MPI_COMM_WORLD);
	if (err) error_msg(hc, err, "send_state:send");
    }
    return err;
}


/**
 * Wait until some (any) event comes in.  Return the column id of sender.
 */
int recv_columns(PVHyperCol* hc, MPI_Request* req, int* conn_id)
{
    int err;
    MPI_Status status;
    double tstart, tend;
    tstart = MPI_Wtime();
    err = MPI_Waitany(hc->n_neighbors, req, conn_id, &status);
    tend = MPI_Wtime();
    hc->mpi_wait_time += tend - tstart;
    if (err) error_msg(hc, err, "recv_state");
    if (DEBUG) printf("[%d] recv_state: index %d, wait_time = %lf\n",
		      hc->comm_id, *conn_id, tend-tstart);
    return err;
}
