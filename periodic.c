/*
Original code - Blaise Barney

Modifications - Soumyadip Ghosh
*/

#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define L 1.0 /* linear size of square region */
#define DIPOLE_STRENGTH 100.0
#define BEGIN 1  /* message tag */
#define LTAG 2   /* message tag */
#define RTAG 10  /* message tag */
#define NONE -1  /* indicates no neighbor */
#define DONE 5   /* message tag */
#define MASTER 0 /* taskid of first process */
#define TOL 1e-5 /* tolerance for convergence */

void inidat();
void update(int, int, int, int, int, int, long int, double, double *, double *,
            double *, double *);
double calc_residual_sum(int, int, int, double *);
double calc_residual_max(int, int, int, double *);

int main(int argc, char *argv[]) {
  int NXPROB = atoi(argv[1]); // dimension of square
  int NSTEP = atoi(argv[2]);  // period of halo exchange and All_reduce
  // void inidat(), update();
  // double calc_residual_sum(), calc_residual_max();

  double *u;        /* array for grid */
  double *psource;  /* value at last sync point */
  double *residual; /* matrix of exact steady state solution */

  int taskid = 0,   /* this task's unique id */
      numtasks = 0, /* number of tasks */
      averow = 0, rows = 0, offset = 0,
      extra = 0,            /* for sending rows of data */
      dest = 0, source = 0, /* to - from for message send-receive */
      left = 0, right = 0,  /* neighbor tasks */
      msgtype = 0,          /* for message types */
      start = 0, end = 0,   /* misc */
      i = 0, ix = 0, iy = 0, iz = 0, it = 0; /* loop variables */
  long int wsteps = 0;
  double exact_value = 0.0; /* exact value of norm */
  MPI_Status status;
  MPI_Request req1, req2;
  double tstart = 0.0, tend = 0.0;
  double cpu_time_used = 0.0;
  double h = L / (NXPROB + 1);
  int pos_loc = NXPROB / 2;
  int neg_loc = NXPROB / 4;
  char name[30], res[30], period_str[2], grid_str[10], proc_str[4];

  // iteration error
  double local_avg_residual = 0.0;
  double local_max_residual = 0.0;
  double global_avg_residual = 0.0;
  double global_max_residual = 0.0;

  // allocate memory to arrays
  u = (double *)malloc(2 * NXPROB * NXPROB * sizeof(double));
  psource = (double *)malloc(NXPROB * NXPROB * sizeof(double));
  residual = (double *)malloc(NXPROB * NXPROB * sizeof(double));

  /* First, find out my taskid and how many tasks are running */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  // initialize all arrays to zero for every process
  for (ix = 0; ix < NXPROB; ix++) {
    for (iy = 0; iy < NXPROB; iy++) {
      *(u + 0 * NXPROB * NXPROB + ix * NXPROB + iy) = 0.0;
      *(u + 1 * NXPROB * NXPROB + ix * NXPROB + iy) = 0.0;
      *(psource + ix * NXPROB + iy) = 0.0;
      *(residual + ix * NXPROB + iy) = 0.0;
    }
  }

  if (taskid == MASTER) {
    printf("Starting mpi_poisson2D with %d processors.\n", numtasks);

    /* Initialize grid */
    printf("Grid size: X = %d Y = %d Communication step= %d\n", NXPROB, NXPROB,
           NSTEP);
    printf("Initializing grid and writing initial.dat file...\n");
    inidat(NXPROB, NXPROB, psource, pos_loc,
           neg_loc); // for first block of 3rd dimension

    /* Distribute work to workers.  Must first figure out how many rows to */
    /* send and what to do with extra rows.  */
    averow = NXPROB / numtasks;
    extra = NXPROB % numtasks;
    offset = 0;
    rows = averow;

    tstart = MPI_Wtime(); // start measuring time

    for (i = 1; i < numtasks; i++) {
      // rows = (i <= extra) ? averow+1 : averow; //consider uniform grid
      // division
      offset = offset + rows;
      /* Tell each worker who its neighbors are, since they must exchange */
      /* data with each other. */
      left = i - 1;
      if (i == numtasks - 1)
        right = NONE; // rightmost processor has no right neighbor
      else
        right = i + 1;

      /*  Now send startup information to all processors above rank 1  */
      dest = i;
      MPI_Send(&offset, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send(&left, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send(&right, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send((psource + offset * NXPROB), rows * NXPROB, MPI_DOUBLE, dest,
               BEGIN, MPI_COMM_WORLD);

      // printf("Sent to task %d: rows= %d offset= %d ",dest,rows,offset);
      // printf("left= %d right= %d\n",left,right);
    }

    // values for MASTER
    left = NONE;
    if (numtasks > 1)
      right = 1;
    else // if there is only 1 PE
      right = NONE;

    offset = 0;
    // rows is considered same as others for uniform division

  } /* End of domain decomposition part of master code */

  if (taskid != MASTER) // only other processors need to receive data
  {
    /* Receive my offset, rows, neighbors and grid partition from master */
    source = MASTER;
    msgtype = BEGIN;
    MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&left, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&right, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    MPI_Recv((psource + offset * NXPROB), rows * NXPROB, MPI_DOUBLE, source,
             msgtype, MPI_COMM_WORLD, &status);
  }

  // common code starts here

  /* Determine border elements.  Need to consider first and last columns. */
  /* Obviously, row 0 can't exchange with row 0-1. Likewise, the last */
  /* row can't exchange with last+1.  */
  start = offset;
  end = offset + rows - 1;
  if (offset == 0)
    start = 1;
  if ((offset + rows) == NXPROB)
    end--;

  iz = 0;

  sprintf(period_str, "%d", NSTEP);
  sprintf(proc_str, "%d", numtasks);
  sprintf(grid_str, "%d", NXPROB);

  /*
  //file to write time evolution of residual
  FILE *fptr;
    strcpy(res,"res");
    strcat(res,period_str);
    strcat(res,proc_str);
    strcat(res,grid_str);
    strcat(res,".dat");

  if(taskid == MASTER) fptr = fopen(res,"w");
   */

  do {
    if (left != NONE && wsteps % NSTEP == 0) {
      MPI_Issend((u + iz * NXPROB * NXPROB + offset * NXPROB), NXPROB,
                 MPI_DOUBLE, left, RTAG * 1, MPI_COMM_WORLD, &req1);
      source = left;
      msgtype = LTAG;
      MPI_Recv((u + iz * NXPROB * NXPROB + (offset - 1) * NXPROB), NXPROB,
               MPI_DOUBLE, source, msgtype * 1, MPI_COMM_WORLD, &status);
      for (iy = 0; iy < NXPROB; iy++) {
        *(u + (1 - iz) * NXPROB * NXPROB + (offset - 1) * NXPROB + iy) =
            *(u + iz * NXPROB * NXPROB + (offset - 1) * NXPROB + iy);
      }
      MPI_Wait(&req1, &status);
    }
    if (right != NONE && wsteps % NSTEP == 0) {
      MPI_Issend((u + iz * NXPROB * NXPROB + (offset + rows - 1) * NXPROB),
                 NXPROB, MPI_DOUBLE, right, LTAG * 1, MPI_COMM_WORLD, &req2);
      source = right;
      msgtype = RTAG;
      MPI_Recv((u + iz * NXPROB * NXPROB + (offset + rows) * NXPROB), NXPROB,
               MPI_DOUBLE, source, msgtype * 1, MPI_COMM_WORLD, &status);
      for (iy = 0; iy < NXPROB; iy++) {
        *(u + (1 - iz) * NXPROB * NXPROB + (offset + rows) * NXPROB + iy) =
            *(u + iz * NXPROB * NXPROB + (offset + rows) * NXPROB + iy);
      }
      MPI_Wait(&req2, &status);
    }

    update(start, end, NXPROB, pos_loc, neg_loc, NSTEP, wsteps, h,
           (u + iz * NXPROB * NXPROB), (u + (1 - iz) * NXPROB * NXPROB),
           psource, residual);

    if (wsteps % NSTEP == 0) {
      local_avg_residual =
          calc_residual_sum(start, end, NXPROB, residual); // sum for this PE
      local_max_residual =
          calc_residual_max(start, end, NXPROB, residual); // max for this PE

      // MPI_Allreduce(&local_avg_residual, &global_avg_residual, 1, MPI_DOUBLE,
      // MPI_SUM, MPI_COMM_WORLD); //sum of error for entire grid
      // global_avg_residual = global_avg_residual/(NXPROB*NXPROB);

      MPI_Allreduce(&local_max_residual, &global_max_residual, 1, MPI_DOUBLE,
                    MPI_MAX, MPI_COMM_WORLD); // sum of error for entire grid

      // if(taskid == MASTER) fprintf(fptr, "%d %f\n", wsteps,
      // global_max_residual);
    }

    // if(taskid == 0) printf("Avg residual - %f, Max residual - %2.6f\n",
    // global_avg_residual, global_max_residual);
    wsteps = wsteps + 1;
    iz = 1 - iz;
  } while (global_max_residual > TOL); // end while

  // printf("Out of loop for proc %d after %d iterations\n",taskid, wsteps-1);
  // if(taskid == MASTER) fclose(fptr);

  if (taskid != MASTER) {
    /* Finally, send my portion of final results back to master */
    MPI_Send(&offset, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
    MPI_Send((u + iz * NXPROB * NXPROB + offset * NXPROB), rows * NXPROB,
             MPI_DOUBLE, MASTER, DONE, MPI_COMM_WORLD);
    MPI_Send((residual + offset * NXPROB), rows * NXPROB, MPI_DOUBLE, MASTER,
             DONE, MPI_COMM_WORLD);
  }

  // now get values from remaining processes
  if (taskid == MASTER) {
    /* Now wait for results from all worker tasks */
    for (i = 1; i < numtasks; i++) {
      source = i;
      msgtype = DONE;
      MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
      MPI_Recv((u + 0 * NXPROB * NXPROB + offset * NXPROB), rows * NXPROB,
               MPI_DOUBLE, source, msgtype, MPI_COMM_WORLD, &status);
      MPI_Recv((residual + offset * NXPROB), rows * NXPROB, MPI_DOUBLE, source,
               msgtype, MPI_COMM_WORLD, &status);
    }

    printf("No of steps - %ld\n", wsteps);
    printf("Sync points - %ld\n", (wsteps - 1) / NSTEP);
    tend = MPI_Wtime();
    cpu_time_used = (tend - tstart);

    printf("Time measured: %2.4f\n", cpu_time_used);

    /*
    FILE *fp1;
    fp1 = fopen("perresidual.dat","w");
    for(ix = 0;ix < NXPROB; ix++)
    {
        for(iy = 0;iy < NXPROB; iy++)
        {
            //fprintf(fp,"%d %d %f\n", ix, iy, *(u+ix*NXPROB+iy));
            fprintf(fp1,"%f ", *(residual+ix*NXPROB+iy));
        }
        fprintf(fp1,"\n");
    }
    fclose(fp1);
    */

    // file with just end "steady state" value
    strcpy(name, "pervalues");
    strcat(name, period_str);
    strcat(name, proc_str);
    strcat(name, grid_str);
    strcat(name, ".dat");

    FILE *fp;
    fp = fopen(name, "w");
    for (ix = 0; ix < NXPROB; ix++) {
      for (iy = 0; iy < NXPROB; iy++) {
        // fprintf(fp,"%d %d %f\n", ix, iy, *(u+ix*NXPROB+iy));
        fprintf(fp, "%2.12f ", *(u + ix * NXPROB + iy));
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  } // end of value gathering of master code
  // MPI_Comm_free ;

  MPI_Finalize();
  // free allocated arrays
  free(u);
  free(psource);
  free(residual);
  // MPI_Finalize();
} /*end of main*/

/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void update(int start, int end, int ny, int pos_loc, int neg_loc, int NSTEP,
            long int wsteps, double h, double *u1, double *u2, double *psource,
            double *residual) {
  int ix, iy;
  for (ix = start; ix <= end; ix++) {
    for (iy = 1; iy <= ny - 2; iy++) {
      // if(ix == pos_loc && iy == pos_loc) continue;
      // if(ix == neg_loc && iy == neg_loc) continue;
      *(u2 + ix * ny + iy) =
          0.25 * (*(u1 + (ix + 1) * ny + iy) + *(u1 + (ix - 1) * ny + iy) +
                  *(u1 + ix * ny + iy + 1) + *(u1 + ix * ny + iy - 1) -
                  h * h * *(psource + ix * ny + iy));

      if (wsteps % NSTEP == 0) {
        *(residual + ix * ny + iy) =
            (*(u1 + (ix + 1) * ny + iy) + *(u1 + (ix - 1) * ny + iy) +
             *(u1 + ix * ny + iy + 1) + *(u1 + ix * ny + iy - 1) -
             4 * *(u1 + ix * ny + iy)) /
                (h * h) -
            *(psource + ix * ny + iy);
      }
      // if(*(residual+ix*ny+iy) != 0.0) printf("Non-zero residual -
      // %f\n",*(residual+ix*ny+iy));
    }
  }
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, double *psource, int pos_loc, int neg_loc) {
  int ix, iy;

  // make source
  for (ix = 0; ix <= nx - 1; ix++) {
    for (iy = 0; iy <= ny - 1; iy++) {
      if (ix == pos_loc && iy == neg_loc)
        *(psource + ix * ny + iy) = DIPOLE_STRENGTH;
      else if (ix == pos_loc && iy == 3 * neg_loc)
        *(psource + ix * ny + iy) = -DIPOLE_STRENGTH;
      else
        *(psource + ix * ny + iy) = 0.0;
    }
  }
}

// for avg error (mat is both iteration error as well as ss error at any time
// step
double calc_residual_sum(int start, int end, int ny, double *mat) {
  int ix, iy;
  double res_sum = 0.0;
  for (ix = start; ix <= end; ix++) {
    for (iy = 1; iy <= ny - 2; iy++)
      res_sum = res_sum + *(mat + ix * ny + iy); // sum
  }
  return res_sum;
}

// for max error (mat is both iteration error as well as ss error at any time
// step
double calc_residual_max(int start, int end, int ny, double *mat) {
  int ix, iy;
  double res_max = 0.0;
  for (ix = start; ix <= end; ix++) {
    for (iy = 1; iy <= ny - 2; iy++) {
      if (fabs(*(mat + ix * ny + iy)) > res_max)
        res_max = *(mat + ix * ny + iy);
    }
  }
  return res_max;
}
