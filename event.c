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

#define L 1.0                 /* linear size of square region */
#define DIPOLE_STRENGTH 100.0 /* dipole strength */
#define BEGIN 1               /* message tag */
#define LTAG 2                /* message tag */
#define RTAG 10               /* message tag */
#define NONE -1               /* indicates no neighbor */
#define DONE 5                /* message tag */
#define MASTER 0              /* taskid of first process */
#define TOL 1e-8              /* tolerance for convergence */

void inidat(int, int, double *, int, int);
void update(int, int, int, int, int, double, double *, double *, double *,
            double *);
double calc_residual_sum(int, int, int, double *);
double calc_residual_max(int, int, int, double *);

int main(int argc, char *argv[]) {
  int NXPROB = atoi(argv[1]);
  double thres = atof(argv[2]);
  int NSTEP = atoi(argv[3]);

  double *u;        /* array for grid */
  double *psource;  /* value at last sync point */
  double *residual; /* matrix of exact steady state solution */
  double *prev_left_value;
  double *prev_right_value;

  int taskid = 0,   /* this task's unique id */
      numtasks = 0, /* number of tasks */
      averow = 0, rows = 0, offset = 0,
      extra = 0,            /* for sending rows of data */
      dest = 0, source = 0, /* to - from for message send-receive */
      left = 0, right = 0,  /* neighbor tasks */
      msgtype = 0,          /* for message types */
      start = 0, end = 0,   /* misc */
      i = 0, ix = 0, iy = 0, iz = 0, it = 0; /* loop variables */
  int left_send_count = 0, right_send_count = 0;

  long int wsteps = 0;
  double exact_value = 0.0; /* exact value of norm */
  double tstart = 0.0, tend = 0.0;
  double cpu_time_used = 0.0;
  double h = L / (NXPROB + 1);
  int pos_loc = NXPROB / 2;
  int neg_loc = NXPROB / 4;
  char name[30], res[30], thres_str[2], grid_str[10], proc_str[4];
  double old_value = 0.0, curr_value = 0.0; // temporary variables
  int left_flag = 0, right_flag = 0;
  int thres_order = 0;

  MPI_Status status;
  MPI_Request req;

  double *win_mem;
  MPI_Win win;

  // iteration error
  double local_avg_residual = 0.0;
  double local_max_residual = 0.0;
  double global_avg_residual = 0.0;
  double global_max_residual = 0.0;

  // allocate memory to arrays
  u = (double *)malloc(2 * NXPROB * NXPROB * sizeof(double));
  psource = (double *)malloc(NXPROB * NXPROB * sizeof(double));
  residual = (double *)malloc(NXPROB * NXPROB * sizeof(double));
  prev_left_value = (double *)malloc(NXPROB * sizeof(double));
  prev_right_value = (double *)malloc(NXPROB * sizeof(double));

  /* First, find out my taskid and how many tasks are running */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

  // create memory window for MPI RMA
  win_mem = (double *)calloc(2 * NXPROB, 2 * NXPROB * sizeof(double));
  MPI_Win_create(win_mem, 2 * NXPROB * sizeof(double), sizeof(double),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  // initialize all arrays to zero for every process
  for (ix = 0; ix < NXPROB; ix++) {
    for (iy = 0; iy < NXPROB; iy++) {
      *(u + 0 * NXPROB * NXPROB + ix * NXPROB + iy) = 0.0;
      *(u + 1 * NXPROB * NXPROB + ix * NXPROB + iy) = 0.0;
      *(psource + ix * NXPROB + iy) = 0.0;
      *(residual + ix * NXPROB + iy) = 0.0;
    }
    *(prev_left_value + ix) = 0.0;
    *(prev_right_value + ix) = 0.0;
  }

  if (taskid == MASTER) {
    printf("Starting mpi_poisson2D with %d processors.\n", numtasks);

    /* Initialize grid */
    printf("Grid size: X = %d Y = %d Threshold = %2.18f Period = %d\n", NXPROB,
           NXPROB, thres, NSTEP);
    printf("Initializing grid and writing initial.dat file...\n");
    inidat(NXPROB, NXPROB, psource, pos_loc,
           neg_loc); // for first block of 3rd dimension

    /* Distribute work to workers.  Must first figure out how many rows to */
    /* send and what to do with extra rows. */
    averow = NXPROB / numtasks;
    extra = NXPROB % numtasks;
    offset = 0;

    tstart = MPI_Wtime(); // start measuring time

    for (i = 1; i < numtasks; i++) {
      rows = (i <= extra) ? averow + 1 : averow;
      offset = offset + rows;
      /* Tell each worker who its neighbors are, since they must exchange */
      /* data with each other. */
      left = i - 1;
      if (i == numtasks - 1)
        right = NONE;
      else
        right = i + 1;

      /*  Now send startup information to each worker  */
      dest = i;
      MPI_Send(&offset, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send(&left, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send(&right, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
      MPI_Send((psource + offset * NXPROB), rows * NXPROB, MPI_DOUBLE, dest,
               BEGIN, MPI_COMM_WORLD);
    }

    // values for MASTER
    left = NONE;
    right = 1;
    offset = 0;
    // rows is considered same as others for uniform division

  } // end of domain decomposition part of master code

  if (taskid != MASTER) // only other processors need to receive data
  {
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
  /* Obviously, row 0 can't exchange with row 0-1.  Likewise, the last */
  /* row can't exchange with last+1.  */
  start = offset;
  end = offset + rows - 1;
  if (offset == 0)
    start = 1;
  if ((offset + rows) == NXPROB)
    end--;

  iz = 0;

  if (thres == 0)
    thres_order = 0;
  else
    thres_order = abs(log10(thres));
  sprintf(thres_str, "%d", thres_order);
  sprintf(proc_str, "%d", numtasks);
  sprintf(grid_str, "%d", NXPROB);

  /*
  //file to write time evolution of residual
  FILE *fptr;
  strcpy(res,"res");
  strcat(res,thres_str);
  strcat(res,proc_str);
  strcat(res,grid_str);
  strcat(res,".dat");

  if(taskid == MASTER) fptr = fopen(res, "w");
  */
  do {
    // sender side
    if (right != NONE) {
      for (iy = 0; iy < NXPROB; iy++) {
        curr_value =
            *(u + iz * NXPROB * NXPROB + (offset + rows - 1) * NXPROB + iy);
        old_value = *(prev_right_value + iy);
        if ((fabs(curr_value - old_value) >= thres) || wsteps == 0) {
          right_flag = 1;
          break;
        }
      }

      if (right_flag == 1) {
        for (iy = 0; iy < NXPROB; iy++) {
          curr_value =
              *(u + iz * NXPROB * NXPROB + (offset + rows - 1) * NXPROB + iy);
          *(prev_right_value + iy) = curr_value;
        }

        MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win);
        MPI_Put((u + iz * NXPROB * NXPROB + (offset + rows - 1) * NXPROB),
                NXPROB, MPI_DOUBLE, right, 0, NXPROB, MPI_DOUBLE,
                win); // to left ghost cells
        // for(iy = 0; iy < NXPROB; iy++)
        // MPI_Put((u+iz*NXPROB*NXPROB+(offset+rows-1)*NXPROB+iy), 1, MPI_DOUBLE,
        // right, iy, 1, MPI_DOUBLE, win);

        MPI_Win_unlock(right, win);
        right_send_count++;
        /*
        else
        {
            printf("This is right msg from PE %d in iter %ld - grid point -
        %d\n",taskid,wsteps,iy);
        }
        */
      }
    }

    if (left != NONE) {
      for (iy = 0; iy < NXPROB; iy++) {
        curr_value = *(u + iz * NXPROB * NXPROB + offset * NXPROB + iy);
        old_value = *(prev_left_value + iy);
        if ((fabs(curr_value - old_value) >= thres) || wsteps == 0) {
          left_flag = 1;
          break;
        }
      }

      if (left_flag == 1) {
        for (iy = 0; iy < NXPROB; iy++) {
          curr_value = *(u + iz * NXPROB * NXPROB + offset * NXPROB + iy);
          *(prev_left_value + iy) = curr_value;
        }

        MPI_Win_lock(MPI_LOCK_SHARED, left, 0, win);
        MPI_Put((u + iz * NXPROB * NXPROB + offset * NXPROB), NXPROB,
                MPI_DOUBLE, left, NXPROB, NXPROB, MPI_DOUBLE, win);
        // to right ghost cells
        // for(iy = 0; iy < NXPROB; iy++)
        // MPI_Put((u+iz*NXPROB*NXPROB+offset*NXPROB+iy), 1, MPI_DOUBLE, left,
        // (NXPROB+iy), 1, MPI_DOUBLE, win);
        MPI_Win_unlock(left, win);
        left_send_count++;
        /*
        else
        {
            printf("This is left msg from PE %d in iter %ld - grid point -
        %d\n",taskid,wsteps,iy);
        }
        */
      }
    }

    // receiver side - this is not needed but we copy the shared memory values
    // to the actual variables
    for (iy = 0; iy < NXPROB; iy++) {
      // copy to left ghost cells
      if (left != NONE) {
        *(u + iz * NXPROB * NXPROB + (offset - 1) * NXPROB + iy) =
            *(win_mem + iy);
        *(u + (1 - iz) * NXPROB * NXPROB + (offset - 1) * NXPROB + iy) =
            *(u + iz * NXPROB * NXPROB + (offset - 1) * NXPROB + iy);
      }

      // copy to right ghost cells
      if (right != NONE) {
        *(u + iz * NXPROB * NXPROB + (offset + rows) * NXPROB + iy) =
            *(win_mem + (NXPROB + iy));
        *(u + (1 - iz) * NXPROB * NXPROB + (offset + rows) * NXPROB + iy) =
            *(u + iz * NXPROB * NXPROB + (offset + rows) * NXPROB + iy);
      }
    }

    // printf("This is proc %d in %d iteration: local max iter error: %2.5f,
    // left flag-%d, right flag-%d\n",
    //                                               taskid,wsteps,local_max_iter_error,left_flag,right_flag);
    // if(taskid == 0) printf("In iteration %ld\n",wsteps);

    update(start, end, NXPROB, pos_loc, neg_loc, h, (u + iz * NXPROB * NXPROB),
           (u + (1 - iz) * NXPROB * NXPROB), psource,
           residual); // first update values

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

    wsteps = wsteps + 1;
    iz = 1 - iz;
    right_flag = 0;
    left_flag = 0;
  } while (global_max_residual > TOL);

  printf("This is PE %d - left sends - %d, right sends - %d\n", taskid,
         left_send_count, right_send_count);
  // if(taskid == MASTER) fclose(fptr);

  if (taskid != MASTER) {
    MPI_Send(&offset, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
    MPI_Send((u + iz * NXPROB * NXPROB + offset * NXPROB), rows * NXPROB,
             MPI_DOUBLE, MASTER, DONE, MPI_COMM_WORLD);
    MPI_Send((residual + offset * NXPROB), rows * NXPROB, MPI_DOUBLE, MASTER,
             DONE, MPI_COMM_WORLD);
  }

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
    fp1 = fopen("eventresidual.dat","w");
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
    strcpy(name, "eventvalues");
    strcat(name, thres_str);
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

  } // end of value gathering part of MASTER code
  // MPI_Win_free(&win);
  // MPI_Finalize();
  // free(u);
  // free(psource);
  // free(residual);
  // free(prev_left_value);
  // free(prev_right_value);
  MPI_Win_free(&win);
  MPI_Finalize();
} /*end of main*/

/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void update(int start, int end, int ny, int pos_loc, int neg_loc, double h,
            double *u1, double *u2, double *psource, double *residual) {
  int ix, iy;
  for (ix = start; ix <= end; ix++) {
    for (iy = 1; iy <= ny - 2; iy++) // by not calculating for boundary values,
                                     // we impose boundary conditions
    {
      // if(ix == pos_loc && iy == pos_loc) continue; //dont calculate for
      // positive dipole if(ix == neg_loc && iy == neg_loc) continue; //dont
      // calculate for negative dipole
      *(u2 + ix * ny + iy) =
          0.25 * (*(u1 + (ix + 1) * ny + iy) + *(u1 + (ix - 1) * ny + iy) +
                  *(u1 + ix * ny + iy + 1) + *(u1 + ix * ny + iy - 1) -
                  h * h * *(psource + ix * ny + iy));

      // if(wsteps % NSTEP == 0)
      //{
      *(residual + ix * ny + iy) =
          (*(u1 + (ix + 1) * ny + iy) + *(u1 + (ix - 1) * ny + iy) +
           *(u1 + ix * ny + iy + 1) + *(u1 + ix * ny + iy - 1) -
           4 * *(u1 + ix * ny + iy)) /
              (h * h) -
          *(psource + ix * ny + iy);
      //}
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
