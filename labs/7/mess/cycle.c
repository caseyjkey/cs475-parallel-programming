#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>


/* cycle
   In this program a message is sent around a circle of processes 0 - (p-1), ]
   where p-1 sends to 0.
*/



int main(int argc, char **argv) {
    int my_id, next_id;	/* process id-s */
    int p;		/* number of processes */
    char* message;	/* storage for the message */
    int i, k, max_msgs, msg_size, v;
    MPI_Status status;	/* return status for receive */


    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_id );
    MPI_Comm_size( MPI_COMM_WORLD, &p );

    if (argc < 3)
	{
	fprintf (stderr, "need msg count and msg size as params\n");
	goto EXIT;
	}

    if ((sscanf (argv[1], "%d", &max_msgs) < 1) ||
        		(sscanf (argv[2], "%d", &msg_size) < 1))
	{
	fprintf (stderr, "need msg count and msg size as params\n");
	goto EXIT;
	}

    message = (char*)malloc (msg_size);
    if (argc > 3) v=1; else v=0;           /*are we in verbose mode*/

    /* don't start timer until everybody is ok */
    MPI_Barrier(MPI_COMM_WORLD); 

    if( my_id == 0 ) {
        // do max_msgs times:
        //   send message of size msg_size chars to process 1
        //   receive message of size msg_size chars from process p-1

        MPI_Barrier(MPI_COMM_WORLD); 
        printf("Number, size of messages: %3d , %3d \n", max_msgs, msg_size);
        fflush(stdout);
    } else {

        // do max_msgs times:
        //   receive message of size msg_size from process to the left
        //   send message of size msg_size to process to the right (p-1 sends to 0)


      MPI_Barrier(MPI_COMM_WORLD); 
    }	    

EXIT:
    MPI_Finalize();
    return 0;
}

