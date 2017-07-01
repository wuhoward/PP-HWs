#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct
{
	int adj_n;
	int pre;
	int min_d;
	int *adj;
	int *w;
}vertex;

int main (int argc, char *argv[]) {
	int task_n, rank, vertex_s, vertex_n, edge_n, i, pre_f, color, send, recv, s_count, r_count;
	vertex *vertices;
	int *work_q;
	int *pre_res;
	double t_comm, t_cal, t, t_in;
	FILE *fh;

	vertex_s = atof(argv[4])-1;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &task_n);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//t = MPI_Wtime();
	fh=fopen(argv[2],"r");
	fscanf(fh,"%d%d", &vertex_n, &edge_n);

	//memory allocation
	vertices = malloc(vertex_n*sizeof(vertex));
	pre_res = malloc(vertex_n*sizeof(int));
	work_q = malloc(vertex_n*sizeof(int));
	for (i=0; i<vertex_n; i++) vertices[i].adj = malloc(vertex_n*sizeof(int));
	for (i=0; i<vertex_n; i++) vertices[i].w = malloc(vertex_n*sizeof(int));

	//global variables initialization
	for (i=0; i<vertex_n; i++){
		vertices[i].adj_n = 0;
		vertices[i].min_d = INT_MAX;
		work_q[i] = -1;	
	}	
	vertices[vertex_s].min_d = 0;
	vertices[vertex_s].pre = -1;
	work_q[0] = vertex_s;
	color = 0;

	//reading input
	int x, y, z;
	for (i=0; i<edge_n; i++){
		fscanf(fh,"%d%d%d",&x,&y,&z);
		x--;
		y--;
		vertices[x].adj[vertices[x].adj_n]=y;
		vertices[x].w[vertices[x].adj_n]=z;
		vertices[x].adj_n++;
		vertices[y].adj[vertices[y].adj_n]=x;
		vertices[y].w[vertices[y].adj_n]=z;
		vertices[y].adj_n++;
	}
	fclose(fh);
	//printf("Input Time: %f @rank%d\n", MPI_Wtime() - t, rank);
	//t_in = MPI_Wtime() - t;

	s_count = 0;
	r_count = 0;
	t_comm = 0;
	//t_cal = MPI_Wtime();
	if(rank == vertex_s){
		for(i=0; i<vertices[rank].adj_n; i++){
			send = vertices[rank].w[i];
			//s_count++;
			//t = MPI_Wtime();
			MPI_Send(&send, 1, MPI_INT, vertices[rank].adj[i], 0, MPI_COMM_WORLD);
			//t_comm += MPI_Wtime() - t;
		}
	}
	if(rank == 0 && task_n>1){
		send = 0; 
		//s_count++;
		//t = MPI_Wtime();
		MPI_Send(&send, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
		//t_comm += MPI_Wtime() - t;
	}
	while(1){
		r_count++;
		//t = MPI_Wtime();
		MPI_Recv(&recv, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		//t_comm += MPI_Wtime() - t;
		if(status.MPI_TAG == 0){
			if(recv < vertices[rank].min_d){
				vertices[rank].min_d = recv;
				vertices[rank].pre = status.MPI_SOURCE;
				for(i=0; i<vertices[rank].adj_n; i++){
					if(vertices[rank].adj[i]<rank)color=1;
					send = vertices[rank].min_d+vertices[rank].w[i];
					//s_count++;
					//t = MPI_Wtime();
					MPI_Send(&send, 1, MPI_INT, vertices[rank].adj[i], 0, MPI_COMM_WORLD);
					//t_comm += MPI_Wtime() - t;
				}
			}
		}
		else if(status.MPI_TAG == 1){
			send = 1;
			//s_count++;
			if(recv == 0){
				if(rank == 0){
					//t = MPI_Wtime();
					for(i=1; i<task_n; i++){
						MPI_Send(&send, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
					}
					//t_comm += MPI_Wtime() - t;
					break;
				}
				if(color == 1)color = 0;
				else send = 0;
			}
			else if(rank == 0)send = 0;
			//t = MPI_Wtime();
			MPI_Send(&send, 1, MPI_INT, (rank+1)%task_n, 1, MPI_COMM_WORLD);
			//t_comm += MPI_Wtime() - t;
		}
		else break;
	}
	pre_f = vertices[rank].pre;

	//t = MPI_Wtime();
	MPI_Gather(&pre_f, 1, MPI_INT, pre_res, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//t_comm += MPI_Wtime() - t;

	//printf("Send Message: %d @rank%d\n", s_count, rank);
	//printf("Recv Message: %d @rank%d\n", r_count, rank);
	//printf("Comm  Time: %f @rank%d\n", t_comm, rank);
	//printf("Cal   Time: %f @rank%d\n", MPI_Wtime() - t_cal - t_comm, rank);
	//printf("Total Time: %f @rank%d\n", MPI_Wtime() - t_cal, rank);

	//t = MPI_Wtime();
	if(rank == 0){
		char buffer [512];
		char result [512];
		fh=fopen(argv[3],"w");
		for(i=0; i<vertex_n; i++){
			sprintf(result, "%d",i+1);
			int now = pre_res[i];
			if(now==-1)sprintf(result, "%d %d", i+1, i+1);
			while(now!=-1){
				sprintf(buffer, "%d %s", now+1, result);
				strcpy(result, buffer);
				now = pre_res[now];
			}
			fprintf(fh,"%s\n", result);
		}		
		fclose(fh);
		//printf("Output Time: %f\n", MPI_Wtime() - t + t_in);
	}

	//freeing memory
	for (i=0; i<vertex_n; i++) free(vertices[i].adj);
	for (i=0; i<vertex_n; i++) free(vertices[i].w);
	free(vertices);
	free(pre_res);
	free(work_q);
	MPI_Finalize();
	return 0;
}
