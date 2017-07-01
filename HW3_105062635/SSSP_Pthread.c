#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

typedef struct
{
	int adj_n;
	int pre;
	int min_d;
	int *adj;
	int *w;
}vertex;

pthread_mutex_t mutex;
pthread_mutex_t *mutex_v;
vertex *vertices;
int *work_q;
int vertex_n, head, tail, terminate, count, qsize, notused;

double diff(struct timespec start, struct timespec end){
	double temp;
	temp = end.tv_sec - start.tv_sec + (end.tv_nsec-start.tv_nsec)/1000000000.0;
	return temp;
}

void *UpdateD() {
	int current, i, j, cur_pos, next, enqueue, new_d;
	double t_lock = 0.0;
	double t_cal = 0.0;
	struct timespec tl1, tl2, tc1, tc2;
	//clock_gettime(CLOCK_MONOTONIC, &tc1);
	while(terminate != 1){

		//clock_gettime(CLOCK_MONOTONIC, &tl1);
		pthread_mutex_lock(&mutex);
		//clock_gettime(CLOCK_MONOTONIC, &tl2);
		//t_lock += diff(tl1, tl2);
		current = work_q[head];
		cur_pos = head;
		if(current != -1)head = (head+1)%qsize;
		pthread_mutex_unlock(&mutex);
	
		if(current != -1){
			for(i=0; i<vertices[current].adj_n; i++){
				next = vertices[current].adj[i];
				new_d = vertices[current].w[i] + vertices[current].min_d;
				//clock_gettime(CLOCK_MONOTONIC, &tl1);
				pthread_mutex_lock(&mutex_v[next]);
				//clock_gettime(CLOCK_MONOTONIC, &tl2);
				//t_lock += diff(tl1, tl2);
				if(new_d < vertices[next].min_d){
					vertices[next].min_d = new_d;
					vertices[next].pre = current;
					pthread_mutex_unlock(&mutex_v[next]);

					enqueue = 1;
					//clock_gettime(CLOCK_MONOTONIC, &tl1);
					pthread_mutex_lock(&mutex);
					//clock_gettime(CLOCK_MONOTONIC, &tl2);
					//t_lock += diff(tl1, tl2);
					if(head<=tail){
						for(j=head+1; j<tail; j++){
							if(work_q[j] == next)enqueue = 0;
						}
					}
					else{
						for(j=head+1; j<qsize; j++){
							if(work_q[j] == next)enqueue = 0;
						}
						for(j=0; j<tail; j++){
							if(work_q[j] == next)enqueue = 0;
						}
					}
					if(enqueue == 1){	
						work_q[tail] = next;
						tail=(tail+1)%qsize;
					}
					pthread_mutex_unlock(&mutex);		
				}		
				else {
					pthread_mutex_unlock(&mutex_v[next]);
				}
			}
			work_q[cur_pos] = -1;
		}
	}
	//clock_gettime(CLOCK_MONOTONIC, &tc2);
	//t_cal += diff(tc1, tc2) - t_lock;
	//printf("cal time:  %f @%ld\n", t_cal, tc2.tv_nsec);
	//printf("lock time: %f @%ld\n", t_lock, tc2.tv_nsec);
	//if(t_lock==0)notused++;
	pthread_exit(NULL);
}

int main (int argc, char *argv[]) {
	int thread_n, vertex_s, edge_n, i, temp;
	FILE *fh;
	pthread_t *threads;
	double t_io = 0.0;
	double t_all = 0.0;
	struct timespec tio1, tio2, ta1;

	thread_n = atof(argv[1]);
	vertex_s = atof(argv[4])-1;

	//clock_gettime(CLOCK_MONOTONIC, &ta1);
	//clock_gettime(CLOCK_MONOTONIC, &tio1);
	fh=fopen(argv[2],"r");
	fscanf(fh,"%d%d", &vertex_n, &edge_n);

	qsize = thread_n+edge_n;

	//memory allocation
	threads = malloc(thread_n*sizeof(pthread_t));
	mutex_v = malloc(vertex_n*sizeof(pthread_mutex_t));
	vertices = malloc(vertex_n*sizeof(vertex));
	work_q = malloc(qsize*sizeof(int));
	for (i=0; i<vertex_n; i++) vertices[i].adj = malloc(vertex_n*sizeof(int));
	for (i=0; i<vertex_n; i++) vertices[i].w = malloc(vertex_n*sizeof(int));

	//global variables initialization
	for (i=0; i<vertex_n; i++){
		vertices[i].adj_n = 0;
		vertices[i].min_d = INT_MAX;
		pthread_mutex_init(&mutex_v[i], NULL);
	}	

	for(i=0; i<qsize; i++)work_q[i]=-1;
	vertices[vertex_s].min_d = 0;
	vertices[vertex_s].pre = -1;
	work_q[0] = vertex_s;
	head = 0;
	tail = 1;
	terminate = 0;

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
	//clock_gettime(CLOCK_MONOTONIC, &tio2);
	//t_io += diff(tio1, tio2);

	//pthread creation
	pthread_mutex_init (&mutex, NULL);

	for(i=0; i<thread_n; i++){
		pthread_create(&threads[i], NULL, UpdateD, NULL);
	}

	//temination flag
	while(terminate != 1){
		temp = 1;
		pthread_mutex_lock(&mutex);
		for(i=0; i<qsize; i++){
			if(work_q[i]!=-1) temp = 0;
		}
		pthread_mutex_unlock(&mutex);
		terminate = temp;
	}

	char buffer [512];
	char result [512];
	//clock_gettime(CLOCK_MONOTONIC, &tio1);
	fh=fopen(argv[3],"w");
	for(i=0; i<vertex_n; i++){
		sprintf(result, "%d",i+1);
		vertex now = vertices[i];
		if(now.pre==-1)sprintf(result, "%d %d", i+1, i+1);
		while(now.pre!=-1){
			sprintf(buffer, "%d %s", now.pre+1, result);
			strcpy(result, buffer);
			now = vertices[now.pre];
		}
		fprintf(fh,"%s\n", result);
	}		
	fclose(fh);
	//clock_gettime(CLOCK_MONOTONIC, &tio2);
	//t_io += diff(tio1, tio2);
	//t_all+= diff(ta1, tio2);
	//printf("IO time:  %f @%ld\n", t_io, tio2.tv_nsec);
	//printf("total time:  %f @%ld\n", t_all, tio2.tv_nsec);
	//printf("not used: %d", notused);

	pthread_mutex_destroy(&mutex);
	for (i=0; i<vertex_n; i++){	
		pthread_mutex_destroy(&mutex_v[i]);
	}

	//freeing memory
	for (i=0; i<vertex_n; i++) free(vertices[i].adj);
	for (i=0; i<vertex_n; i++) free(vertices[i].w);
	free(vertices);
	free(work_q);
	free(threads);
	pthread_exit(NULL);
}
