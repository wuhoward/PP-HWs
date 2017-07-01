#include <X11/Xlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

typedef struct complextype
{
	double real, imag;
} Compl;

int main(int argc, char *argv[] )
{
	Display *display;
	Window window;      
	int screen;          
	GC gc;
	
	if(argc != 9) {
		printf("input error!");
		return 0;
	}

	int thread_n, width, height, window_en, terminate, finish, row, provided, i, j, first;
	int *row_buf, *done_buf;
	double sx, bx, sy, by, t;
	thread_n = atoi(argv[1]);
	//{little, big} x {x, y}
	sx = atof(argv[2]);
	bx = atof(argv[3]);
	sy = atof(argv[4]);
	by = atof(argv[5]);
	width = atoi(argv[6]);
	height = atoi(argv[7]); 
	window_en = 1-strcmp(argv[8],"enable");
	
	int task_n, rank;
	int *count = malloc(thread_n * sizeof(int) );
    for (i=0; i<thread_n; i++) count[i] = 0;
	int **color_buf = (int **)malloc(width * sizeof(int *));
    for (i=0; i<width; i++) color_buf[i] = (int *)malloc(height * sizeof(int));
	int **draw_buf = (int **)malloc(width * sizeof(int *));
    for (i=0; i<width; i++) draw_buf[i] = (int *)malloc(height * sizeof(int));

	MPI_Status status;
	MPI_Init(&argc, &argv);//, MPI_THREAD_MULTIPLE, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &task_n);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	row_buf=malloc((width+1)*sizeof(int));
	done_buf=malloc(task_n*sizeof(int));
	//MPI_Win win;
	//MPI_Alloc_mem(task_n * sizeof(int), MPI_INFO_NULL, &done_buf);
	//MPI_Win_allocate(task_n*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &done_buf, &win);
	
	if(rank == 0 && window_en == 1){
		display = XOpenDisplay(NULL);
		if(display == NULL) {
			fprintf(stderr, "cannot open display\n");
			return 0;
		}
		screen = DefaultScreen(display);
	}

	int x = 0;
	int y = 0;
	int border_width = 0;

	if(rank == 0 && window_en == 1){
		window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, 
			width, height, border_width,BlackPixel(display, screen), WhitePixel(display, screen));
		XSelectInput ( display, window, ExposureMask | ResizeRedirectMask);
		XGCValues values;
		long valuemask = 0;
	
		gc = XCreateGC(display, window, valuemask, &values);
		XSetForeground (display, gc, BlackPixel (display, screen));
		XSetBackground(display, gc, 0X0000FF00);
		XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
	
		XMapWindow(display, window);
		XSync(display, 0);
	}

	/* draw points */
	Compl z, c;
	int repeats, send_count, tid;
	double temp, lengthsq;
	first = 1;
	finish = 0;
	terminate = 0;
	row = task_n*((height/2)/task_n);
	t = MPI_Wtime();
	//if(rank==0)MPI_Win_create(done_buf, task_n*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    //else MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
	while(terminate == 0){
		if(rank == 0){
			//send_count = 0;
			if(first == 1){
				#pragma omp parallel num_threads(thread_n) private(z, c, i, j, tid) 
				#pragma omp for collapse(2) schedule(guided, 8)
				for(i = 0; i< width; i++){
					for(j = 0; j < height/2 ; j++){
						if(rank == j%task_n){
							tid = omp_get_thread_num();
							count[tid]++;
							c.real = sx + i*(bx-sx)/(double)width;
							c.imag = sy + j*(by-sy)/(double)height;
							color_buf[i][j] = mandelbrotCal(z, c);
						}
					}
				}
				first = 0;
			}
			else{
				MPI_Recv(&finish, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				if(row < height) {
					MPI_Send(&row, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
					row++;
				}
				MPI_Recv(&finish, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				if(row < height) {
					MPI_Send(&row, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
					row++;
				}
				MPI_Recv(&finish, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				if(row < height) {
					MPI_Send(&row, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
					row++;
				}
				if(row < height){
					//calculation on root node
					#pragma omp parallel num_threads(thread_n) private(z, c, i, tid) 
					#pragma omp for schedule(guided, 8)
					for(i = 0; i < width; i++){
						tid = omp_get_thread_num();
						count[tid]++;
						c.real = sx + i*(bx-sx)/(double)width;
						c.imag = sy + row*(by-sy)/(double)height;
						color_buf[i][row] = mandelbrotCal(z, c);
					}
					row++;
				}
			}
			if(row >= height){
				//send terminate signal	
				for(i = 1; i < task_n; i++){
					MPI_Send(&row, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
				}
				terminate=1;
			}
		}
		else {
			//allocate static parts
			//MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
			if(first == 1){
				#pragma omp parallel num_threads(thread_n) private(z, c, i, j) 
				#pragma omp for collapse(2) schedule(guided, 8)
				for(i = 0; i< width; i++){
					for(j = 0; j< height/2; j++){
						if(rank == j%task_n){
							c.real = sx + i*(bx-sx)/(double)width;
							c.imag = sy + j*(by-sy)/(double)height;
							color_buf[i][j] = mandelbrotCal(z, c);
						}
					}
				}
				first = 0;
				finish = 1;
				MPI_Send(&finish, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				//MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
				//MPI_Put(&finish, 1, MPI_INT, 0, rank, 1, MPI_INT, win); 
				//MPI_Win_unlock(0, win);
			}
			else{
				MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				finish = 0;
				//MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
				//MPI_Put(&finish, 1, MPI_INT, 0, rank, 1, MPI_INT, win); 
				//MPI_Win_unlock(0, win);
				if(status.MPI_TAG == 1){
					terminate = 1;
				}
				else{
					#pragma omp parallel num_threads(thread_n) private(z, c, i) 
					#pragma omp for schedule(guided, 8)
					for(i = 0; i < width; i++){
						c.real = sx + i*(bx-sx)/(double)width;
						c.imag = sy + row*(by-sy)/(double)height;
						color_buf[i][row] = mandelbrotCal(z, c);
					}
					finish = 1;
					MPI_Send(&finish, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
					//MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
					//MPI_Put(&finish, 1, MPI_INT, 0, rank, 1, MPI_INT, win); 
					//MPI_Win_unlock(0, win);
				}
			}
		}	
	}
	for(i=0; i<width; i++){
		MPI_Reduce(color_buf[i], draw_buf[i], height, MPI_INT, MPI_SUM , 0, MPI_COMM_WORLD);
	}
	printf("t: %f rank:%d\n", MPI_Wtime() - t, rank);
	if(rank == 0 && window_en == 1){
		for(i = 0; i < width; i++) {
			for(j = 0; j < height; j++) {
				XSetForeground (display, gc,  1024 * 1024 * (draw_buf[i][j] % 256));		
				XDrawPoint (display, window, gc, i, j);	
			}
		}
		XFlush(display);
		sleep(5);
	}
	//MPI_Win_free(&win);
	//MPI_Free_mem(done_buf);
	MPI_Finalize();
	free(row_buf);
	free(done_buf);
    for (i=0; i<width; i++) free(color_buf[i]);
    for (i=0; i<width; i++) free(draw_buf[i]);
	free(color_buf);
	free(draw_buf);
	return 0;
}

int mandelbrotCal(Compl z, Compl c){
	int repeats = 0;
	double temp, lengthsq = 0.0;
	z.real = 0.0;
	z.imag = 0.0;
	if(c.real*c.real + c.imag*c.imag <= 0.0625)repeats=100000;
	while(repeats < 100000 && lengthsq < 4.0) { 
		temp = z.real*z.real - z.imag*z.imag + c.real;
		z.imag = 2*z.real*z.imag + c.imag;
		z.real = temp;
		lengthsq = z.real*z.real + z.imag*z.imag; 
		repeats++;
	}
	return repeats;
}


