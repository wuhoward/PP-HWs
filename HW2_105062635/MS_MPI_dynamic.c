#include <X11/Xlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

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

	int thread_n, width, height, window_en, done, terminate, finish, row, send_count, count=0;
	int *buf_row, *buf_done;
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
	
	buf_row=malloc((width+1)*sizeof(int));
	buf_done=malloc(width*sizeof(int));

	int task_n, rank;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &task_n);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	/* open connection with the server */ 
	if(rank == 0 && window_en == 1){
		display = XOpenDisplay(NULL);
		if(display == NULL) {
			fprintf(stderr, "cannot open display\n");
			return 0;
		}
		screen = DefaultScreen(display);
	}

	/* set window position */
	int x = 0;
	int y = 0;

	/* border width in pixels */
	int border_width = 0;

	/* create window */
	if(rank == 0 && window_en == 1){
		window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, 
			width, height, border_width,BlackPixel(display, screen), WhitePixel(display, screen));
		/* create graph */
		XSelectInput ( display, window, ExposureMask | ResizeRedirectMask);
		XGCValues values;
		long valuemask = 0;
	
		gc = XCreateGC(display, window, valuemask, &values);
		XSetForeground (display, gc, BlackPixel (display, screen));
		XSetBackground(display, gc, 0X0000FF00);
		XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
	
		/* map(show) the window */
		XMapWindow(display, window);
		XSync(display, 0);
	}

	/* draw points */
	Compl z, c;
	int repeats;
	double temp, lengthsq;
	int i, j;
 
	terminate = 0;
	row = 0;
	t = MPI_Wtime();
	if(rank == 0){
		//send initializing tasks
		for(i = 1; i < task_n; i++){	
			MPI_Send(&row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			row++;
		}
	}
	while(terminate == 0){
		send_count = 0;
		if(rank == 0){
			MPI_Recv(buf_row, width+1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			if(window_en == 1){
				for(j = 0; j < width; j++){
					XSetForeground (display, gc,  1024 * 1024 * (buf_row[j] % 256));		
					XDrawPoint (display, window, gc, j, buf_row[width]);
				}
			}
			//if(row < height){
				MPI_Send(&row, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
				send_count++;
				row++;
			//}
			if(row < height){
				//calculation on root node
				for(i = 0; i < width; i++){
					count++;
					z.real = 0.0;
					z.imag = 0.0;
					c.real = sx + i*(bx-sx)/(double)width;
					c.imag = sy + row*(by-sy)/(double)height;
					repeats = 0;
					lengthsq = 0.0;
					while(repeats < 100000 && lengthsq < 4.0) { 
						temp = z.real*z.real - z.imag*z.imag + c.real;
						z.imag = 2*z.real*z.imag + c.imag;
						z.real = temp;
						lengthsq = z.real*z.real + z.imag*z.imag; 
						repeats++;
					}
					if(window_en == 1){
						XSetForeground (display, gc,  1024 * 1024 * (repeats % 256));		
				    	XDrawPoint (display, window, gc, i, row);
					}
				}
				row++;
			}
			if(row >= height){
				//send rest of the tasks
				for(i = 1;i<task_n;i++){
					MPI_Recv(buf_row, width+1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
					if(window_en == 1){
						for(j = 0; j < width; j++){
							XSetForeground (display, gc,  1024 * 1024 * (buf_row[j] % 256));		
							XDrawPoint (display, window, gc, j, buf_row[width]);
						}
					}
				}
				//send terminate signal	
				for(i = 1; i < task_n; i++){
					MPI_Send(&row, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
				}
				terminate=1;
			}
		}
		else {
			MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			finish = 0;
			if(status.MPI_TAG == 1){
				terminate = 1;
			}
			else{
				for(i = 0; i < width; i++){
					count++;
					z.real = 0.0;
					z.imag = 0.0;
					c.real = sx + i*(bx-sx)/(double)width;
					c.imag = sy + row*(by-sy)/(double)height;
					repeats = 0;
					lengthsq = 0.0;
					while(repeats < 100000 && lengthsq < 4.0) { 
						temp = z.real*z.real - z.imag*z.imag + c.real;
						z.imag = 2*z.real*z.imag + c.imag;
						z.real = temp;
						lengthsq = z.real*z.real + z.imag*z.imag; 
						repeats++;
					}
					buf_row[i]=repeats;
				}
				buf_row[width]=row;
				MPI_Send(buf_row, width+1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				finish = 1;
			}
		}	
	}

	printf("t:%f count:%d rank:%d \n", MPI_Wtime() - t, count, rank);
	if(rank == 0 && window_en == 1){
		XFlush(display);
		sleep(5);
	}
	MPI_Finalize();
	free(buf_row);
	free(buf_done);
	return 0;
}

