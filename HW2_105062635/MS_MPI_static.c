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
	Window window;      //initialization for a window
	int screen;         //which screen 
	GC gc;
	
	if(argc != 9) {
		printf("input error!");
		return 0;
	}

	int thread_n, width, height, window_en, grid_size=10, grid_width, grid_height, gx, gy, count=0;
	double sx, bx, sy, by, t, t1, count_t=0.0;
	thread_n = atoi(argv[1]);
	//{little, big} x {x, y}
	sx = atof(argv[2]);
	bx = atof(argv[3]);
	sy = atof(argv[4]);
	by = atof(argv[5]);
	width = atoi(argv[6]);
	height = atoi(argv[7]); 
	window_en = 1-strcmp(argv[8],"enable");
	grid_width = (width % grid_size) ? width/grid_size + 1 : width/grid_size;
	grid_height = (height % grid_size) ? height/grid_size + 1 : height/grid_size;

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
	t = MPI_Wtime();
	for(i = 0; i < width; i++) {
		for(j = 0; j < height; j++) {
			gx = i/grid_size;
			gy = j/grid_size;
			if((gx*grid_width+gy) % task_n == rank){
				t1 = MPI_Wtime();
				count++;
				z.real = 0.0;
				z.imag = 0.0;
				c.real = sx + i*(bx-sx)/(double)width;
				c.imag = sy + j*(by-sy)/(double)height;
				repeats = 0;
				lengthsq = 0.0;
				while(repeats < 100000 && lengthsq < 4.0) { 
					temp = z.real*z.real - z.imag*z.imag + c.real;
					z.imag = 2*z.real*z.imag + c.imag;
					z.real = temp;
					lengthsq = z.real*z.real + z.imag*z.imag; 
					repeats++;
				}
				count_t+=MPI_Wtime()-t1;
				if(rank != 0){
					int send_buf[3];
					send_buf[0] = i;
					send_buf[1] = j;
					send_buf[2] = repeats;
					MPI_Send(&send_buf, 3, MPI_INT, 0, 0, MPI_COMM_WORLD);
				}
				else if(window_en == 1){
					XSetForeground (display, gc,  1024 * 1024 * (repeats % 256));		
					XDrawPoint (display, window, gc, i, j);	
				}
			}
			else if(rank == 0){
				int recv_buf[3];
				MPI_Recv(&recv_buf, 3, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
				if(window_en == 1){
					XSetForeground (display, gc,  1024 * 1024 * (recv_buf[2] % 256));		
					XDrawPoint (display, window, gc, recv_buf[0], recv_buf[1]);
				}
			}
		}
	}
	printf("total time:%f cal time:%f count:%d rank:%d\n", MPI_Wtime() - t,count_t, count, rank);
	if(rank == 0 && window_en == 1){
		XFlush(display);
		sleep(5);
	}
	MPI_Finalize();
	return 0;
}

