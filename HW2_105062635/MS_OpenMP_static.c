#include <X11/Xlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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

	int thread_n, width, height, window_en, grid_size=10, grid_width, grid_height, gx, gy;
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
	grid_width = (width % grid_size) ? width/grid_size + 1 : width/grid_size;
	grid_height = (height % grid_size) ? height/grid_size + 1 : height/grid_size;

	int i, j;
	int *count = malloc(thread_n * sizeof(int) );
    for (i=0; i<thread_n; i++) count[i] = 0;
	int **color_buf = (int **)malloc(width * sizeof(int *));
    for (i=0; i<width; i++) color_buf[i] = (int *)malloc(height * sizeof(int));

	/* open connection with the server */ 
	if(window_en == 1){
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
	if(window_en == 1){
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
	int repeats, tid;
	double temp, lengthsq;
	//int i, j;
	t = omp_get_wtime();
	#pragma omp parallel num_threads(thread_n) private(z, c, repeats, temp, lengthsq, i ,j, tid)
	{
	#pragma omp for collapse(2) schedule(static, 30) 
	for(i = 0; i < width; i++) {
		for(j = 0; j < height; j++) {
			tid = omp_get_thread_num();
			count[tid]++;
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
			color_buf[i][j]=repeats;
		}
	}
	printf("t=%f count:%d rank:%d \n", omp_get_wtime() - t, count[tid], tid);
	}
	if(window_en == 1){
		for(i = 0; i < width; i++) {
			for(j = 0; j < height; j++) {
				XSetForeground (display, gc,  1024 * 1024 * (color_buf[i][j] % 256));		
				XDrawPoint (display, window, gc, i, j);	
			}
		}
		XFlush(display);
		sleep(5);
	}
	free(count);
	free(color_buf);
	return 0;
}

