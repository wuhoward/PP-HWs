#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int INF = 10000000;
const int NS = 16;
void input(char *inFileName);
void output(char *outFileName);

int n, m;	// Number of vertices, edges
int *Dist;

bool InitCUDA()
{
	int count;
	cudaGetDeviceCount(&count);
	int i;
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			cudaGetDeviceProperties(&prop, i);
			printf("Device Number: %d\n", i);
			printf("  Device name: %s\n", prop.name);
			printf("  Major: %d Overlap: %d\n", prop.major, prop.deviceOverlap);
			printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
			printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
			printf("  Shared Memory Size per Block (bytes): %d\n", prop.sharedMemPerBlock);
			printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		}
	}
	return true;
}

void input(char *inFileName)
{
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);
	cudaMallocHost ((void**) &Dist, n*n*sizeof(int));

	for (int i = 0; i < n*n; ++i) {
		if (i/n == i%n)	Dist[i] = 0;
		else Dist[i] = INF;
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		--a, --b;
		Dist[a*n + b] = v;
	}
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n*n; ++i) {
		if(Dist[i] >= INF) fprintf(outfile, "INF ");
		else fprintf(outfile, "%d ", Dist[i]);
		if(i%n == n-1)fprintf(outfile, "\n");
	}
}

__host__ __device__ int ceil(int a, int b)
{
	return (a + b -1)/b;
}

__global__ void block_FW1_Kernel(int B, int r, int n, int *d_Dist)
{
	extern __shared__ int b_Dist[];
	int xi, yi, x1, x2, x3, x4, y1, y2, y3, y4;
	bool within1, within2, within3, within4;
	xi = threadIdx.x%B;
	yi = threadIdx.x/B;
	bool within = r*B+yi<n && r*B+xi<n;
	if(B<=32){
		if(within)b_Dist[yi*B+xi] = d_Dist[(r*B+yi)*n+r*B+xi];
		__syncthreads();
		for (int k = 0; k < B && r*B+k<n ; ++k) {
			if(within && b_Dist[yi*B+xi] > b_Dist[k*B+xi] + b_Dist[yi*B+k])
				b_Dist[yi*B+xi] = b_Dist[k*B+xi] + b_Dist[yi*B+k];
			__syncthreads();
		}
		if(within)d_Dist[(r*B+yi)*n+r*B+xi] = b_Dist[yi*B+xi];
	}
	else if(B<=64){
		x1 = xi;
		y1 = yi;
		within1 = r*B+y1<n && r*B+x1<n && y1<B;
		y2 = y1+(x1+1024)/B;
		x2 = (x1+1024)%B;
		within2 = r*B+y2<n && r*B+x2<n && y2<B;
		y3 = y2+(x2+1024)/B;
		x3 = (x2+1024)%B;
		within3 = r*B+y3<n && r*B+x3<n && y3<B;
		y4 = y3+(x3+1024)/B;
		x4 = (x3+1024)%B;
		within4 = r*B+y4<n && r*B+x4<n && y4<B;
		if(within1)b_Dist[y1*B+x1] = d_Dist[(r*B+y1)*n+r*B+x1];
		if(within2)b_Dist[y2*B+x2] = d_Dist[(r*B+y2)*n+r*B+x2];
		if(within3)b_Dist[y3*B+x3] = d_Dist[(r*B+y3)*n+r*B+x3];
		if(within4)b_Dist[y4*B+x4] = d_Dist[(r*B+y4)*n+r*B+x4];
		__syncthreads();
		for (int k = 0; k < B && r*B+k<n ; ++k) {
			if(within1 && b_Dist[y1*B+x1] > b_Dist[k*B+x1] + b_Dist[y1*B+k])
				b_Dist[y1*B+x1] = b_Dist[k*B+x1] + b_Dist[y1*B+k];
			if(within2 && b_Dist[y2*B+x2] > b_Dist[k*B+x2] + b_Dist[y2*B+k])
				b_Dist[y2*B+x2] = b_Dist[k*B+x2] + b_Dist[y2*B+k];
			if(within3 && b_Dist[y3*B+x3] > b_Dist[k*B+x3] + b_Dist[y3*B+k])
				b_Dist[y3*B+x3] = b_Dist[k*B+x3] + b_Dist[y3*B+k];
			if(within4 && b_Dist[y4*B+x4] > b_Dist[k*B+x4] + b_Dist[y4*B+k])
				b_Dist[y4*B+x4] = b_Dist[k*B+x4] + b_Dist[y4*B+k];
			__syncthreads();
		}
		if(within1)d_Dist[(r*B+y1)*n+r*B+x1] = b_Dist[y1*B+x1];
		if(within2)d_Dist[(r*B+y2)*n+r*B+x2] = b_Dist[y2*B+x2];
		if(within3)d_Dist[(r*B+y3)*n+r*B+x3] = b_Dist[y3*B+x3];
		if(within4)d_Dist[(r*B+y4)*n+r*B+x4] = b_Dist[y4*B+x4];
	}
	else{
		int iter = ceil(B*B, 1024);
		for (int k = 0; k < B && r*B+k<n; ++k) {
			xi = threadIdx.x%B;
			yi = threadIdx.x/B;
			for(int i=0; i<iter; i++){
				within = r*B+yi<n && r*B+xi<n && yi<B;
				if(within && d_Dist[(r*B+yi)*n+r*B+xi] > d_Dist[(r*B+k)*n+r*B+xi] + d_Dist[(r*B+yi)*n+r*B+k])
					d_Dist[(r*B+yi)*n+r*B+xi] = d_Dist[(r*B+k)*n+r*B+xi] + d_Dist[(r*B+yi)*n+r*B+k];
				yi = yi+(xi+1024)/B;
				xi = (xi+1024)%B;
			}
			__syncthreads();
		}
	}
}

__global__ void block_FW2_Kernel(int B, int r, int n, int *d_Dist)
{
	extern __shared__ int b_Dist[];
	int xi, yi, xb, yb, x1, x2, x3, x4, y1, y2, y3, y4;
	bool within1, within2, within3, within4;
	int width = ceil(n, B);
	xb = blockIdx.x < width ? blockIdx.x : r;
	yb = blockIdx.x < width ? r : blockIdx.x - width;
	xi = threadIdx.x%B;
	yi = threadIdx.x/B;
	bool within = yb*B+yi<n && xb*B+xi<n && !(xb==r && yb==r);  
	if(B<=32){
		if(within)b_Dist[yi*B+xi] = d_Dist[(yb*B+yi)*n+xb*B+xi];
		if(r*B+yi<n && r*B+xi<n)b_Dist[B*B+yi*B+xi] = d_Dist[(r*B+yi)*n+r*B+xi];
		__syncthreads();
		for (int k = 0; k < B && r*B+k<n; ++k) {
			if(blockIdx.x>=width && within && b_Dist[yi*B+xi] > b_Dist[B*B+k*B+xi] + b_Dist[yi*B+k])
				b_Dist[yi*B+xi] = b_Dist[B*B+k*B+xi] + b_Dist[yi*B+k];
			else if(blockIdx.x<width && within && b_Dist[yi*B+xi] > b_Dist[B*B+yi*B+k] + b_Dist[k*B+xi])
				b_Dist[yi*B+xi] = b_Dist[B*B+yi*B+k] + b_Dist[k*B+xi];
			__syncthreads();
		}
		if(within)d_Dist[(yb*B+yi)*n+xb*B+xi] = b_Dist[yi*B+xi];
	}
	else if(B<=64){
		x1 = xi;
		y1 = yi;
		within1 = yb*B+y1<n && xb*B+x1<n && !(xb==r && yb==r) && y1<B;  
		y2 = y1+(x1+1024)/B;
		x2 = (x1+1024)%B;
		within2 = yb*B+y2<n && xb*B+x2<n && !(xb==r && yb==r) && y2<B;  
		y3 = y2+(x2+1024)/B;
		x3 = (x2+1024)%B;
		within3 = yb*B+y3<n && xb*B+x3<n && !(xb==r && yb==r) && y3<B;  
		y4 = y3+(x3+1024)/B;
		x4 = (x3+1024)%B;
		within4 = yb*B+y4<n && xb*B+x4<n && !(xb==r && yb==r) && y4<B;  
		if(within1)b_Dist[y1*B+x1] = d_Dist[(yb*B+y1)*n+xb*B+x1];
		if(r*B+y1<n && r*B+x1<n && y1<B)b_Dist[B*B+y1*B+x1] = d_Dist[(r*B+y1)*n+r*B+x1];
		if(within2)b_Dist[y2*B+x2] = d_Dist[(yb*B+y2)*n+xb*B+x2];
		if(r*B+y2<n && r*B+x2<n && y2<B)b_Dist[B*B+y2*B+x2] = d_Dist[(r*B+y2)*n+r*B+x2];
		if(within3)b_Dist[y3*B+x3] = d_Dist[(yb*B+y3)*n+xb*B+x3];
		if(r*B+y3<n && r*B+x3<n && y3<B)b_Dist[B*B+y3*B+x3] = d_Dist[(r*B+y3)*n+r*B+x3];
		if(within4)b_Dist[y4*B+x4] = d_Dist[(yb*B+y4)*n+xb*B+x4];
		if(r*B+y4<n && r*B+x4<n && y4<B)b_Dist[B*B+y4*B+x4] = d_Dist[(r*B+y4)*n+r*B+x4];
		__syncthreads();
		for (int k = 0; k < B && r*B+k<n; ++k) {
			if(blockIdx.x>=width){
				if(within1 && b_Dist[y1*B+x1] > b_Dist[B*B+k*B+x1] + b_Dist[y1*B+k])
					b_Dist[y1*B+x1] = b_Dist[B*B+k*B+x1] + b_Dist[y1*B+k];
				if(within2 && b_Dist[y2*B+x2] > b_Dist[B*B+k*B+x2] + b_Dist[y2*B+k])
					b_Dist[y2*B+x2] = b_Dist[B*B+k*B+x2] + b_Dist[y2*B+k];
				if(within3 && b_Dist[y3*B+x3] > b_Dist[B*B+k*B+x3] + b_Dist[y3*B+k])
					b_Dist[y3*B+x3] = b_Dist[B*B+k*B+x3] + b_Dist[y3*B+k];
				if(within4 && b_Dist[y4*B+x4] > b_Dist[B*B+k*B+x4] + b_Dist[y4*B+k])
					b_Dist[y4*B+x4] = b_Dist[B*B+k*B+x4] + b_Dist[y4*B+k];
			}
			else {
				if(within1 && b_Dist[y1*B+x1] > b_Dist[B*B+y1*B+k] + b_Dist[k*B+x1])
					b_Dist[y1*B+x1] = b_Dist[B*B+y1*B+k] + b_Dist[k*B+x1];
				if(within2 && b_Dist[y2*B+x2] > b_Dist[B*B+y2*B+k] + b_Dist[k*B+x2])
					b_Dist[y2*B+x2] = b_Dist[B*B+y2*B+k] + b_Dist[k*B+x2];
				if(within3 && b_Dist[y3*B+x3] > b_Dist[B*B+y3*B+k] + b_Dist[k*B+x3])
					b_Dist[y3*B+x3] = b_Dist[B*B+y3*B+k] + b_Dist[k*B+x3];
				if(within4 && b_Dist[y4*B+x4] > b_Dist[B*B+y4*B+k] + b_Dist[k*B+x4])
					b_Dist[y4*B+x4] = b_Dist[B*B+y4*B+k] + b_Dist[k*B+x4];
			}
			__syncthreads();
		}
		if(within1)d_Dist[(yb*B+y1)*n+xb*B+x1] = b_Dist[y1*B+x1];
		if(within2)d_Dist[(yb*B+y2)*n+xb*B+x2] = b_Dist[y2*B+x2];
		if(within3)d_Dist[(yb*B+y3)*n+xb*B+x3] = b_Dist[y3*B+x3];
		if(within4)d_Dist[(yb*B+y4)*n+xb*B+x4] = b_Dist[y4*B+x4];
	}
	else{
		int iter = ceil(B*B, 1024);
		for (int k = 0; k < B && r*B+k<n; ++k) {
			xi = threadIdx.x%B;
			yi = threadIdx.x/B;
			for(int i=0; i<iter; i++){
				within = yb*B+yi<n && xb*B+xi<n && !(xb==r && yb==r) && yi<B;  
				if(blockIdx.x>=width && within && d_Dist[(yb*B+yi)*n+xb*B+xi] > d_Dist[(r*B+k)*n+r*B+xi] + d_Dist[(yb*B+yi)*n+xb*B+k])
					d_Dist[(yb*B+yi)*n+xb*B+xi] = d_Dist[(r*B+k)*n+r*B+xi] + d_Dist[(yb*B+yi)*n+xb*B+k];
				else if(blockIdx.x<width && within && d_Dist[(yb*B+yi)*n+xb*B+xi] > d_Dist[(r*B+yi)*n+r*B+k] + d_Dist[(yb*B+k)*n+xb*B+xi])
					d_Dist[(yb*B+yi)*n+xb*B+xi] = d_Dist[(r*B+yi)*n+r*B+k] + d_Dist[(yb*B+k)*n+xb*B+xi];
				yi = yi+(xi+1024)/B;
				xi = (xi+1024)%B;
			}
			__syncthreads();
		}
	}
}

__global__ void block_FW3_Kernel(int B, int r, int n, int boff, int *d_Dist)
{
	extern __shared__ int b_Dist[];
	int xi, yi, xb, yb, x1, x2, x3, x4, y1, y2, y3, y4;
	bool within1, within2, within3, within4;
	xb = blockIdx.x;
	yb = blockIdx.y + boff;
	xi = threadIdx.x%B;
	yi = threadIdx.x/B;
	bool within = yb*B+yi<n && xb*B+xi<n && xb!=r && yb !=r; 
	if(B<=32){
		if(within)b_Dist[yi*B+xi] = d_Dist[(yb*B+yi)*n+xb*B+xi];
		if(r*B+yi<n && xb*B+xi<n)b_Dist[B*B+yi*B+xi] = d_Dist[(r*B+yi)*n+xb*B+xi];
		if(yb*B+yi<n && r*B+xi<n)b_Dist[2*B*B+yi*B+xi] = d_Dist[(yb*B+yi)*n+r*B+xi];
		__syncthreads();
		for (int k = 0; k < B && r*B+k<n; ++k) {
			if(within && b_Dist[yi*B+xi] > b_Dist[B*B+k*B+xi] + b_Dist[2*B*B+yi*B+k])
				b_Dist[yi*B+xi] = b_Dist[B*B+k*B+xi] + b_Dist[2*B*B+yi*B+k];
			__syncthreads();
		}
		if(within)d_Dist[(yb*B+yi)*n+xb*B+xi] = b_Dist[yi*B+xi];
	}
	else if(B<=64){
		x1 = xi;
		y1 = yi;
		within1 = yb*B+y1<n && xb*B+x1<n && xb!=r && yb!=r && y1<B; 
		y2 = y1+(x1+1024)/B;
		x2 = (x1+1024)%B;
		within2 = yb*B+y2<n && xb*B+x2<n && xb!=r && yb!=r && y2<B; 
		y3 = y2+(x2+1024)/B;
		x3 = (x2+1024)%B;
		within3 = yb*B+y3<n && xb*B+x3<n && xb!=r && yb!=r && y3<B; 
		y4 = y3+(x3+1024)/B;
		x4 = (x3+1024)%B;
		within4 = yb*B+y4<n && xb*B+x4<n && xb!=r && yb!=r && y4<B; 
		if(within1)b_Dist[y1*B+x1] = d_Dist[(yb*B+y1)*n+xb*B+x1];
		if(r*B+y1<n && xb*B+x1<n && y1<B)b_Dist[B*B+y1*B+x1] = d_Dist[(r*B+y1)*n+xb*B+x1];
		if(yb*B+y1<n && r*B+x1<n && y1<B)b_Dist[2*B*B+y1*B+x1] = d_Dist[(yb*B+y1)*n+r*B+x1];
		if(within2)b_Dist[y2*B+x2] = d_Dist[(yb*B+y2)*n+xb*B+x2];
		if(r*B+y2<n && xb*B+x2<n && y2<B)b_Dist[B*B+y2*B+x2] = d_Dist[(r*B+y2)*n+xb*B+x2];
		if(yb*B+y2<n && r*B+x2<n && y2<B)b_Dist[2*B*B+y2*B+x2] = d_Dist[(yb*B+y2)*n+r*B+x2];
		if(within3)b_Dist[y3*B+x3] = d_Dist[(yb*B+y3)*n+xb*B+x3];
		if(r*B+y3<n && xb*B+x3<n && y3<B)b_Dist[B*B+y3*B+x3] = d_Dist[(r*B+y3)*n+xb*B+x3];
		if(yb*B+y3<n && r*B+x3<n && y3<B)b_Dist[2*B*B+y3*B+x3] = d_Dist[(yb*B+y3)*n+r*B+x3];
		if(within4)b_Dist[y4*B+x4] = d_Dist[(yb*B+y4)*n+xb*B+x4];
		if(r*B+y4<n && xb*B+x4<n && y4<B)b_Dist[B*B+y4*B+x4] = d_Dist[(r*B+y4)*n+xb*B+x4];
		if(yb*B+y4<n && r*B+x4<n && y4<B)b_Dist[2*B*B+y4*B+x4] = d_Dist[(yb*B+y4)*n+r*B+x4];
		__syncthreads();
		for (int k = 0; k < B && r*B+k<n; ++k) {
			if(within1 && b_Dist[y1*B+x1] > b_Dist[B*B+k*B+x1] + b_Dist[2*B*B+y1*B+k])
				b_Dist[y1*B+x1] = b_Dist[B*B+k*B+x1] + b_Dist[2*B*B+y1*B+k];
			if(within2 && b_Dist[y2*B+x2] > b_Dist[B*B+k*B+x2] + b_Dist[2*B*B+y2*B+k])
				b_Dist[y2*B+x2] = b_Dist[B*B+k*B+x2] + b_Dist[2*B*B+y2*B+k];
			if(within3 && b_Dist[y3*B+x3] > b_Dist[B*B+k*B+x3] + b_Dist[2*B*B+y3*B+k])
				b_Dist[y3*B+x3] = b_Dist[B*B+k*B+x3] + b_Dist[2*B*B+y3*B+k];
			if(within4 && b_Dist[y4*B+x4] > b_Dist[B*B+k*B+x4] + b_Dist[2*B*B+y4*B+k])
				b_Dist[y4*B+x4] = b_Dist[B*B+k*B+x4] + b_Dist[2*B*B+y4*B+k];
			__syncthreads();
		}
		if(within1)d_Dist[(yb*B+y1)*n+xb*B+x1] = b_Dist[y1*B+x1];
		if(within2)d_Dist[(yb*B+y2)*n+xb*B+x2] = b_Dist[y2*B+x2];
		if(within3)d_Dist[(yb*B+y3)*n+xb*B+x3] = b_Dist[y3*B+x3];
		if(within4)d_Dist[(yb*B+y4)*n+xb*B+x4] = b_Dist[y4*B+x4];
	}
	else{
		int iter = ceil(B*B, 1024);
		for (int k = 0; k < B && r*B+k<n; ++k) {
			xi = threadIdx.x%B;
			yi = threadIdx.x/B;
			for(int i=0; i<iter; i++){
				within = yb*B+yi<n && xb*B+xi<n && xb!=r && yb !=r && yi<B; 
				if(within && d_Dist[(yb*B+yi)*n+xb*B+xi] > d_Dist[(r*B+k)*n+xb*B+xi] + d_Dist[(yb*B+yi)*n+r*B+k])
					d_Dist[(yb*B+yi)*n+xb*B+xi] = d_Dist[(r*B+k)*n+xb*B+xi] + d_Dist[(yb*B+yi)*n+r*B+k];
					yi = yi+(xi+1024)/B;
					xi = (xi+1024)%B;
			}
			 __syncthreads();
		}
	}
}

int main(int argc, char* argv[])
{
	//InitCUDA();
	//cudaEvent_t start, stop;
	//float time;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	input(argv[1]);
	int B = atoi(argv[3]) > n ? n : atoi(argv[3]);
	int NT = B > 32 ? 32*32 : B*B;
	int NB = B > 64 ? 64*64 : B*B;
	int *d_Dist;
	cudaStream_t streams[NS];
	for(int i=0; i<NS; i++)cudaStreamCreate(&streams[i]);
	cudaMalloc(&d_Dist, n*n*sizeof(int));
	cudaMemcpy(d_Dist, Dist, n*n*sizeof(int), cudaMemcpyHostToDevice);
	int round = ceil(n, B);
	int bps = round == 1 ? 1 : ceil(round-1, NS-1); //# blocks per stream
	int NSused = ceil(round-1, bps); //# streams really needed
	//printf("round:%d bps:%d used:%d\n",round,bps,NSused);
	dim3 block(round, round);
	dim3 block2(round, bps); 
	//cudaEventRecord(start);
	for (int r = 0; r < round; ++r) {	
		block_FW1_Kernel<<<1, NT, NB*sizeof(int)>>>(B, r, n, d_Dist);
		block_FW2_Kernel<<<2*round, NT, 2*NB*sizeof(int)>>>(B, r, n, d_Dist);
		if(r == round-1){
			cudaMemcpyAsync(Dist+r*B*n, d_Dist+r*B*n, (n-r*B)*n*sizeof(int), cudaMemcpyDeviceToHost, streams[NSused]); 
			for(int i=0; i<NSused-1; i++){
				block_FW3_Kernel<<<block2, NT, 3*NB*sizeof(int), streams[i]>>>(B, r, n, i*bps, d_Dist);
				cudaMemcpyAsync(Dist+n*bps*B*i, d_Dist+n*bps*B*i, n*bps*B*sizeof(int), cudaMemcpyDeviceToHost, streams[i]); 
			}
			block_FW3_Kernel<<<block2, NT, 3*NB*sizeof(int), streams[NSused-1]>>>(B, r, n, (NSused-1)*bps, d_Dist);
			int remain = (round-1)%bps == 0 ? bps : (round-1)%bps;
			cudaMemcpyAsync(Dist+n*bps*B*(NSused-1), d_Dist+n*bps*B*(NSused-1), n*(remain*B)*sizeof(int), cudaMemcpyDeviceToHost, streams[NSused-1]); 
		}
		else block_FW3_Kernel<<<block, NT, 3*NB*sizeof(int)>>>(B, r, n, 0, d_Dist);
	}	
	for(int i=0; i<NS; i++)cudaStreamSynchronize(streams[i]);
	//cudaMemcpy(Dist, d_Dist, n*n*sizeof(int), cudaMemcpyDeviceToHost); 
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	output(argv[2]);
	for(int i=0; i<NS; i++)cudaStreamDestroy(streams[i]);
	//cudaEventElapsedTime(&time, start, stop);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	cudaFree(d_Dist);
	cudaFreeHost(Dist);
	//printf("time elapsed for calculation: %lf!!!\n", time);
	return 0;
}

