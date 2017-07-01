#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int epn(int, int);
void swap(int*);
int compare(const void*, const void*);
int merge(int*,int*,int,int*,int);

int main(int argc, char *argv[]){
	int size_i, size_norm, size_each, tasks_num, tasks_used, rank, i, finish, offset, phase, tmp, ratio;
	int *buf_io, *buf_commu, *buf_merge, *buf_head, *buf_tail;
	double t1,t2,t_cal1, t_cal2, t_main, t_commu;
	MPI_File fh;
	MPI_Status status;
	
	size_i=atoi(argv[1]);
	t_cal2=0;
	ratio=1;

	MPI_Init(&argc, &argv);
	t1=MPI_Wtime();

	MPI_Comm_size(MPI_COMM_WORLD, &tasks_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	size_norm=epn(size_i,tasks_num);
	if(size_i<tasks_num*2){
		size_norm=2;
		for(i=0;i<tasks_num;i++){
			size_each=0;
			if(rank<size_i/2)size_each=2;
			if(size_i%2==1&&rank==size_i/2)size_each=1;
		}
		if(rank==0)tasks_used=size_i%2==1?size_i/2+1:size_i/2;
	}
	else{
		size_each=size_norm;
		if(size_i/size_norm==tasks_num){
			if(size_i%size_norm>0&&rank==size_i/size_norm-1)size_each=size_norm+size_i%size_norm;
			tasks_used=tasks_num;	
		}
		else{
			if(size_i%size_norm>0&&rank==size_i/size_norm)size_each=size_i%size_norm;
			else if (rank>=size_i/size_norm)size_each=0;
			if(rank==0)tasks_used=size_i%size_norm>0?size_i/size_each+1:size_i/size_each;
		}
	}
	t2=MPI_Wtime();
	t_cal1=t2-t1;
	t1=MPI_Wtime();
	MPI_Bcast(&tasks_used, 1, MPI_INT, 0, MPI_COMM_WORLD);

	buf_io=malloc((size_each)*sizeof(int));
	buf_commu=malloc(tasks_num*sizeof(int));
	buf_head=malloc(size_norm/ratio*sizeof(int));
	buf_tail=malloc(size_norm/ratio*sizeof(int));
	buf_merge=malloc((size_each+size_norm/ratio)*sizeof(int));

	MPI_Gather(&size_each,1,MPI_INT,buf_commu,1,MPI_INT,0,MPI_COMM_WORLD);
	if(rank==0){
		for(i=1;i<tasks_num;i++){
			buf_commu[i]+=buf_commu[i-1];
		}
		for(i=tasks_num-1;i>0;i--){
			buf_commu[i]=buf_commu[i-1];
		}
		buf_commu[0]=0;
	}
	MPI_Scatter(buf_commu,1,MPI_INT,&offset,1,MPI_INT,0,MPI_COMM_WORLD);
	t2=MPI_Wtime();
	t_commu=t2-t1;

	t1=MPI_Wtime();
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	MPI_File_seek(fh, offset*sizeof(int), MPI_SEEK_SET); 
	MPI_File_read_all(fh, buf_io, size_each, MPI_INT, &status);
	t2=MPI_Wtime();
	if(rank==0)printf("input time:%f\n",t2-t1);

	t_main=MPI_Wtime();
	finish=0;
	tmp=0;
	phase=0;
	while(finish==0){
		finish = 1;
		if(tmp==0){
			t1=MPI_Wtime();
			qsort(buf_io,size_each,sizeof(int),compare);
			tmp=1;
			MPI_Barrier(MPI_COMM_WORLD);
			t2=MPI_Wtime();
			t_cal2=t_cal2+t2-t1;
		}
		
		//send a batch of numbers to next process
		for(i=phase;i<tasks_used-1;i+=2){
			if(rank==i){
				MPI_Send(&buf_io[size_each-size_norm/ratio],size_norm/ratio,MPI_INT,i+1,0,MPI_COMM_WORLD);
			}
			if(rank==i+1){
				MPI_Recv(buf_head,size_norm/ratio,MPI_INT,i,0,MPI_COMM_WORLD,&status);
			}
		}
		if(rank%2==1-phase&&rank!=0){
			t1=MPI_Wtime();
			finish=merge(buf_merge,buf_head,size_norm/ratio,buf_io,size_each);
			memcpy(buf_head,buf_merge,size_norm/ratio*sizeof(int));
			memcpy(buf_io,&buf_merge[size_norm/ratio],size_each*sizeof(int));	
			t2=MPI_Wtime();
			t_cal2=t_cal2+t2-t1;
		}
		
		//send them back	
		for(i=tasks_used-1;i>0;i--){
			if(i%2==1-phase){
				if(rank==i){
					MPI_Send(buf_head,size_norm/ratio,MPI_INT,i-1,0,MPI_COMM_WORLD);
				}
				if(rank==i-1){
					MPI_Recv(buf_tail,size_norm/ratio,MPI_INT,i,0,MPI_COMM_WORLD, &status);
				}
			}
		}
		if(rank%2==phase&&rank!=tasks_used-1){
			t1=MPI_Wtime();
			finish=merge(buf_merge,buf_io,size_each-size_norm/ratio,buf_tail,size_norm/ratio);
			memcpy(buf_io,buf_merge,size_each*sizeof(int));
			t2=MPI_Wtime();
			t_cal2=t_cal2+t2-t1;
		}
		

		//check if all tasks were done
		MPI_Gather(&finish,1,MPI_INT,buf_commu,1,MPI_INT,0,MPI_COMM_WORLD);
		if(rank==0){
			finish=1;
			for(i=0;i<tasks_used;i++){
				if(buf_commu[i]==0)finish=0;
			}
		}
		phase=1-phase;
		MPI_Bcast(&finish, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}	
	t2=MPI_Wtime();
	t_commu=t_commu+t2-t_main-t_cal2;
	t_cal2=t_cal1+t_cal2;
	if(rank==0)printf("calculation time:%f\n",t_cal2);
	if(rank==0)printf("communication time:%f\n",t_commu);
	
	//debug message
	/*printf("offset=%d\nrank=%d\nsize=%d\n", offset, rank,size_each);
	for(i=0; i<size_each ;i++){
		printf("%d ",buf_io[i]);
	}
	printf("\n");*/

	t1=MPI_Wtime();
	MPI_File_close(&fh);
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_RDWR | MPI_MODE_CREATE , MPI_INFO_NULL, &fh);
	MPI_File_seek(fh, offset*sizeof(int), MPI_SEEK_SET); 
	MPI_File_write_all(fh, buf_io, size_each, MPI_INT, &status);
	MPI_File_close(&fh);
	t2=MPI_Wtime();
	if(rank==0)printf("output time:%f\n",t2-t1);

	free(buf_io);
	free(buf_commu);
	free(buf_head);
	free(buf_tail);
	free(buf_merge);
	MPI_Finalize();
	return 0;
}

//return elements per node
int epn(int element_n, int task_n){
	int pos1, pos2;
	pos1 = element_n/task_n;
	pos2 = element_n%task_n;
	return pos1+1>pos2+pos1?pos1:pos1+1;
}

void swap(int *pair){
	int tmp = pair[1];
	pair[1] = pair[0];
	pair[0] = tmp;
}

int compare(const void *a, const void *b)
{
      int c = *(int *)a;
      int d = *(int *)b;
      if(c < d) {return -1;}               
      else if (c == d) {return 0;}      
      else return 1;            
}

int merge(int *sort,int *left,int leftcount,int *right,int rightcount) {
	int i,j,k,finish;
	i = 0;j = 0; k =0;
	finish=1;
	while(i<leftcount&&j<rightcount) {
		if(left[i] <= right[j]) sort[k++] = left[i++];
		else{
			sort[k++] = right[j++];
			finish=0;
		}
	}
	while(i < leftcount) sort[k++] = left[i++];
	while(j < rightcount) sort[k++] = right[j++];
	return finish;
}
