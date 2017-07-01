#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int epn(int, int);
void swap(int*);

int main(int argc, char *argv[]){
	int size_i, size_norm, size_each, tasks_num, tasks_used, rank, i, finish, offset, tmp;
	int *buf_io, *buf_commu, *buf_seq;
	double t1,t2,t_cal1, t_cal2, t_main, t_commu;
	MPI_File fh;
	FILE *fh2;
	MPI_Status status;
	
	size_i=atoi(argv[1]);
	t_cal2=0;

	MPI_Init(&argc, &argv);
	t1=MPI_Wtime();

	MPI_Comm_size(MPI_COMM_WORLD, &tasks_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	size_norm=epn(size_i,tasks_num);

	//how many tasks actually used and number of elements of each process
	//handle small input
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
			if(rank==0)tasks_used=size_i%size_norm>0?size_i/size_norm+1:size_i/size_norm;
		}
	}
	t2=MPI_Wtime();
	t_cal1=t2-t1;
	t1=MPI_Wtime();
	MPI_Bcast(&tasks_used, 1, MPI_INT, 0, MPI_COMM_WORLD);
	buf_io=malloc(size_each*sizeof(int));
	buf_commu=malloc(tasks_num*sizeof(int));
	if(rank==0)buf_seq=malloc(size_i*sizeof(int));

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
	MPI_File_close(&fh);
	t2=MPI_Wtime();
	/*if(rank==0){
		fh2=fopen(argv[2],"rb");
		fread(buf_seq,sizeof(int),size_i,fh2);
		fclose(fh2);
	}*/
	if(rank==0)printf("input time:%f\n",t2-t1);

	t_main=MPI_Wtime();
	finish=0;
	while(finish==0){
		finish = 1;
		//Even
		t1=MPI_Wtime();
		for(i=0;i<size_each-1;i+=2){
			if(buf_io[i]>buf_io[i+1]){
				swap(&buf_io[i]);
				finish = 0;
			}
		}
		//Odd
		for(i=1;i<size_each-1;i+=2){
			if(buf_io[i]>buf_io[i+1]){
				swap(&buf_io[i]);
				finish=0;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		t2=MPI_Wtime();
		t_cal2=t_cal2+t2-t1;
		
		//send last element to next process
		for(i=0;i<tasks_used-1;i++){
			if(rank==i){
				MPI_Send(&buf_io[size_each-1],1,MPI_INT,i+1,0,MPI_COMM_WORLD);
			}
			if(rank==i+1){
				MPI_Recv(&tmp,1,MPI_INT,i,0,MPI_COMM_WORLD,&status);
			}
		}

		//send smaller value to previous process
		for(i=tasks_used-1;i>0;i--){
			if(rank==i){
				if(tmp>buf_io[0]){
					MPI_Send(&buf_io[0],1,MPI_INT,i-1,0,MPI_COMM_WORLD);
					buf_io[0]=tmp;
					finish = 0;
				}
				else{
					MPI_Send(&tmp,1,MPI_INT,i-1,0,MPI_COMM_WORLD);
				}
			}
			if(rank==i-1){
				MPI_Recv(&buf_io[size_each-1],1,MPI_INT,i,0,MPI_COMM_WORLD,&status);	
			}
		}

		//check if all tasks were done
		MPI_Gather(&finish,1,MPI_INT,buf_commu,1,MPI_INT,0,MPI_COMM_WORLD);
		if(rank==0){
			finish=1;
			for(i=0;i<tasks_used;i++){
				if(buf_commu[i]==0)finish=0;
			}
		}
		MPI_Bcast(&finish, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}	
	t2=MPI_Wtime();
	t_commu=t_commu+t2-t_main-t_cal2;
	t_cal2=t_cal1+t_cal2;
	if(rank==0)printf("calculation time:%f\n",t_cal2);
	if(rank==0)printf("communication time:%f\n",t_commu);

	/*printf("offset=%d\nrank=%d\nsize=%d\n", offset, rank,size_each);
	for(i=0; i<size_each ;i++){
		printf("%d ",buf_io[i]);
	}
	printf("\n");*/

	t1=MPI_Wtime();
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_RDWR|MPI_MODE_CREATE , MPI_INFO_NULL, &fh);
	MPI_File_seek(fh, offset*sizeof(int), MPI_SEEK_SET); 
	MPI_File_write_all(fh, buf_io, size_each, MPI_INT, &status);
	MPI_File_close(&fh);
	/*if(rank==0){
		fh2=fopen(argv[3],"wb");
		fwrite(buf_seq,sizeof(int),size_i,fh2);
		fclose(fh2);
	}*/
	t2=MPI_Wtime();
	if(rank==0)printf("output time:%f\n",t2-t1);

	free(buf_io);
	free(buf_commu);
	MPI_Finalize();
	return 0;
}

//return elements per node
int epn(int element_n, int task_n){
	int tmp, pos1, pos2;
	tmp = element_n/task_n;
	if(tmp%2==0){
		pos1 = tmp;
		pos2 = tmp+2;
	}
	else{
		pos1 = tmp-1;
		pos2 = tmp+1;
	}
	return pos2 > element_n-(task_n-1)*pos1 ? pos1 : pos2;
}

void swap(int *pair){
	int tmp = pair[1];
	pair[1] = pair[0];
	pair[0] = tmp;
}
