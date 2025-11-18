

#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

omp_lock_t lock;

int main()
{
	omp_init_lock(&lock);
	srand(time(NULL));
	int a=0,i;
	omp_set_num_threads(4);
	printf("Petla firstprivate:\n");
	#pragma omp parallel for firstprivate(a) 
		for(i=0;i<10;i++)
		{
			printf("Watek %d a=%d\n",omp_get_thread_num(),a);			
			a++;						
		}
	printf("Po petli firstprivate a=%d\n\n",a);
	
	printf("Petla private:\n");
	#pragma omp parallel for private(a) 
		for(a=0;a<10;a++)
		{			
			printf("Watek %d a=%d\n",omp_get_thread_num(),a);
		}
	printf("Po petli private a=%d\n\n",a);
	printf("Petla lastprivate:\n");
	#pragma omp parallel for lastprivate(a) 
		for(i=0;i<10;i++)
		{
			//1. spróbuj zmienić tą wartość na zmienną losową i zobacz jak to działa
			a=rand()%11;			
			printf("Watek %d a=%d\n",omp_get_thread_num(),a);
		}
	printf("Po petli lastprivate a=%d\n\n",a);
	
	printf("Petla shared:\n");	
	a=0;
	#pragma omp parallel for shared(a) 
	for(i=0;i<10;i++)
	{
		//2. Co się stanie gdy wyłączymy zamek?
		omp_set_lock(&lock);
		a=omp_get_thread_num();
		printf("Watek %d a=%d\n",omp_get_thread_num(),a);	
		omp_unset_lock(&lock);				
	}
	//3. Jaka bedzie wartosc "a" po kilkukrotnym wywołaniu programu? 
	printf("Po petli shared a=%d\n\n",a);
	
	printf("Petla bez zadnej klauzuli:\n");	
	a=0;
	#pragma omp parallel for 
	for(i=0;i<100;i++)
	{		
		a++;
		printf("Thread %d Iteration %d a=%d\n",omp_get_thread_num(),i,a);
	}
	//4. Jaka jest domysla klauzula? 
	printf("Po petli bez klauzuli a=%d\n",a);	
}	
