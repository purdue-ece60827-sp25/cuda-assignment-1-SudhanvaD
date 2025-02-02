
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    if (idx < size) {
        y[idx] = scale * x[idx] + y[idx];
    }
}

int runGpuSaxpy(int vectorSize) {

    std::cout << "Hello GPU Saxpy!\n";
    
    // Allocate host memory for x, original y, and computed y
    float *x, *org, *newY;
    float *d_x, *d_y;
    float scale = 6.0; 

    x = (float*) malloc(vectorSize * sizeof(float));
    org = (float*) malloc(vectorSize * sizeof(float));
    newY = (float*) malloc(vectorSize * sizeof(float));

    if (x == NULL || org == NULL || newY == NULL) {
        std::cout << "Unable to allocate memory\n";
        return -1;
    }

    
    vectorInit(x, vectorSize);
    vectorInit(org, vectorSize);
    
    // Copy org into newY so that newY starts with the same values
    memcpy(newY, org, vectorSize * sizeof(float));

   
    cudaMalloc((void**)&d_x, vectorSize * sizeof(float));
    cudaMalloc((void**)&d_y, vectorSize * sizeof(float));

    
    cudaMemcpy(d_x, x, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, newY, vectorSize * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256; 
    int numBlocks = (vectorSize + blockSize - 1) / blockSize;  

    saxpy_gpu<<<numBlocks, blockSize>>>(d_x, d_y, scale, vectorSize);

  
    cudaDeviceSynchronize();

    
    gpuAssert(cudaGetLastError(), __FILE__, __LINE__, true);

    
    cudaMemcpy(newY, d_y, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

 
    int errorCount = verifyVector(x, org, newY, scale, vectorSize);
    std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

    
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(org);
    free(newY);

    std::cout << "Lazy, you are!\n";
    std::cout << "Write code, you must\n";

    return 0;
}


/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < pSumSize) {
        
        curandState_t rng;
        curand_init(clock64(), id, 0, &rng);
        uint64_t hitCount = 0;
        for (uint64_t i = 0; i < sampleSize; i++) {
            float x = curand_uniform(&rng);
            float y = curand_uniform(&rng);
            if ((x*x) + (y*y) <= 1.0) {
                hitCount++;
			}
        }
        pSums[id] = hitCount;
    }
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t sum = 0;
    uint64_t start = id * reduceSize;
    for (uint64_t i = 0; i < reduceSize; i++) {
        uint64_t index = start + i;
        if (index < pSumSize)
            sum += pSums[index];
    }
    if (id < pSumSize / reduceSize + 1)
        totals[id] = sum;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}



double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;


	//      Insert code here
	uint64_t pSumSize = generateThreadCount;

    uint64_t *d_pSums, *d_totals;
    cudaMalloc(&d_pSums, pSumSize * sizeof(uint64_t));
    cudaMalloc(&d_totals, reduceThreadCount * sizeof(uint64_t));


    int blockSize = 256;
    int numBlocks = (generateThreadCount + blockSize - 1) / blockSize;
    generatePoints<<<numBlocks, blockSize>>>(d_pSums, pSumSize, sampleSize);
    cudaDeviceSynchronize();

    
    numBlocks = (reduceThreadCount + blockSize - 1) / blockSize;
    reduceCounts<<<numBlocks, blockSize>>>(d_pSums, d_totals, pSumSize, reduceSize);
    cudaDeviceSynchronize();

    
    uint64_t *h_totals = (uint64_t *) malloc(reduceThreadCount * sizeof(uint64_t));
    cudaMemcpy(h_totals, d_totals, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);


    uint64_t totalHits = 0;
    for (uint64_t i = 0; i < reduceThreadCount; i++){
         totalHits += h_totals[i];
    }

    approxPi = 4.0f * ((double) totalHits / (double) (generateThreadCount * sampleSize));

    cudaFree(d_pSums);
    cudaFree(d_totals);
    free(h_totals);
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
