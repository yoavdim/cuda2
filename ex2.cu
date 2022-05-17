#include "ex2.h"
#include <cuda/atomic>

#define IMG_SIZE (IMG_HEIGHT*IMG_WIDTH)
#define TILE_HEIGHT TILE_WIDTH
#define IMG_TILES (TILE_COUNT*TILE_COUNT)
#define MAP_SIZE (IMG_TILES*256)
#define THREAD_NUM 1024
#define NO_ID 0

__device__ void prefix_sum(int arr[256], int arr_size) {
    int tid = threadIdx.x;
    int increment;
    // trick: allow running multiple times (modulu) & skip threads (negative arr_size)
    // arr_size must be the same for all or __syncthreads will cause deadlock
    tid      = (arr_size > 0)? tid % arr_size : 0; 
    arr_size = (arr_size > 0)? arr_size       : -arr_size;

    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
    return; 
}

/**
* map between a thread to its tile, total tile number is TILES_COUNT^2
*/
__device__ int get_tile_id(int index) {
    int line = index / IMG_WIDTH;
    int col  = index % IMG_WIDTH;
    line = line / TILE_HEIGHT; // round down
    col  = col / TILE_WIDTH; 
    return line * TILE_COUNT + col;
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);


__device__ void process_image(uchar *all_in, uchar *all_out, uchar *maps) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int tnum = blockDim.x;

    __shared__ int histograms[IMG_TILES][256];

    for (int i = tid; i < IMG_TILES*256; i += tnum) { // set zero
        ((int*) histograms)[i] = 0;
    }
    __syncthreads();

    for (int index = tid; index < IMG_SIZE; index += tnum) { // calc histograms
	int tile = get_tile_id(index);
	uchar pix_val = all_in[IMG_SIZE*bid + index];
	int *hist = &(histograms[tile][pix_val]);
        atomicAdd(hist, 1);
    }
    __syncthreads();

    // run prefix sum in each tile --- ASSUME: tnum  >= 256
    for (int run=0; run < (IMG_TILES/(tnum/256)+1); run++) { // enforce same amount of entries to prefix_sum
        int tile = (tid/256) + run*(tnum/256);
        if (tile >= IMG_TILES) 
            prefix_sum(NULL, -256);  // keep internal syncthread from blocking the rest
        else 
            prefix_sum(&(histograms[tile][0]), 256);
    }

//    for (int i = 0; i < IMG_TILES ; i++) {
//	    prefix_sum(histograms[bid][i], 256);
//	    __syncthreads();
//    }

    __syncthreads();

    // create map
    for (int i = tid; i < IMG_TILES*256; i += tnum) { 
        int cdf = ((int*) histograms)[i];
//        maps[MAP_SIZE*bid + i] = (uchar) ((((double)cdf)*255)/(TILE_WIDTH*TILE_HEIGHT)); // cast will round down
	    uchar map_value =(((double)cdf) / (TILE_WIDTH*TILE_HEIGHT)) * 255;
	    maps[MAP_SIZE*bid + i] = map_value;
    }
    __syncthreads();

    interpolate_device(maps + MAP_SIZE*bid, all_in + IMG_SIZE * bid, all_out + IMG_SIZE * bid);

    __syncthreads();
}

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

// ----------------------------------

struct task_context {
    uchar *d_image_in;
    uchar *d_image_out;
    uchar *d_maps; 
};

class streams_server : public image_processing_server
{
private:
    cudaStream_t streams[STREAM_COUNT];
    task_context contexts[STREAM_COUNT];
    int ids[STREAM_COUNT];
public:
    streams_server()
    {
        for(int i=0; i<STREAM_COUNT; i++) {
            cudaStreamCreate(&streams[i]);

            CUDA_CHECK(cudaMalloc((void**)&(contexts[i].d_image_in),  IMG_SIZE));
            CUDA_CHECK(cudaMalloc((void**)&(contexts[i].d_image_out), IMG_SIZE));
            CUDA_CHECK(cudaMalloc((void**)&(contexts[i].d_maps), MAP_SIZE));

            ids[i] = NO_ID;
        }
    }

    ~streams_server() override
    {
        for(int i=0; i<STREAM_COUNT; i++) {
            cudaFree(context[i].d_image_in);
            cudaFree(context[i].d_image_out);
            cudaFree(context[i].d_maps);

            cudaStreamDestroy(streams[i]);
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        // not safe for MT in cpu
        for( int s=0; s<STREAM_COUNT; s++) {
            // check empty
            if(ids[s] != NO_ID)
                continue;
            // start
            CUDA_CHECK(cudaMemcpyAsync(contexts[s].d_image_in, img_in, IMG_SIZE, cudaMemcpyHostToDevice, streams[s]));
            process_image_kernel<<<1,THREAD_NUM,0,streams[s]>>>(contexts[s].d_image_in, contexts[s].d_image_out, contexts[s].d_maps); 
            CUDA_CHECK(cudaMemcpyAsync(img_out, contexts[s].d_image_out, IMG_SIZE, cudaMemcpyDeviceToHost, streams[s]));
            ids[s] = img_id;
            return true;
        }
        return false;
    }

    bool dequeue(int *img_id) override
    {
        return false;

        // TODO query (don't block) streams for any completed requests.
        for( int s=0; s<STREAM_COUNT; s++) {
            if(ids[s] == NO_ID)
                continue;
            cudaError_t status = cudaStreamQuery(streams[s]); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                *img_id = ids[s];
                ids[s] = NO_ID; // mark for reuse
                return true;
            case cudaErrorNotReady:
                break; // and continue loop
            default:
                CUDA_CHECK(status);
                return false;
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

// TODO implement a lock
// TODO implement a MPMC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
public:
    queue_server(int threads)
    {
        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        return false;

        // TODO return the img_id of the request that was completed.
        //*img_id = ... 
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
