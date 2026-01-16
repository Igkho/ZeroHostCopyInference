#pragma once
#include <cuda_runtime.h>
#include "DetectionRaw.h"
#include "helpers.h"

namespace cropandweed {

/**
 * @brief Draws bounding boxes and Track IDs directly onto the GPU image buffer.
 * * @param imageBatch Pointer to the batch of images (Device Memory)
 * @param batchSize Number of images in the batch
 * @param width Width of the images
 * @param height Height of the images
 * @param detections Pointer to detection array (Device Memory)
 * @param counts Pointer to counts array (Device Memory)
 * @param stream CUDA stream to use
 */
CudaError DrawDetections(float* imageBatch, 
                         int batchSize, 
                         int width, 
                         int height, 
                         const DetectionRaw* detections, 
                         const int* counts,
                         cudaStream_t stream = 0);

}