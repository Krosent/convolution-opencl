__kernel void sobel(const int img_height, const int img_width, const int kernel_height, const int kernel_width, __global int *src, 
                    __global int *sobel_x, __global int *sobel_y, __global int *dst) {
  
   float pixelX = 0;
   float pixelY = 0;
   int i = get_global_id(0);
   int j = get_global_id(1);

    for (int k_row = 0; k_row < kernel_height; k_row++) {
        for(int k_col = 0; k_col < kernel_width; k_col++) {
            int offset_row = (-1 + k_row);
            int offset_col = (-1 + k_col);
              
            pixelX += sobel_x[(k_row * kernel_width) + k_col] * src[(i + offset_row) * img_width + (j + offset_col)];
            pixelY += sobel_y[(k_row * kernel_width) + k_col] * src[(i + offset_row) * img_width + (j + offset_col)];
        }
    }
    
    dst[(i * img_width) + j] = ceil(sqrt((pixelX * pixelX) + (pixelY * pixelY)));
}