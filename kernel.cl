int modulo(int x,int N);

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
            
            // Mod offset values in order to handle values outside of image boundries.
            int offset_row_mod = modulo(offset_row,3);
            int offset_col_mod = modulo(offset_col,3);

            // Or we can just multiple offsets by -1 to handle by mirror method
            //int offset_row = (-1 + k_row) * -1;
            //int offset_col = (-1 + k_col) * -1;

            // I will stick with the first solution in final code.
            
            pixelX += sobel_x[(k_row * kernel_width) + k_col] * src[(i + offset_row_mod) * img_width + (j + offset_col_mod)];
            pixelY += sobel_y[(k_row * kernel_width) + k_col] * src[(i + offset_row_mod) * img_width + (j + offset_col_mod)];
        }
    }
    
    dst[(i * img_width) + j] = ceil(sqrt((pixelX * pixelX) + (pixelY * pixelY)));
}

// This function taken from StackOverflow author. Source: https://stackoverflow.com/a/42131603
int modulo(int x,int N){
    return (x % N + N) %N;
}