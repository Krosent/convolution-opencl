__kernel void sobel(const int img_height, const int img_width, const int kernel_n, __global int *src, 
                    __global int *sobel_x, __global int *sobel_y, __global int *dst) {
  
    int i = get_global_id(0);

    int minBoundry = 0;

    for(int j=0; j<img_width; j++) {
        float pixelX = 0;
        float pixelY = 0;
        for (int k_row = 0; k_row < kernel_n; k_row++) {
            for(int k_col = 0; k_col < kernel_n; k_col++) {
                int offset_row = (-1 + k_row);
                int offset_col = (-1 + k_col);

                int xPos = i + offset_row;
                int yPos = j + offset_col;

                // Mirror method to fix issue on edges. When index is beyond image edges we mirror value to inside of the image.
                // Outside image on 3 points means inside image to 3 points. 

                if (xPos >= img_height) {
                    int diff = xPos - img_height;
                    xPos = (img_height - diff) - 1;
                } if(xPos < minBoundry) {
                    int diff = xPos + minBoundry;
                    xPos = (minBoundry + diff) + 1;
                }
                if (yPos >= img_width) {
                    int diff = yPos - img_width;
                    yPos = (img_width - diff) - 1;
                } if(xPos < minBoundry) {
                    int diff = yPos - minBoundry;
                    yPos = (minBoundry + diff) + 1;
                }
                
                pixelX += sobel_x[(k_row * kernel_n) + k_col] * src[xPos * img_width + yPos];
                pixelY += sobel_y[(k_row * kernel_n) + k_col] * src[xPos * img_width + yPos];
            }
        }
        
        dst[(i * img_width) + j] = ceil(sqrt((pixelX * pixelX) + (pixelY * pixelY)));
    }
}