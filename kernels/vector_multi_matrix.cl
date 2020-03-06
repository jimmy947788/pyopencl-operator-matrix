__kernel void vector_multi_matrix(__global float* vector,
                                 __global float* matrix,
                                 __global float* result,
                                 int row_size )
{

   int col_index = get_global_id(0);
   int col_size = get_global_size(0);

   int bit = (int)vector[col_index];
   //printf("vector[%d]=%d \n", col_index, bit);
   if(bit > 0)
   {
      float sum = 0;
      for(int i=0; i<= row_size -1; i++)
      {
         int index = (i * col_size ) + col_index; 
         sum += matrix[index];
         //printf("matrix[%d]=%f, ", index, matrix[index]);
      }
      printf("\n");
      //result[col_index] = vector[col_index] * sum;
   }
   else
   {
      result[col_index] = 0;
   }
}