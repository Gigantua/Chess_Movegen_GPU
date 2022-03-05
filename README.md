# Chess Movegen GPU
State of the art comparison of chess sliding piece algorithms on the GPU
It seems that for gpu having no memory lookup at all yields a performance of around 60 Billion lookups/second. 
This is very impressive since using 32 threads on an 5950X yields a performance of around 13 Billion Queens/second. 


## Currently compares CUDA performance for these algorithms:
NVIDIA GeForce RTX 3080
| Name | Performance [MQueens/s] |
| ------------- | ------------- |
 |Black Magic - Fixed shift |     6958.00  
 |QBB Algo                  |     58959.55 
 |Bob Lookup                |     1635.08  
 |Kogge Stone               |     39972.16 
 |Hyperbola Quiescence      |     16260.91 
 |Switch Lookup             |     4425.89  
 |Slide Arithm              |     18508.00 
 |Pext Lookup               |     16821.82 
 |SISSY Lookup              |     8050.17  
 |Hypercube Alg             |     1304.38  
 |Dumb 7 Fill               |     21842.60 
 |Obstruction Difference    |     59202.99 
 |Leorik                    |     55653.71 
 |SBAMG o^(o-3cbn)          |     59564.33 
 |NO HEADACHE               |     27982.63 
 |AVX Branchless Shift      |     28124.91 
 |Slide Arithmetic Inline   |     61837.82 

