#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "mmio.c"

#include <omp.h>
#include <pthread.h>

// Structure to represent a triplet (COO) non zero element
typedef struct {
    int row;
    int col;
    double value;
} Triplet;

// Structure to represent a sparse matrix in COO form
typedef struct {
    int rows;
    int cols;
    int num_elements;
    Triplet *elements;
} SparseMatrix;

typedef struct {
    const SparseMatrix* mat1;
    const SparseMatrix* mat2;
    SparseMatrix* result;
    int start;  // Start row index for the thread
    int end;    // End row index for the thread
    int thread_id; // ID of thread
    pthread_mutex_t* mutex;
} ThreadData;


//Function to print out info about a sparse matrix 
void printSparseMatrixInfo(const SparseMatrix *matrix){

    // Write out matrix info
    printf("\nRows: %d, Cols: %d, Num Elements: %d\n", matrix->rows, matrix->cols, matrix->num_elements);
}


// Function to print a sparse matrix in COO form
void printSparseMatrix(const SparseMatrix *matrix) {
    
    for (int i = 0; i < matrix->num_elements; i++) {
        printf("(%d, %d,\t %lg)\n", matrix->elements[i].row, matrix->elements[i].col, matrix->elements[i].value);
    }

    // Write out matrix info
    printSparseMatrixInfo(matrix);
}


// Function to free memory allocated for a sparse matrix
void freeSparseMatrix(SparseMatrix *matrix) {
    free(matrix->elements);
    matrix->elements = NULL;
}


// Comparison function for qsort to sort Triplet array
int compareTriplets(const void *a, const void *b) {
    Triplet *tripletA = (Triplet *)a;
    Triplet *tripletB = (Triplet *)b;
    if (tripletA->row != tripletB->row) {
        return tripletA->row - tripletB->row;
    } else {
        return tripletA->col - tripletB->col;
    }
}

// Function to initialize the matrix and sort its elements
void initializeAndSort(const SparseMatrix* mat) {
    // Sort the elements array based on row and column
    qsort(mat->elements, mat->num_elements, sizeof(Triplet), compareTriplets);
}

// Updated getElement function using binary search
double getElement(const SparseMatrix* mat, int row, int col) {
    // If the matrix is not sorted, sort it
    // You can add a flag to check whether the matrix is already sorted
    // If the matrix is mostly static, consider sorting it once during initialization

    int low = 0, high = mat->num_elements - 1;

    while (low <= high) {
        int mid = (low + high) / 2;
        Triplet midElement = mat->elements[mid];

        if (midElement.row == row && midElement.col == col) {
            return midElement.value;
        } else if (midElement.row < row || (midElement.row == row && midElement.col < col)) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return 0.0; // Return 0.0 if the element is not found
}


SparseMatrix extractRows(const SparseMatrix* inputMatrix, int startRow, int endRow) {
    SparseMatrix outputMatrix;
    outputMatrix.rows = endRow - startRow + 1;
    outputMatrix.cols = inputMatrix->cols;
    outputMatrix.num_elements = 0;

    // Count the number of elements in the specified range
    for (int i = 0; i < inputMatrix->num_elements; ++i) {
        if (inputMatrix->elements[i].row >= startRow && inputMatrix->elements[i].row <= endRow) {
            outputMatrix.num_elements++;
        }
    }

    // Allocate memory for the elements
    outputMatrix.elements = (Triplet*)malloc(outputMatrix.num_elements * sizeof(Triplet));

    if (outputMatrix.elements == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    int outputIndex = 0;

    // Copy the elements in the specified range
    for (int i = 0; i < inputMatrix->num_elements; ++i) {
        if (inputMatrix->elements[i].row >= startRow && inputMatrix->elements[i].row <= endRow) {
            outputMatrix.elements[outputIndex].row = inputMatrix->elements[i].row;
            outputMatrix.elements[outputIndex].col = inputMatrix->elements[i].col;
            outputMatrix.elements[outputIndex].value = inputMatrix->elements[i].value;
            outputIndex++;
        }
    }

    return outputMatrix;
}

// Helper function to create/handle threads and perform sparse mat mul algorithm
//Input: Thread
void* threadCalculation(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    // Calculation of resulting rows
    printf("Thread %d: Processing rows from %d to %d ...\n", data->thread_id, data->start, data->end - 1);

    
    // for every threads assigned row
    for (int i = data->start; i < data->end; i++) {

        //And for each of mat2's column vectors
        for (int j = 0; j < data->mat2->cols; j++) {
            double result_value = 0.0;
            
            //For every mat1 element in that row
            for (int k = 0; k < data->mat1->cols; k++) {
                // Find corresponding elements in mat1 and mat2
                double mat1_value =  getElement(data->mat1, i, k);
                double mat2_value = getElement(data->mat2, k, j);

                // Multiply and accumulate the result
                result_value += mat1_value * mat2_value; //Inner product is here
            }

            // Add the result to the result matrix
            if (result_value != 0.0) {

                

                // Search for existing element with the same row and col indices in the result matrix
                int row = i;
                int col = j;
                int k;
                for (k = 0; k < data->result->num_elements; k++) {
                    if (data->result->elements[k].row == row && data->result->elements[k].col == col) {
                        data->result->elements[k].value += result_value;
                        break;
                    }
                }

                //Locking mutex for the critical section
                pthread_mutex_lock(data->mutex);

                // If no existing element is found, add a new element to the result matrix
                if (k == data->result->num_elements) {
                    data->result->elements[data->result->num_elements].row = row;
                    data->result->elements[data->result->num_elements].col = col;
                    data->result->elements[data->result->num_elements].value = result_value;
                    data->result->num_elements++;
                }

                pthread_mutex_unlock(data->mutex);
            }
        }
    }

    printf("Thread %d: Finished processing\n", data->thread_id);

    pthread_exit(NULL);
}

// Function to perform matrix multiplication
// Input: The  two sparse matrices to be multiplied (struct SparseMatrix)
// Output: The resulting matrix (struct SparseMatrix)
SparseMatrix multiplySparseMatrices(const SparseMatrix* mat1, const SparseMatrix* mat2, int num_threads) {

    //Initializing result matrix 
    SparseMatrix result;
    result.rows = mat1->rows;
    result.cols = mat2->cols;
    result.num_elements = 0;
    result.elements = NULL;

    if (mat1->cols != mat2->rows) {
        printf("Error: Incompatible matrix dimensions for multiplication\n");
        return result;
    }

    //Allocating the maximum possible number of nnz to result so that we dont run out of memory
    result.elements = (Triplet*)malloc(mat1->rows * mat2->cols * sizeof(Triplet));
    if (result.elements == NULL) {
        printf("Error: Memory allocation failed: No elements in mat1 && mat2\n");
        return result;
    }

    // Initialize mutex
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    // Create threads
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    int rows_per_thread = mat1->rows / num_threads; // Divide the workload among threads
    int remaining_rows = mat1->rows % num_threads; //
    int start_row = 0;
    

    // Sorting the matrices once during initialization
    initializeAndSort(mat1);
    initializeAndSort(mat2);

    for (int i = 0; i < num_threads; i++) {

        thread_data[i].mat2 = mat2;
        thread_data[i].result = &result;

        // Start end variables dictate how many elements each thread will process
        thread_data[i].start = start_row;
        thread_data[i].end = start_row + rows_per_thread + (i < remaining_rows ? 1 : 0);

        SparseMatrix extractedMat = extractRows(mat1, start_row, start_row + rows_per_thread + (i < remaining_rows ? 1 : 0));

        thread_data[i].mat1 = &extractedMat;

        //Next thread will start from here
        start_row = thread_data[i].end;

        
        
        thread_data[i].mutex = &mutex;
        thread_data[i].thread_id = i + 1;

        pthread_create(&threads[i], NULL, threadCalculation, &thread_data[i]);

        
    }

    // Join threads
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Destroy mutex
    pthread_mutex_destroy(&mutex);

    return result;
}

// Function to multiply two sparse matrices in COO form
SparseMatrix multiplySparseMatricesSeq(const SparseMatrix *mat1, const SparseMatrix *mat2) {

    // Initialize result matrix
    SparseMatrix result;
    result.rows = mat1->rows;
    result.cols = mat2->cols;
    result.num_elements = 0;
    result.elements = NULL;

    // Check if matrices can be multiplied
    if (mat1->cols != mat2->rows) {
        printf("Error: Incompatible matrix dimensions for multiplication\n");
        return result;
    }

    // Allocate memory for the result matrix
    result.elements = (Triplet *)malloc(mat1->num_elements * mat2->num_elements * sizeof(Triplet));
    if (result.elements == NULL) {
        printf("Error: Memory allocation failed: No elements in mat1 && mat2\n");
        return result;
    }

    // Perform algorithm for sparse matrix multiplication
    for (int i = 0; i < mat1->num_elements; i++) {
        for (int j = 0; j < mat2->num_elements; j++) {
            if (mat1->elements[i].col == mat2->elements[j].row) {
                // Multiply corresponding elements and add to result
                int row = mat1->elements[i].row;
                int col = mat2->elements[j].col;
                int value = mat1->elements[i].value * mat2->elements[j].value; //(MulTIPLICATION)

                // Search for existing element with the same row and col indices in the result matrix
                int k;
                for (k = 0; k < result.num_elements; k++) {
                    if (result.elements[k].row == row && result.elements[k].col == col) {
                        result.elements[k].value += value;  // Accumulate values (ADDITION)
                        break;
                    }
                }

                // If no existing element is found, add a new element to the result matrix
                if (k == result.num_elements) {
                    result.elements[result.num_elements].row = row;
                    result.elements[result.num_elements].col = col;
                    result.elements[result.num_elements].value = value;
                    result.num_elements++;

                    printf("element added: %d,  %d           %lg\n", row, col, result);
                }
            }
        }
    }

    return result;
}

// Function to perform in-place transpose of a sparse matrix in COO form
//
void transposeSparseMatrixInPlace(SparseMatrix *mat) {
    // Swap rows and columns
    int temp = mat->rows;
    mat->rows = mat->cols;
    mat->cols = temp;

    // Swap row and col in each triplet
    for (int i = 0; i < mat->num_elements; i++) {
        temp = mat->elements[i].row;
        mat->elements[i].row = mat->elements[i].col;
        mat->elements[i].col = temp;
    }
}

//Cluster vector generator; Inputs nodes from matrix and assigns them randomly to clusters
void generateClusterIDVector(int n, int MAX_VAL, int clusterID[]) {
    // Seed the random number generator with the current time
    srand(time(NULL));

    // Generate random cluster IDs
    for (int i = 0; i < n; ++i) {
        clusterID[i] = rand() % (MAX_VAL + 1);  // Generate random integer between 0 and MAX_VAL
    }
}

// Function to create a sparse matrix from a cluster ID vector
SparseMatrix createConfigMatrix(int *clusterID, int num_nodes, int num_clusters) {
    SparseMatrix omega;
    omega.rows = num_nodes;
    omega.cols = num_clusters;
    omega.num_elements = 0;
    omega.elements = NULL;

    // Count the number of non-zero elements (nodes that belong to clusters)
    for (int i = 0; i < num_nodes; ++i) {
        if (clusterID[i] > 0) {
            omega.num_elements++;
        }
    }

    // Allocate memory for the elements array
    omega.elements = (Triplet *)malloc(omega.num_elements * sizeof(Triplet));

    // Check if memory allocation is successful
    if (omega.elements == NULL) {
        printf("Memory allocation failed.\n");
        exit(1); 
    }

    // Fill the elements array with non-zero elements
    int elementIndex = 0;
    for (int i = 0; i < num_nodes; ++i) {
        if (clusterID[i] > 0) {
            omega.elements[elementIndex].row = i;
            omega.elements[elementIndex].col = clusterID[i] - 1; // Adjust for 0-based indexing
            omega.elements[elementIndex].value = 1.0; // Value is 1 for nodes that belong to clusters
            elementIndex++;
        }
    }

    return omega;
}



//Function to read a matrix from a matrix-market format file
SparseMatrix readCOOfile(FILE* f){

    MM_typecode matcode;
    int M, N, nz;

    //Perform same routines as example.c


    // Read Matrix A banner and size
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Could not process Matrix Market banner for matrix A.\n");
        fclose(f);
        // fclose(fileOmega); // Close fileOmega before exiting
        exit(1);
    }

    // Check matrix types
    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode) || mm_is_complex(matcode)) {
        fprintf(stderr, "Matrix A has an unsupported type.\n");
        fclose(f);
        // fclose(fileOmega);
        exit(1);
    }

    // Read matrix sizes
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "Error reading matrix size for matrix A.\n");
        fclose(f);
        // fclose(fileOmega);
        exit(1);
    }

    SparseMatrix mat;


    mat.rows = M;
    mat.cols = N;
    mat.num_elements = nz;
    mat.elements = (Triplet *)malloc(mat.num_elements * sizeof(Triplet));

    // Read matrix A entries from file
    for (int i = 0; i < nz; ++i) {
        fscanf(f, "%d %d %lg\n", &(mat.elements[i].row), &(mat.elements[i].col), &(mat.elements[i].value));
        mat.elements[i].row--; // Adjust from 1-based to 0-based
        mat.elements[i].col--;
    }

    return mat;

}

int main(int argc, char* argv[]) {


    printf("\n\nGraph Minor using Sparse matrix multiplication; Pthreads;  Version 2.0\n\n");
 
    FILE* fileA;

    // Check if the correct number of arguments is provided
    if ((fileA = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", argv[1]);
        return 1;
    }

    SparseMatrix A = readCOOfile(fileA);

    // Close file
    fclose(fileA);


    //Create cluster vector

    int nodes = A.rows;
    int clusters;

    int threads;

    //Displaying the adjacency matrix A
    printf("\nUndirected Graph Adjacency Matrix:\n\n");
    printSparseMatrixInfo(&A);

    //User enters the number of clusters
    printf("\n\nEnter the number of clusters: ");
    scanf("%d", &clusters);
    

    //User enters the number of threads
    // Keep prompting the user until a valid number of threads is entered
    do {
        printf("\nEnter the number of threads (1, 2, 4, or 8): ");
        scanf("%d", &threads);

        if (threads != 1 && threads != 2 && threads != 4 && threads != 8) {
            printf("Invalid number of threads. Please enter 1, 2, 4, or 8.\n");
        }
    } while (threads != 1 && threads != 2 && threads != 4 && threads != 8);
    

    // Record time
    clock_t start_time = clock();

    // Declare and allocate memory for the cluster ID vector
    int *clusterVector = (int *)malloc(nodes * sizeof(int));

    // Check if memory allocation is successful
    if (clusterVector == NULL) {
        printf("Memory allocation failed.\n");
        return 1; // Exit with an error code
    }

    // Generate the cluster ID vector
    generateClusterIDVector(nodes, clusters, clusterVector);



    // Create the configuration matrix (And transpose it (nodes)x(clusters) -> (clusters)x(nodes))
    SparseMatrix Omega = createConfigMatrix(clusterVector, nodes, clusters);

    printf("\n\nMatrix Omega:\n");
    printSparseMatrixInfo(&Omega);

    transposeSparseMatrixInPlace(&Omega);



    printf("\n\nFirst multipliction; Intermediate result C = Omega x A\n\n");

    SparseMatrix intermed = multiplySparseMatrices(&Omega, &A, threads);

    printSparseMatrixInfo(&intermed);


    transposeSparseMatrixInPlace(&Omega);


    printf("\nSecond multiplication; Final Result M = Omega x A x Omega^T\n\n");

    SparseMatrix grminor = multiplySparseMatrices(&intermed, &Omega, threads);

    // Stop recording time
    clock_t end_time = clock();
    // Calculate & print multiplication execution time
    double duration = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\n\nExecution time: %f seconds\n\n", duration);


    printf("\nGraph Minor:\n");
    printSparseMatrixInfo(&grminor);


    // Free allocated memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&Omega);
    freeSparseMatrix(&intermed);
    freeSparseMatrix(&grminor);

    return 0;
}