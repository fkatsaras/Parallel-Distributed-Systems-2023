#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "mmio.c"

#include <pthread.h>


// Structure to represent a triplet (COO) element
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


// Function to multiply two sparse matrices in COO form
SparseMatrix multiplySparseMatrices(const SparseMatrix *mat1, const SparseMatrix *mat2) {

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
    result.elements = (Triplet *)malloc(mat1->rows * mat2->cols * sizeof(Triplet)); // Checking for mem allocation issue
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
                double value = mat1->elements[i].value * mat2->elements[j].value;

                // Search for existing element with the same row and col indices in the result matrix
                int k;
                for (k = 0; k < result.num_elements; k++) {
                    if (result.elements[k].row == row && result.elements[k].col == col) {
                        result.elements[k].value += value;  // Accumulate values
                        break;
                    }
                }

                // If no existing element is found, add a new element to the result matrix
                if (k == result.num_elements) {
                    result.elements[result.num_elements].row = row;
                    result.elements[result.num_elements].col = col;
                    result.elements[result.num_elements].value = value;
                    result.num_elements++;
                }
            }
        }
    }

    return result;
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


// Function to perform in-place transpose of a sparse matrix in COO form
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

// Function to print a sparse matrix in COO form
void printSparseMatrix(const SparseMatrix *matrix) {
    printf("Rows: %d, Cols: %d, Num Elements: %d\n", matrix->rows, matrix->cols, matrix->num_elements);
    for (int i = 0; i < matrix->num_elements; i++) {
        printf("(%d, %d, %20.19g)\n", matrix->elements[i].row, matrix->elements[i].col, matrix->elements[i].value);
    }
}

void printSparseMatrixInfo(const SparseMatrix *matrix){

    // Write out matrix info
    printf("Rows: %d, Cols: %d, Num Elements: %d\n", matrix->rows, matrix->cols, matrix->num_elements);
}

// Function to free memory allocated for a sparse matrix
void freeSparseMatrix(SparseMatrix *matrix) {
    free(matrix->elements);
    matrix->elements = NULL;
}

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

    //Displaying the adjacency matrix A
    printf("\nUndirected Graph Adjacency Matrix:\n\n");
    printSparseMatrixInfo(&A);

    //User enters the number of clusters
    printf("\n\nEnter the number of clusters: ");
    scanf("%d", &clusters);
    printf("\nGenerating Graph Minor...\n");

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

    printf("\nMatrix Omega:\n");
    printSparseMatrixInfo(&Omega);

    transposeSparseMatrixInPlace(&Omega);



    printf("\nFirst multiplication; Intermediate result C = Omega x A...\n\n");

    SparseMatrix intermed = multiplySparseMatrices(&Omega, &A);


    transposeSparseMatrixInPlace(&Omega);


    printf("\nSecond multiplication; Final Result M = Omega x A x Omega^T...\n\n");

    SparseMatrix grminor = multiplySparseMatrices(&intermed, &Omega);

    // Stop recording time
    clock_t end_time = clock();
    // Calculate & print multiplication execution time
    double duration = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\n\nExecution time: %f seconds\n\n", duration);


    printf("\nGraph Minor:\n");
    printSparseMatrixInfo(&grminor);
    // printSparseMatrix(&grminor);


    // Free allocated memory
    freeSparseMatrix(&A);
    freeSparseMatrix(&Omega);
    freeSparseMatrix(&intermed);
    freeSparseMatrix(&grminor);

    return 0;
}