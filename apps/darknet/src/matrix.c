#include "matrix.h"
#include "utils.h"
#include "printf.h"
#include "tiny_malloc.h"

void free_matrix(matrix m)
{
    int i;
    for(i = 0; i < m.rows; ++i) free(m.vals[i]);
    free(m.vals);
}

float matrix_topk_accuracy(matrix truth, matrix guess, int k)
{
    int *indexes = calloc(k, sizeof(int));
    int n = truth.cols;
    int i,j;
    int correct = 0;
    for(i = 0; i < truth.rows; ++i){
        top_k(guess.vals[i], n, k, indexes);
        for(j = 0; j < k; ++j){
            int class = indexes[j];
            if(truth.vals[i][class]){
                ++correct;
                break;
            }
        }
    }
    free(indexes);
    return (float)correct/truth.rows;
}

void scale_matrix(matrix m, float scale)
{
    int i,j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            m.vals[i][j] *= scale;
        }
    }
}

matrix resize_matrix(matrix m, int size)
{
    int i;
    if (m.rows == size) return m;
    if (m.rows < size) {
        m.vals = realloc(m.vals, size*sizeof(float*));
        for (i = m.rows; i < size; ++i) {
            m.vals[i] = calloc(m.cols, sizeof(float));
        }
    } else if (m.rows > size) {
        for (i = size; i < m.rows; ++i) {
            free(m.vals[i]);
        }
        m.vals = realloc(m.vals, size*sizeof(float*));
    }
    m.rows = size;
    return m;
}

void matrix_add_matrix(matrix from, matrix to)
{
    if (from.rows == to.rows && from.cols == to.cols)
    {
        printf("matrix_add_matrix: matrix size mismatch\n");
        return;
    }
    int i,j;
    for(i = 0; i < from.rows; ++i){
        for(j = 0; j < from.cols; ++j){
            to.vals[i][j] += from.vals[i][j];
        }
    }
}

matrix make_matrix(int rows, int cols)
{
    int i;
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.vals = calloc(m.rows, sizeof(float *));
    for(i = 0; i < m.rows; ++i){
        m.vals[i] = calloc(m.cols, sizeof(float));
    }
    return m;
}

matrix hold_out_matrix(matrix *m, int n)
{
    int i;
    matrix h;
    h.rows = n;
    h.cols = m->cols;
    h.vals = calloc(h.rows, sizeof(float *));
    for(i = 0; i < n; ++i){
        int index = rand()%m->rows;
        h.vals[i] = m->vals[index];
        m->vals[index] = m->vals[--(m->rows)];
    }
    return h;
}

float *pop_column(matrix *m, int c)
{
    float *col = calloc(m->rows, sizeof(float));
    int i, j;
    for(i = 0; i < m->rows; ++i){
        col[i] = m->vals[i][c];
        for(j = c; j < m->cols-1; ++j){
            m->vals[i][j] = m->vals[i][j+1];
        }
    }
    --m->cols;
    return col;
}

void print_matrix(matrix m)
{
    int i, j;
    printf("%d X %d Matrix:\n",m.rows, m.cols);
    printf(" __");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < m.rows; ++i){
        printf("|  ");
        for(j = 0; j < m.cols; ++j){
            printf("%15.7f ", m.vals[i][j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__|\n");
}
