#include <stdio.h>
#define MaxSize 20
typedef struct ElemType
{
    double coefficient;
    int exponent;/* data */
};

typedef struct SqList
{
    ElemType data[MaxSize];/* data */
    int length;
};

int main(){
    return 0;
}



 