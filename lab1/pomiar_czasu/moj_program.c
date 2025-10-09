#include "pomiar_czasu.h"
#include <stdio.h>

void ioloop(int count)
{
    for (int i = 0; i < count; i++)
    {
        printf("a");
    }
    printf("\n");
}

int oploop(int count)
{
    int a = 1;
    for (int i = 0; i < count; i++)
    {
        a = a + i;
    }
    return a;
}

int main()
{
    inicjuj_czas();
    ioloop(100000);
    drukuj_czas();

    inicjuj_czas();
    printf("Wynik operacji: %d\n", oploop(100000000));
    drukuj_czas();
}