#include "util.h"

#include <iostream>

using namespace std;

char rankToChar(int rank) {
    if (rank == 2 || rank == 15) {
        return '2';
    }
    if (rank < 10) {
        return '0' + rank;
    }
    if (rank == 10) {
        return '0';
    }
    if (rank == 11) {
        return 'J';
    }
    if (rank == 12) {
        return 'Q';
    }
    if (rank == 13) {
        return 'K';
    }
    return 'A';
}