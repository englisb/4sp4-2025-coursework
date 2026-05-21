// (c) 2023 ENCCS, CSC and the contributors

// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab
// Distribution of the code is not
// allowed in any form without permission
// from SwiftWare Lab.



#include <cstdlib>
#include <cassert>

#include "../include/heat.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../include/heat.h"
#include "pngwriter.h"

// Copy data on heat1 into heat2
void field_copy(field *heat1, field *heat2)
{
    assert(heat1->nx == heat2->nx);
    assert(heat1->ny == heat2->ny);
    assert(heat1->data.size() == heat2->data.size());
    std::copy(heat1->data.begin(), heat1->data.end(),
              heat2->data.begin());
}

// Swap the field data for heat1 and heat2
void field_swap(field *heat1, field *heat2)
{
    assert(heat1->nx == heat2->nx);
    assert(heat1->ny == heat2->ny);
    assert(heat1->data.size() == heat2->data.size());
    std::swap(heat1->data, heat2->data);
}

// Allocate memory for a temperature field and initialise it to zero
void field_allocate(field *heat)
{
    // Include also boundary layers
    int newSize = (heat->nx + 2) * (heat->ny + 2);
    heat->data.resize(newSize, 0.0);
}

// Calculate average temperature over the non-boundary grid cells
double field_average(field *heat)
{
     double average = 0.0;

     for (int i = 1; i < heat->nx + 1; i++) {
       for (int j = 1; j < heat->ny + 1; j++) {
         int ind = i * (heat->ny + 2) + j;
         average += heat->data[ind];
       }
     }

     average /= (heat->nx * heat->ny);
     return average;
}



// Print out a picture of the temperature distribution
void field_write(field *heat, int iter)
{
#if HAVE_PNG
    char filename[64];

    // The actual write routine takes only the actual data
    // (without boundary layers) so we need to copy an array with that.
    std::vector<double> inner_data(heat->nx * heat->ny);
    auto inner_it = inner_data.begin();
    auto row_begin = heat->data.begin() + (heat->ny + 2) + 1;
    for (int i = 0; i < heat->nx; i++) {
        auto row_end = row_begin + heat->ny;
        std::copy(row_begin, row_end, inner_it);
        inner_it += heat->ny;
        row_begin = row_end + 2;
    }

    // Write out the data to a png file
    sprintf(filename, "%s_%04d.png", "heat", iter);
    save_png(inner_data.data(), heat->nx, heat->ny, filename, 'c');
#endif //HAVE_PNG
}

