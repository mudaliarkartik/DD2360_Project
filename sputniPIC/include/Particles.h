#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"

struct particles {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;
    
    
    
};

// PARTICLE CPU

/** allocate particle arrays */
void particle_allocate(parameters*, particles*, int);

/** deallocate */
void particle_deallocate(particles*);



// GRID AND FIELD GPU

/** Allocate and copy field and grid to gpu */
void grid_field_allocate_and_copy_gpu(EMfield* field,
									  EMfield* field_gpu,
									  grid* grd,
									  grid* grd_gpu);

/** Deallocate grid and field from gpu */
void grid_field_free_gpu(EMfield* field_gpu, grid* grd_gpu);



// PARTICLE GPU

/** Allocate and copy particles to gpu */
void particles_allocate_and_copy_gpu(particles* part,
									 particles* part_gpu);

/** Copy particles from gpu to cpu */
void particles_copy_gpu_to_cpu(particles* part, 
							   particles* part_gpu);

/** Deallocate particles from gpu */
void particles_free_gpu(particles* part_gpu);



// IDS GPU

/** Allocate IDS on gpu */
void ids_allocate_gpu(interpDensSpecies* ids_gpu, grid* grd);

/** Set all the dynamic arrays in ids to zero */
void ids_set_zero_gpu(interpDensSpecies* ids_gpu, grid* grd, int species_count);

/** Copy IDS memory from gpu to cpu */
void ids_copy_gpu_to_cpu(interpDensSpecies* ids, 
						 interpDensSpecies* ids_gpu,
						 grid* grd);

/** Deallocate ids from gpu */
void ids_free_gpu(interpDensSpecies* ids_gpu);




// Main functions GPU
int mover_PC(struct particles*, struct EMfield*, struct grid*, struct parameters*);
void interpP2G(struct particles*, struct interpDensSpecies*, struct grid*);

// Main functions CPU
int mover_PC_cpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param);
void interpP2G_cpu(struct particles* part, struct interpDensSpecies* ids, struct grid* grd);


#endif
