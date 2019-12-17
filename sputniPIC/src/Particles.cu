#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__host__ __device__
inline long get_idx(long v, long w, long x, long y, long z, long stride_w, long stride_x, long stride_y, long stride_z)
{
    return stride_x * stride_y * stride_z * w + stride_y * stride_z * x + stride_z * y + z;
}

__host__ __device__
inline long get_idx(long w, long x, long y, long z, long stride_x, long stride_y, long stride_z)
{
    return stride_x * stride_y * stride_z * w + stride_y * stride_z * x + stride_z * y + z;
}

__host__ __device__
inline long get_idx(long x, long y, long z, long stride_y, long stride_z)
{
    return stride_y * stride_z * x + stride_z * y + z;
}

__host__ __device__
inline long get_idx(long x, long y, long s1)
{
    return x + (y * s1);
}

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}


/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

void allocate_gpu_variables(struct particles* part, struct particles* part_gpu,
                            struct EMfield* field, struct EMfield* field_gpu,
                            struct grid* grd,      struct grid* grd_gpu)
{
    const uint32_t grid_size = grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield);
    
    // GRID 
    cudaMalloc(&(grd_gpu->XN_flat), grid_size);
    cudaMalloc(&(grd_gpu->YN_flat), grid_size);
    cudaMalloc(&(grd_gpu->ZN_flat), grid_size);
    
    cudaMemcpy(grd_gpu->XN_flat, grd->XN_flat, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(grd_gpu->YN_flat, grd->YN_flat, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(grd_gpu->ZN_flat, grd->ZN_flat, grid_size, cudaMemcpyHostToDevice);
    
    // FIELD
    cudaMalloc(&(field_gpu->Ex_flat), grid_size);
    cudaMalloc(&(field_gpu->Ey_flat), grid_size);
    cudaMalloc(&(field_gpu->Ez_flat), grid_size);
    
    cudaMalloc(&(field_gpu->Bxn_flat), grid_size);
    cudaMalloc(&(field_gpu->Byn_flat), grid_size);
    cudaMalloc(&(field_gpu->Bzn_flat), grid_size);
    
    cudaMemcpy(field_gpu->Ex_flat, field->Ex_flat, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(field_gpu->Ey_flat, field->Ey_flat, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(field_gpu->Ez_flat, field->Ez_flat, grid_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(field_gpu->Bxn_flat, field->Bxn_flat, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(field_gpu->Byn_flat, field->Byn_flat, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(field_gpu->Bzn_flat, field->Bzn_flat, grid_size, cudaMemcpyHostToDevice);

    // PARTICLE
    int part_size = part->npmax * sizeof(float);

    cudaMalloc(&(part_gpu->x), part_size);
    cudaMalloc(&(part_gpu->y), part_size);
    cudaMalloc(&(part_gpu->z), part_size);
    cudaMalloc(&(part_gpu->u), part_size);
    cudaMalloc(&(part_gpu->v), part_size);
    cudaMalloc(&(part_gpu->w), part_size);
    cudaMalloc(&(part_gpu->q), part_size);

    cudaMemcpy(part_gpu->x, part->x, part_size, cudaMemcpyHostToDevice);
    cudaMemcpy(part_gpu->y, part->y, part_size, cudaMemcpyHostToDevice);
    cudaMemcpy(part_gpu->z, part->z, part_size, cudaMemcpyHostToDevice);
    cudaMemcpy(part_gpu->u, part->u, part_size, cudaMemcpyHostToDevice);
    cudaMemcpy(part_gpu->v, part->v, part_size, cudaMemcpyHostToDevice);
    cudaMemcpy(part_gpu->w, part->w, part_size, cudaMemcpyHostToDevice);
    cudaMemcpy(part_gpu->q, part->q, part_size, cudaMemcpyHostToDevice);
}
    
__global__
void subcycle(struct particles part, struct EMfield field, struct grid grd, struct parameters param, int*index)
{
    // Safety
    atomicAdd(index, 1);
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= part.nop)
    {
        return;
    }

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param.dt/((double) part.n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part.qom*dto2/param.c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde=0, vptilde=0, wptilde=0;

    xptilde = part.x[i];
    yptilde = part.y[i];
    zptilde = part.z[i];

    // calculate the average velocity iteratively
    for(int innter=0; innter < part.NiterMover; innter++){
        // interpolation G-->P
        ix = 2 +  int((part.x[i] - grd.xStart)*grd.invdx);
        iy = 2 +  int((part.y[i] - grd.yStart)*grd.invdy);
        iz = 2 +  int((part.z[i] - grd.zStart)*grd.invdz);

        // calculate weights
        xi[0]   = part.x[i] - grd.XN[ix - 1][iy][iz];
        eta[0]  = part.y[i] - grd.YN[ix][iy - 1][iz];
        zeta[0] = part.z[i] - grd.ZN[ix][iy][iz - 1];
        xi[1]   = grd.XN[ix][iy][iz] - part.x[i];
        eta[1]  = grd.YN[ix][iy][iz] - part.y[i];
        zeta[1] = grd.ZN[ix][iy][iz] - part.z[i];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd.invVOL;
        
        // set to zero local electric and magnetic field
        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    Exl += weight[ii][jj][kk]*field.Ex[ix- ii][iy -jj][iz- kk ];
                    Eyl += weight[ii][jj][kk]*field.Ey[ix- ii][iy -jj][iz- kk ];
                    Ezl += weight[ii][jj][kk]*field.Ez[ix- ii][iy -jj][iz -kk ];
                    Bxl += weight[ii][jj][kk]*field.Bxn[ix- ii][iy -jj][iz -kk ];
                    Byl += weight[ii][jj][kk]*field.Byn[ix- ii][iy -jj][iz -kk ];
                    Bzl += weight[ii][jj][kk]*field.Bzn[ix- ii][iy -jj][iz -kk ];
                }
        
        // end interpolation
        omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);
        // solve the position equation
        ut= part.u[i] + qomdt2*Exl;
        vt= part.v[i] + qomdt2*Eyl;
        wt= part.w[i] + qomdt2*Ezl;

        udotb = ut*Bxl + vt*Byl + wt*Bzl;
            
        // THIS CODE IS BAD VVV
        
        // solve the velocity equation
        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
        
        continue;
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
        
        // THIS CODE IS BAD ^^^
        
        // update position
        part.x[i] = xptilde + uptilde*dto2;
        part.y[i] = yptilde + vptilde*dto2;
        part.z[i] = zptilde + wptilde*dto2;
    
    } // end of iteration

    // update the final position and velocity
    part.u[i]= 2.0*uptilde - part.u[i];
    part.v[i]= 2.0*vptilde - part.v[i];
    part.w[i]= 2.0*wptilde - part.w[i];
    part.x[i] = xptilde + uptilde*dt_sub_cycling;
    part.y[i] = yptilde + vptilde*dt_sub_cycling;
    part.z[i] = zptilde + wptilde*dt_sub_cycling;
    
    
    //////////
    //////////
    ////////// BC
                                
    // X-DIRECTION: BC particles
    if (part.x[i] > grd.Lx){
        if (param.PERIODICX==true){ // PERIODIC
            part.x[i] = part.x[i] - grd.Lx;
        } else { // REFLECTING BC
            part.u[i] = -part.u[i];
            part.x[i] = 2*grd.Lx - part.x[i];
        }
    }
                                                                
    if (part.x[i] < 0){
        if (param.PERIODICX==true){ // PERIODIC
            part.x[i] = part.x[i] + grd.Lx;
        } else { // REFLECTING BC
            part.u[i] = -part.u[i];
            part.x[i] = -part.x[i];
        }
    }
        
    // Y-DIRECTION: BC particles
    if (part.y[i] > grd.Ly){
        if (param.PERIODICY==true){ // PERIODIC
            part.y[i] = part.y[i] - grd.Ly;
        } else { // REFLECTING BC
            part.v[i] = -part.v[i];
            part.y[i] = 2*grd.Ly - part.y[i];
        }
    }
                                                                
    if (part.y[i] < 0){
        if (param.PERIODICY==true){ // PERIODIC
            part.y[i] = part.y[i] + grd.Ly;
        } else { // REFLECTING BC
            part.v[i] = -part.v[i];
            part.y[i] = -part.y[i];
        }
    }
                                                                
    // Z-DIRECTION: BC particles
    if (part.z[i] > grd.Lz){
        if (param.PERIODICZ==true){ // PERIODIC
            part.z[i] = part.z[i] - grd.Lz;
        } else { // REFLECTING BC
            part.w[i] = -part.w[i];
            part.z[i] = 2*grd.Lz - part.z[i];
        }
    }
                                                                
    if (part.z[i] < 0){
        if (param.PERIODICZ==true){ // PERIODIC
            part.z[i] = part.z[i] + grd.Lz;
        } else { // REFLECTING BC
            part.w[i] = -part.w[i];
            part.z[i] = -part.z[i];
        }
    }
}
    
/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    const int BLOCK_COUNT = (part->nop + (BLOCK_SIZE - 1)) / BLOCK_SIZE;

    // GPU memory
    struct particles part_gpu = *part;
    struct grid grd_gpu = *grd;
    struct EMfield field_gpu = *field;
    struct parameters param_gpu = *param;
    
    // Allocate and copy to gpu
    allocate_gpu_variables(part, &part_gpu, field, &field_gpu, grd, &grd_gpu);

    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields

        // do this V on GPU
        
        int *index, ind;
        cudaMalloc(&index, sizeof(int));
        cudaMemset(index, 0, sizeof(int));
        subcycle<<<BLOCK_COUNT, BLOCK_SIZE>>>(part_gpu, field_gpu, grd_gpu, param_gpu, index);
        cudaDeviceSynchronize();

        cudaMemcpy(&ind, index, sizeof(int), cudaMemcpyDeviceToHost);
        printf("INDEX %d\n", ind);

        
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
