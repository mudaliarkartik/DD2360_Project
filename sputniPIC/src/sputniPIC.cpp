/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

#include <vector>

// Use this to choose code to compile&run
#define VERIFY 0
#define USE_GPU 1
//////////////////////////////////////////

// Don't change this
#define USE_CPU (USE_GPU == VERIFY)

inline bool not_eq(float f1, float f2)
{
	return fabs(f1 - f2) > 0.0001f;
}

void verify_mover_PC(parameters& param, particles* part, particles* part_gpu)
{
	// Check if mover_PC computed particles correctly
	particles part_gpu_out;
	particle_allocate(&param, &part_gpu_out, 0);

	for (int is = 0; is < param.ns; is++)
	{
		bool correct = true;

		// Copy dynamic memory
		particles_copy_gpu_to_cpu(&part_gpu_out, &part_gpu[is]);

		// Compare gpu and cpu dynamic memories
		for (int i = 0; i < part_gpu_out.nop; i++)
		{
			if (not_eq(part[is].x[i], part_gpu_out.x[i]) ||
				not_eq(part[is].y[i], part_gpu_out.y[i]) ||
				not_eq(part[is].z[i], part_gpu_out.z[i]) ||
				not_eq(part[is].u[i], part_gpu_out.u[i]) ||
				not_eq(part[is].v[i], part_gpu_out.v[i]) ||
				not_eq(part[is].w[i], part_gpu_out.w[i]) ||
				not_eq(part[is].q[i], part_gpu_out.q[i]))
			{
				printf("ERROR: part not equal at species: %d, index: %d\n", is, i);
				correct = false;
				break;
			}
		}

		if (correct)
			printf("Mover_PC %d VERIFIED.\n", is);
	}

	particle_deallocate(&part_gpu_out);
}

void verify_interpP2G(grid& grd, parameters& param, interpDensSpecies* ids, interpDensSpecies* ids_gpu)
{
	// Check if mover_PC computed particles correctly
	interpDensSpecies ids_gpu_out;
	interp_dens_species_allocate(&grd, &ids_gpu_out, 0);

	for (int is = 0; is < param.ns; is++)
	{
		bool correct = true;

		// Copy dynamic memory
		ids_copy_gpu_to_cpu(&ids_gpu_out, &ids_gpu[is], &grd);

		const uint32_t grid_size = grd.nxn * grd.nyn * grd.nzn;

		// Compare gpu and cpu dynamic memories
		for (int i = 0; i < grid_size; i++)
		{
			if (not_eq(ids[is].rhon_flat[i], ids_gpu_out.rhon_flat[i]) ||
				not_eq(ids[is].Jx_flat[i], ids_gpu_out.Jx_flat[i]) ||
				not_eq(ids[is].Jy_flat[i], ids_gpu_out.Jy_flat[i]) ||
				not_eq(ids[is].Jz_flat[i], ids_gpu_out.Jz_flat[i]) ||
				not_eq(ids[is].pxx_flat[i], ids_gpu_out.pxx_flat[i]) ||
				not_eq(ids[is].pxy_flat[i], ids_gpu_out.pxy_flat[i]) ||
				not_eq(ids[is].pxz_flat[i], ids_gpu_out.pxz_flat[i]) ||
				not_eq(ids[is].pyy_flat[i], ids_gpu_out.pyy_flat[i]) ||
				not_eq(ids[is].pyz_flat[i], ids_gpu_out.pyz_flat[i]) ||
				not_eq(ids[is].pzz_flat[i], ids_gpu_out.pzz_flat[i]))
			{
				printf("ERROR: ids not equal at species: %d, index: %d\n", is, i);
				correct = false;
				break;
			}
		}

		if (correct)
			printf("interpP2G %d VERIFIED.\n", is);
	}

	interp_dens_species_deallocate(&grd, &ids_gpu_out);
}

int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
    }

    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);



#ifdef USE_GPU
	//// GPU Allocation ===========================================

		// >>> Grid and field
		grid grd_gpu = grd;
		EMfield field_gpu = field;

		// Allocate and copy dynamic memory
		grid_field_allocate_and_copy_gpu(&field, &field_gpu, &grd, &grd_gpu);


		// >>> Particles and Ids
		particles *part_gpu = new particles[param.ns];
		interpDensSpecies* ids_gpu = new interpDensSpecies[param.ns];

		for (int is = 0; is < param.ns; is++) {
			// Copy static memory
			part_gpu[is] = part[is];
			ids_gpu[is] = ids[is];

			// Allocate dynamic memory
			particles_allocate_and_copy_gpu(&part[is], &part_gpu[is]);
			ids_allocate_gpu(&ids_gpu[is], &grd);
		}

	//// ====================================================== GPU
#endif



    // Times of each iteration
    std::vector<double> timesMover;
	std::vector<double> timesInterp;

    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);

#if USE_GPU
		// GPU - Set ids to zero
		ids_set_zero_gpu(ids_gpu, &grd, param.ns);
#endif


        
        // Mover CPU & GPU
        iMover = cpuSecond(); // start timer for mover
		for (int is = 0; is < param.ns; is++)
		{
#if USE_GPU
			mover_PC(&part_gpu[is], &field_gpu, &grd_gpu, &param);
#endif
#if USE_CPU
			mover_PC_cpu(&part[is], &field, &grd, &param);
#endif
		}

        // Save current iteration time
		timesMover.push_back(cpuSecond() - iMover);
        eMover += timesMover[timesMover.size() - 1];

#if VERIFY
		// Test if result GPU == CPU
		verify_mover_PC(param, part, part_gpu);
#endif
        



        
        // InterpP2G CPU & GPU
        iInterp = cpuSecond(); // start timer for the interpolation step
		for (int is = 0; is < param.ns; is++)
		{
#if USE_GPU
			interpP2G(&part_gpu[is], &ids_gpu[is], &grd_gpu);
#endif
#if USE_CPU
			interpP2G_cpu(&part[is], &ids[is], &grd);
#endif
		}

		// Save current iteration time
		timesInterp.push_back(cpuSecond() - iInterp);
		iInterp += timesInterp[timesInterp.size() - 1];

#if VERIFY
		verify_interpP2G(grd, param, ids, ids_gpu);
#endif




		// TODO: write copy-back for the rest of the program to use 
		//		 gpu-calculated values





        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
    }  // end of one PIC cycle
    

#if USE_GPU
	//// GPU De-allocation ========================================

		// Grid and field
		grid_field_free_gpu(&field_gpu, &grd_gpu);

		// Particles and ids
		for (int is = 0; is < param.ns; is++) 
		{
			particles_free_gpu(&part_gpu[is]);
			ids_free_gpu(&ids_gpu[is]);
		}

	//// ====================================================== GPU
#endif


    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);

    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }

    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;

    // Print times of each mover_PC
    std::cout << "Mover: " << std::endl;

    for (double iter : timesMover)
    {
        std::cout << iter << std::endl;
    }

	// Print times of each iterpP2G
	std::cout << "IterpP2G: " << std::endl;

	for (double iter : timesInterp)
	{
		std::cout << iter << std::endl;
	}

    // exit
    return 0;
}


