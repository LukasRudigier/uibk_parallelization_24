#include "core/config.hpp"
#include "setup/fluid.hpp"
#include "setup/grid.hpp"
#include "solver/finite_volume_solver.hpp"
#include "solver/time_integrator.hpp"
#include "util/matrix.hpp"
#include "util/utility_functions.hpp"
#include "setup/mpi_handler.hpp"

#include <cmath>
#include <functional>
#include <iostream>



double Sedov_volume;

void init_Sedov(fluid_cell &fluid, double x_position, double y_position, double z_position) {
	double radius = sqrt(sim_util::square(x_position) + sim_util::square(y_position) + sim_util::square(z_position));
	double radius_init = 0.05;
	double volume_init = 4.0 / 3.0 * M_PI * radius_init * radius_init * radius_init;
	volume_init = Sedov_volume;
	double E_init = 1.0;
	double e_dens_init = E_init / volume_init;
	if (radius < 0.1) {
		fluid.fluid_data[fluid.get_index_energy()] = e_dens_init;
		fluid.fluid_data[fluid.get_index_tracer()] = 1.0;
	} else {
		fluid.fluid_data[fluid.get_index_energy()] = 1.e-5 / 0.4;
		fluid.fluid_data[fluid.get_index_tracer()] = 0.0;
	}
	fluid.fluid_data[fluid.get_index_density()] = 1.0;
	fluid.fluid_data[fluid.get_index_v_x()] = 0.0;
	fluid.fluid_data[fluid.get_index_v_y()] = 0.0;
	fluid.fluid_data[fluid.get_index_v_z()] = 0.0;
}

int main(int argc, char **argv) {

	int myRank, numProcs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

	std::cout << "My Rank: " << myRank << " Number of Rank: " << numProcs << std::endl;

	std::vector<double> bound_low(3), bound_up(3);
	bound_low[0] = -0.5;
	bound_low[1] = -0.5;
	bound_low[2] = -0.5;

	bound_up[0] = 0.5;
	bound_up[1] = 0.5;
	bound_up[2] = 0.5;

	std::vector<int> num_cells(3);
	num_cells[0] = 64;
	num_cells[1] = 64;
	num_cells[2] = 64;

	std::vector<int> tasks(3);
	tasks[0] = 2;
	tasks[1] = 1;
	tasks[2] = 1;

	// Start the MPI handler
	mpi_handler parallel_stuff(tasks);

	grid_3D global_grid(bound_low, bound_up, num_cells, 2);
	// grid_3D my_grid(bound_low, bound_up, num_cells, 2);

	// now create local grid
	grid_3D my_grid = parallel_stuff.make_local_grid(global_grid);

	std::cout <<"Anfang: " << my_grid.x_grid.get_center(0) << " " << my_grid.x_grid.get_left(0) << "\n";
	int num = my_grid.get_num_cells(0);
	std::cout << "Ende: " << my_grid.x_grid.get_center(num-1) << my_grid.x_grid.get_left(num-1) << "\n";


	// Get number of Sedov cells
	Sedov_volume = 0.0;
	int num_Sedov_cells = 0;
	int num_Sedov_cells_local = 0;
	double volume_cell = my_grid.x_grid.get_dx() * my_grid.y_grid.get_dx() * my_grid.z_grid.get_dx();

	for (int ix = 0; ix < my_grid.get_num_cells(0); ++ix) {
		double x_position = my_grid.x_grid.get_center(ix);
		for (int iy = 0; iy < my_grid.get_num_cells(1); ++iy) {
			double y_position = my_grid.y_grid.get_center(iy);
			for (int iz = 0; iz < my_grid.get_num_cells(2); ++iz) {
				double z_position = my_grid.z_grid.get_center(iz);
				double dist = sqrt(sim_util::square(x_position) + sim_util::square(y_position) + sim_util::square(z_position));
				if (dist < 0.1) {
					Sedov_volume += volume_cell;
					num_Sedov_cells_local++;
				}
			}
		}
	}

	//std::cout << " Volume of Sedov region: " << Sedov_volume << " in " << num_Sedov_cells_local << " cells\n";




	int total_num_Sedov_cells = 0;
	double total_Sedov_volume = 0.0;

	// Apply MPI_Allreduce to sum the Sedov_volume and num_Sedov_cells_local across all processes
    MPI_Allreduce(&Sedov_volume, &total_Sedov_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&num_Sedov_cells_local, &total_num_Sedov_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);


	
	if(parallel_stuff.get_rank()==0) {
		std::cout << " Volume of Sedov region: " << Sedov_volume << " in " << num_Sedov_cells << " cells\n";
	}
	


	// Now, I will create a HD fluid
	fluid hd_fluid(parallelisation::FluidType::adiabatic);
	hd_fluid.setup(my_grid);

	std::function<void(fluid_cell &, double, double, double)> function_init = init_Sedov;
	// std::function<void(fluid_cell &, double, double, double)> function_init = init_linear;

	finite_volume_solver solver_parallel(hd_fluid, parallel_stuff, global_grid);
	solver_parallel.set_init_function(function_init);

	double t_final = 0.01;
	double dt_out = 0.005;

	solver_parallel.run(my_grid, hd_fluid, t_final, dt_out);

	MPI_Finalize();
	return 0;
	
}