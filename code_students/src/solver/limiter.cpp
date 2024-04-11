#include "solver/limiter.hpp"

#include <algorithm>
#include <iostream>

limiter_base::limiter_base() {}

limiter_minmod::limiter_minmod(double theta) { this->theta = theta; }

double limiter_minmod::compute(double first, double second, double third) {
	// done by students
	if(first > 0 && second > 0 && third > 0) {
		return std::min({first, second, third});
	}
	else if (first < 0 && second < 0 && third < 0) {
		return std::max({first, second, third});
	}
	else {
		return 0;
	}
}