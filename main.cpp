#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/algorithm/string.hpp>
#include <thread>
#include <fstream>

struct params {
	double mu;
	double sd;
	double s;
	params ( double i, int N, double s ) : mu(1.0*i/N), sd(1.0*i/N*(1-1.0*i/N)+0.1), s(s) {};
	void operator() (int i) {
		auto old_mu = mu;
		mu = (1+s)*mu/(1+s*mu);
		sd = (1.0/i)*(mu*(1-mu)) + pow((1+s)/(1+s*old_mu),2)*sd;
	}
};

boost::math::normal_distribution<> delta_approximation(unsigned i,const std::vector<int>& N, double s) {
	/*params param(i, N[0], s);
	for_each(N.begin(), N.end(), param);
	if( i == 1000) {
		std::cerr << i << " " << param.mu << " " << param.sd << std::endl;
	}*/
	double mu = 1.0*i/N[0];
	double sd = sqrt((1.0*i/N[0])*(1-1.0*i/N[0]))/N[0];
	for(auto n : N) {
		auto old_mu = mu;
		mu = (1+s)*mu/(1+s*mu);
		sd = (1.0/n)*(mu*(1-mu)) + pow((1+s)/(1+s*old_mu),2)*sd;
	}
	boost::math::normal_distribution<> normal(mu,sd);
	//boost::math::normal_distribution<> normal(param.mu,param.sd);
	return normal;
}

boost::math::binomial_distribution<> emission_gen(double sample_f, unsigned sample_n) {
	boost::math::binomial_distribution<> bin(sample_n, sample_f);
	return bin;
}

void calc_l(double* l,const std::vector<int>& N, unsigned i, double s,const std::vector<double>& prior, double ancient_frq, double modern_frq) {
	auto transition = delta_approximation(i, N, s);
	unsigned n1 = 184;
	unsigned n2 = 183;
	for(unsigned j=1; j<std::floor(N.back()); ++j) {
		//*l = prior[i]*boost::math::pdf(transition,j*1.0/N[0])*boost::math::pdf(emission1, i*1.0/N[0])*boost::math::pdf(emission2, j*1.0/N.back());
		auto emission1 = emission_gen(i*1.0/N[0], n1);
		auto emission2 = emission_gen(j*1.0/N.back(), n2);
		//std::cerr << boost::math::pdf(transition,j*1.0/N[0]) << " " << boost::math::pdf(emission1, 0.5*n1) << " " << boost::math::pdf(emission2, 0.721*n2) << std::endl;
		*l += prior[i]*boost::math::pdf(transition,j*1.0/N.back())*boost::math::pdf(emission1, ancient_frq*n1)*boost::math::pdf(emission2, modern_frq*n2);
	}
}

double likelihood(double s, double ancient_frq, double modern_frq) {
	double l = 0;
	int generations = 155;
	std::vector<int> N;
	double N0 = 13393.34;
	for(int i = 1; i <= generations; ++i) {
		N.push_back(std::floor(N0));
		N0 *= 1;
	}
	std::vector<double> prior;
	prior.push_back(0);
	for(int i = 1; i <= N[0]; ++i) {
		prior.push_back(1.0/i);
	}
	double sum = std::accumulate(prior.begin(), prior.end(), 0.0);
	std::transform( prior.begin(), prior.end(), prior.begin(), [sum]( double i ) { return i/sum; } );
	unsigned int THREADS = std::thread::hardware_concurrency();
	//unsigned int THREADS = 1;
	std::vector<std::thread *> t;
	std::vector<double*> part_sums;
	for(unsigned i=1; i<static_cast<unsigned>(N[0]); i+=THREADS) {
		auto max_j = std::min(THREADS, static_cast<unsigned>(N[0]-i));
		t.clear();
		part_sums.clear();
		for(unsigned j=0; j< THREADS; ++j) {
			part_sums.push_back(new double(0));
		}
		for(unsigned j=1; j<max_j; ++j) {
			t.push_back(new std::thread(calc_l, part_sums[j], std::cref(N), j+i, s, std::cref(prior), ancient_frq, modern_frq));
		}
		for( auto& thread: t ) {
			thread->join();
		}
		std::for_each(part_sums.begin(), part_sums.end(), [&l] (double *subtotal) { l += *subtotal; });
		THREADS = std::thread::hardware_concurrency();
	}
	return l;
}


int main(int argc, char* argv[]) {
	std::string filename = argv[1];
	std::ifstream file(filename);
	std::string line;
	std::vector<std::string> strs;
	std::getline(file, line);
	boost::math::chi_squared mydist(1);
	while(std::getline(file, line)) {
		boost::split(strs, line, boost::is_any_of("\t "));
		std::cerr << strs[0] << std::endl;
		double modern_frq = std::stod(strs[9]);
		double ancient_frq = std::stod(strs[6])*0.196 + std::stod(strs[7])*0.257 + std::stod(strs[6])*0.547;
		double max_s = 0;
		double max_l = 0;
		/*for(double j = 0.001; j< 1; j+=0.001) {
			auto l = likelihood(j, ancient_frq, modern_frq);
			if ( l > max_l ) {
				max_s = j;
				max_l = l;
			}
		}*/
		double s = 0.01;
		double old_l = 0.0;
		do {
			old_l = max_l;
			auto l = likelihood(s, ancient_frq, modern_frq);
			if ( l > max_l ) {
				max_s = s;
				max_l = l;
			}
			std::cerr << s << std::endl;
			s += 0.01;
		} while( old_l < max_l );
		auto H0 = likelihood(0, ancient_frq, modern_frq);
		double test = 0;
		if( H0 < max_l ) {
			test = 2*(log(max_l)-log(H0));
		}
		std::cout << strs[0] << "\t" << strs[1] << "\t" << max_s << "\t" << 1-boost::math::cdf(mydist, test) << std::endl;
	}
	return 0;
}
