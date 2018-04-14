/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
#define N_PARTICLES (50)

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	// construct a trivial random generator engine from a time-based seed:
	num_particles = N_PARTICLES;
	std::default_random_engine gen(0);
    // This line creates a normal (Gaussian) distribution for x.
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_t(theta, std[2]);

	for(int i=0; i<num_particles; i++){
		Particle tmp;
		tmp.id = i;
		tmp.x = dist_x(gen);
		tmp.y = dist_y(gen);
		tmp.theta = dist_t(gen);
		tmp.weight = 1.0/num_particles;
		//std::cout << "partcle" << i << ": " << tmp.x << ":" << tmp.y << ":" <<  tmp.theta << std::endl;
		weights.push_back(tmp.weight);
		particles.push_back(tmp);
	}
	is_initialized = true;
	//std::cout << "init()" << std::endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine gen(0);
	std::normal_distribution<double> dist_x (0.0, std_pos[0]);
	std::normal_distribution<double> dist_y (0.0, std_pos[1]);
	std::normal_distribution<double> dist_t (0.0, std_pos[2]);

	for(int i=0; i<num_particles; i++){
		double theta = particles[i].theta;
		if(yaw_rate > 0.0001){
			particles[i].x += velocity/yaw_rate*(sin(theta + yaw_rate*delta_t) - sin(theta));
			particles[i].y += velocity/yaw_rate*(cos(theta) - cos(theta + yaw_rate*delta_t));
		}else{
			particles[i].x += velocity*cos(theta)*delta_t;
			particles[i].y += velocity*sin(theta)*delta_t;
		}
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += yaw_rate*delta_t + dist_t(gen);
		//std::cout << "partcle" << i << ": " << particles[i].x << ":" << particles[i].y << ":" <<  particles[i].theta << std::endl;

	}
	//std::cout << "prediction()" << std::endl;
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double sigma_x2 = std_landmark[0] * std_landmark[0];
	double sigma_y2 = std_landmark[1] * std_landmark[1];

	for(int i=0; i<num_particles; i++){
		double px = particles[i].x;
		double py = particles[i].y;
		double ptheta = particles[i].theta;
		double exponent = 0.0;
		for(int j=0; j<observations.size(); j++){
			LandmarkObs obsW;
			Map::single_landmark_s nearest;
			bool in_range = false;
			//convert observation object to world coordinate
			double lx = observations[j].x;
			double ly = observations[j].y;
			//std::cout << "lx:" <<lx << " ly:" << ly <<std::endl;
			obsW.x = px + cos(ptheta)*lx - sin(ptheta)*ly;
			obsW.y = py + sin(ptheta)*lx + cos(ptheta)*ly;
			obsW.id = observations[j].id;
			//std::cout << "j:" << j << " obsW" << ":" << obsW.x << ":" << obsW.y << ":" << obsW.id << std::endl;
			//find nearest landmark with obsW
			double min = 1000000000000000000;
			int idx = -1; 
			for(int k=0; k<map_landmarks.landmark_list.size(); k++){
				Map::single_landmark_s tmp_lm = map_landmarks.landmark_list[k];
				double tmp = dist(tmp_lm.x_f, tmp_lm.y_f, obsW.x, obsW.y);
				if(tmp < min){
					min = tmp;
					idx = k;
				}
			}
			nearest = map_landmarks.landmark_list[idx];
			//std::cout << "min:" << min << " nearest" << ":" << nearest.x_f << ":" << nearest.y_f << " idx:" << idx << std::endl;
		
			if (min < sensor_range) {
				in_range = true;
			}
			if (in_range){
				double dx = obsW.x - nearest.x_f;
				double dy = obsW.y - nearest.y_f;
				//std::cout << "obsW.x:" << obsW.x << " obsW.y: " << obsW.y << "x_f:" << nearest.x_f <<"y_f:" << nearest.y_f << std::endl;
				exponent += dx*dx/sigma_x2 + dy*dy*sigma_y2;
			}else{
				exponent += 100;
			}

		}
		double gauss_norm = (1/(2*M_PI*std_landmark[0]*std_landmark[1]));
		particles[i].weight = gauss_norm * exp(-0.5*exponent);
		//std::cout << gauss_norm << ":" << exponent << std::endl;
		//std::cout << gauss_norm << ":" << exponent << ":" << particles[i].weight << std::endl;
		}
	//std::cout << "updateWeights()" << std::endl;

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<Particle> new_particle;
	double w_sum = 0.0;

	//calculate total sum of the weights
	for(int i=0; i<num_particles; i++){
		w_sum += particles[i].weight;
	}
	//normalize weights
	for(int i=0; i<num_particles; i++){
		particles[i].weight /= w_sum;
	}
	//cumulate weight values
	std::vector<double> cum;
	cum.push_back(particles[0].weight);
	for(int i=0; i<num_particles-1; i++){
		cum.push_back(cum.back() + particles[i+1].weight);
	}
	//do resampling
	for(int i=0; i<num_particles; i++){
		double rnd = (double)rand()/((double)RAND_MAX+1);
		int chosenIdx = -1;
		for(int j=0; j<num_particles; j++){
			if(rnd < cum[j]){
				chosenIdx = j;
				break;
			}
		}
		new_particle.push_back(particles[chosenIdx]);
	}
	particles = new_particle;
	//std::cout << "resample()" << std::endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
