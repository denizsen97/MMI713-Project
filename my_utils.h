#include <device_atomic_functions.h>
#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>
#include <thrust/host_vector.h>	
#include <mutex>
#include <thread>
#include "reduction.h"
#include "omp.h"		




struct Link {
	int from;
	int to;
	public:
		Link(int f, int t) {
			this->from = f;
			this->to = t;
		}
};

struct Degree{
	int in = 0;
	int out = 0;
};


struct compute_degrees{

	Degree* d_degrees;

	public:
	compute_degrees(thrust::device_vector<Degree> &d) {
		d_degrees = thrust::raw_pointer_cast(d.data());

	}

	__device__
	void operator()(Link l){
		
		atomicAdd(&d_degrees[l.from].out, 1);
		atomicAdd(&d_degrees[l.to].in, 1);
		
	}

};




struct compute_degrees_host{

	Degree* d_degrees;
	//std::mutex *mutexes;

public:
	compute_degrees_host(thrust::host_vector<Degree>& d) {
		d_degrees = thrust::raw_pointer_cast(d.data());
		//mutexes = thrust::raw_pointer_cast(m.data());
	}

	__host__
	void operator()(Link l) {

		int f = l.from;
		int t = l.to;

		//mutexes[f].lock();

			d_degrees[f].out += 1;
			//mutexes[f].unlock();
		


			//mutexes[t].lock();
			d_degrees[t].in += 1;
			//mutexes[t].unlock();
		
	}

};


struct add_pagerank{

	float* newrank;
	float* oldrank;
	Degree* degrees;
	float beta;

public:
	add_pagerank(thrust::device_vector<float>& nranks, thrust::device_vector<float>& oranks, thrust::device_vector<Degree>& deg, float b) {
		newrank = thrust::raw_pointer_cast(nranks.data());
		oldrank = thrust::raw_pointer_cast(oranks.data());
		degrees = thrust::raw_pointer_cast(deg.data());
		beta = b;
	}


	__device__
	void operator()(Link l) {
		atomicAdd(&newrank[l.to], static_cast<float>(beta*oldrank[l.from]/degrees[l.from].out));
	}
};


struct add_pagerank_host {

	float* newrank;
	float* oldrank;
	Degree* degrees;
	float beta;
	//mutable std::mutex* mutexes;

public:
	add_pagerank_host(thrust::host_vector<float>& nranks, thrust::host_vector<float>& oranks, thrust::host_vector<Degree>& deg, float b) {
		newrank = thrust::raw_pointer_cast(nranks.data());
		oldrank = thrust::raw_pointer_cast(oranks.data());
		degrees = thrust::raw_pointer_cast(deg.data());
		//mutexes = thrust::raw_pointer_cast(m.data());
		beta = b;
	}


	__host__
	void operator()(Link l) {
		int t = l.to;
		//mutexes[t].lock();
		newrank[t] += static_cast<float>(beta*oldrank[l.from] / degrees[l.from].out);
		//mutexes[t].unlock();
	}
};


template<class RankType>
struct add_leaked_pagerank{

	float* newrank;
	float leak;

public:
	add_leaked_pagerank(RankType& nranks, float l) {
		newrank = thrust::raw_pointer_cast(nranks.data());
		leak = l;
	}

	__device__ __host__
	void operator()(int index) {

		newrank[index] += leak;

	}
};


template<class RankType, class TrustedType>
struct add_leaked_trustrank{

	float* newrank;
	float leak;
	int* trusted_indices;

public:
	add_leaked_trustrank(RankType& nranks, TrustedType& trusted, float l) {
		newrank = thrust::raw_pointer_cast(nranks.data());
		leak = l;
		trusted_indices = thrust::raw_pointer_cast(trusted.data());
	}

	__device__ __host__
	void operator()(int index) {

		newrank[trusted_indices[index]] += leak;


	}
};


template<class RankType, class DegreeType>
struct prune_pagerank{

	float* newrank;
	Degree* degrees;

public:
	prune_pagerank(RankType& nranks, DegreeType& deg) {
		newrank = thrust::raw_pointer_cast(nranks.data());
		degrees = thrust::raw_pointer_cast(deg.data());
	}

	__device__ __host__
	void operator()(int index) {

		if (degrees[index].in == 0)newrank[index] = 0.0f;

	}
};

template<class RankType>
struct calculate_error {

	float* newrank;
	float* oldrank;

public:
	
	calculate_error(RankType& nranks, RankType& oranks) {
		newrank = thrust::raw_pointer_cast(nranks.data());
		oldrank = thrust::raw_pointer_cast(oranks.data());
	}

	__device__ __host__
	void operator()(const int index) const {

		float a = newrank[index] - oldrank[index];
		oldrank[index] = a >= 0 ? a : -a;
	}

};


struct convert_sr_map {

	std::string* sr_names;
	
public:
	convert_sr_map(thrust::host_vector<std::string> &names) {
		sr_names = thrust::raw_pointer_cast(names.data());

	}

	__host__
	void operator()(const std::pair<std::string, int> &p){
		sr_names[p.second] = p.first;
	}
};


template<class IndexType>
struct fill_indices {

	int* indices;

public:

	fill_indices(IndexType& in) {
		indices = thrust::raw_pointer_cast(in.data());
	}

	__device__ __host__
	void operator()(const int index) const {

		indices[index] = index;
	}


};



int loadTSV(std::string path, std::unordered_map<std::string, int> &hash_map, thrust::host_vector<Link> &links, int& subreddit_count, int& link_count, int& max_char, int augment=0) {
	std::ifstream ip(path);

	if (!ip.is_open()) { std::cout << "ERROR opening the file" << std::endl; return 1; }

	std::string from, to, post_id, time_stamp, link_sentiment, properties;
	int indexFrom, indexTo;

	while(ip.good()){
		std::getline(ip, from, '\t');
		std::getline(ip, to, '\t');
		std::getline(ip, post_id, '\t');
		std::getline(ip, time_stamp, '\t');
		std::getline(ip, link_sentiment, '\t');
		std::getline(ip, properties, '\n');
		
		int counter = 0;

		do
		{

			++link_count;

			//std::cout << from << " " << to << std::endl;

			indexFrom = hash_map[from];
			if (indexFrom == 0)
			{
				hash_map[from] = subreddit_count++;
				if (from.size() > max_char)max_char = from.size();
				indexFrom = subreddit_count - 1;

			}

			indexTo = hash_map[to];
			if (indexTo == 0)
			{
				hash_map[to] = subreddit_count++;
				if (to.size() > max_char)max_char = to.size();
				indexTo = subreddit_count - 1;
			}

			links.push_back(Link(indexFrom, indexTo));

			from += "_aug_" + std::to_string(link_count);
			to += "_aug_" + std::to_string(link_count);


			counter++;
		} while (counter <= augment);


	}
	return 0;
}



template<typename K, typename V>
void print_map(std::unordered_map<K, V> const& m)
{
	for (auto const& pair : m) {
		std::cout << "{" << pair.first << ": " << pair.second << "}\n";
	}
}




void pageRank(
	thrust::device_vector<Link> &links, 
	thrust::device_vector<Degree> &degrees, 
	int sr_count, 
	float epsilon, 
	float beta,          
	thrust::device_vector<float> &rank_old,
	thrust::device_vector<float> &rank_new,
	bool withThrust = true
	) {
	
	float error = 0.0f;
	//thrust::device_vector<float> rank_old(sr_count, 1/(float)sr_count);
	//thrust::device_vector<float> rank_new(sr_count, 0.0f);
	long iter_count = 0;
	
	long sum_iter_time = 0;
	long sum_error_time = 0;


	while (1) {
		auto start = std::chrono::high_resolution_clock::now();
		++iter_count;
		//distribute pageranks according to the links
		thrust::for_each(links.begin(), links.end(), add_pagerank(rank_new, rank_old, degrees, beta));
		thrust::counting_iterator<int> first(0);

		//prune the pageranks of 0 in degree nodes
		thrust::for_each(first, first + sr_count, prune_pagerank<thrust::device_vector<float>, thrust::device_vector<Degree>>(rank_new, degrees));

		float x = thrust::reduce(rank_new.begin(), rank_new.end(), (float)0.0f, thrust::plus<float>());
		float leak = 1.0 - x;
		//float leak = 1.0f - reduce(rank_new, rank_old);
		thrust::for_each(first, first + sr_count, add_leaked_pagerank<thrust::device_vector<float>>(rank_new, static_cast<float>(leak / sr_count)));

		// bunlar kullanýlacak
		// auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(rank_new.begin(), rank_old.begin()));
		// auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(rank_new.end(), rank_old.end()));
		
		auto error_start = std::chrono::high_resolution_clock::now();

		if (withThrust) {
			thrust::for_each(first, first + sr_count, calculate_error<thrust::device_vector<float>>(rank_new, rank_old));
			error = thrust::reduce(rank_old.begin(), rank_old.end());
		}
		else
			error = reduce(rank_new, rank_old);
		auto error_stop = std::chrono::high_resolution_clock::now();
		auto error_duration = std::chrono::duration_cast<std::chrono::microseconds>(error_stop - error_start);

		sum_error_time += error_duration.count();

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		sum_iter_time += duration.count();
		std::cout << "Iter: " << iter_count << " Leak: " << leak << " Error: " << error << " Duration:" << duration.count() << "mcs" << " Error Duration:" << error_duration.count() << "mcs" << std::endl;

		if (iter_count >= 100) {
			break;
		}
		if (error > epsilon) {
			rank_old = rank_new;
			thrust::fill(rank_new.begin(), rank_new.end(), 0);
		}
		else break;

	}
	std::cout << "Average Error Calculation: " << sum_error_time / iter_count << " Average Iteration Time: " << sum_iter_time / iter_count << std::endl;

}


void trustRank(
	const thrust::device_vector<Link>& links,
	thrust::device_vector<Degree>& degrees,
	int sr_count,
	float epsilon,
	float beta,
	thrust::device_vector<float> &rank_old,
	thrust::device_vector<float> &rank_new,
	thrust::device_vector<int> &trusted_indices,
	int trusted_page_count,
	bool withThrust = true
	)
{

	float error = 0.0f;
	//thrust::device_vector<float> rank_old(sr_count, 1/(float)sr_count);
	//thrust::device_vector<float> rank_new(sr_count, 0.0f);
	long iter_count = 0;

	long sum_iter_time = 0;
	long sum_error_time = 0;
	long leakage_dist_time = 0;

	while (1) {
		auto start= std::chrono::high_resolution_clock::now();
		//distribute pageranks according to the links
		++iter_count;
		thrust::for_each(links.begin(), links.end(), add_pagerank(rank_new, rank_old, degrees, beta));
		thrust::counting_iterator<int> first(0);

		//prune the pageranks of 0 in degree nodes
		thrust::for_each(first, first + sr_count, prune_pagerank<thrust::device_vector<float>, thrust::device_vector<Degree>>(rank_new, degrees));

		float x = thrust::reduce(rank_new.begin(), rank_new.end(), (float)0.0f, thrust::plus<float>());
		float leak = 1.0f - x;
		//float leak = 1.0f - reduce(rank_new, rank_old);



		//as trusted rank count is mostly very small, it is better to 
		//distibute the leakage in CPU
		float portion = leak / static_cast<float>(trusted_page_count);
		thrust::for_each(first, first + trusted_indices.size(), add_leaked_trustrank<thrust::device_vector<float>, thrust::device_vector<int>>(rank_new, trusted_indices, portion));
		/*

#pragma omp parallel for num_threads(5)
		for (int i = 0; i < trusted_indices.size(); ++i) {
			int trusted_index = trusted_indices[i];

			rank_new[trusted_index] += portion;
		}*/

		// bunlar kullanýlacak
		// auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(rank_new.begin(), rank_old.begin()));
		// auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(rank_new.end(), rank_old.end()));
		auto error_start = std::chrono::high_resolution_clock::now();


		if (withThrust) {
			thrust::for_each(first, first + sr_count, calculate_error<thrust::device_vector<float>>(rank_new, rank_old));
			error = thrust::reduce(rank_old.begin(), rank_old.end());
		}
		else
			error = reduce(rank_new, rank_old);

		auto error_stop = std::chrono::high_resolution_clock::now();
		auto error_duration = std::chrono::duration_cast<std::chrono::microseconds>(error_stop - error_start);
		sum_error_time += error_duration.count();
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		sum_iter_time += duration.count();
		std::cout << "Iter: " << iter_count << " Leak: " << leak << " Error: " << error << " Duration:" << duration.count() << "mcs" << " Error Duration:" << error_duration.count() << "mcs" << std::endl;

		
		if (iter_count >= 100) {
			break;
		}
		if (error > epsilon) {
			rank_old = rank_new;
			thrust::fill(rank_new.begin(), rank_new.end(), 0);
		}
		else break;

	}

	std::cout << "Average Error Calculation: " << sum_error_time / iter_count << " Average Iteration Time: " << sum_iter_time / iter_count << std::endl;

}


void pageRank_host(
	thrust::host_vector<Link>& links,
	thrust::host_vector<Degree>& degrees,
	int sr_count,
	float epsilon,
	float beta,
	thrust::host_vector<float>& rank_old,
	thrust::host_vector<float>& rank_new
) {

	float error = 0.0f;
	//thrust::device_vector<float> rank_old(sr_count, 1/(float)sr_count);
	//thrust::device_vector<float> rank_new(sr_count, 0.0f);
	long iter_count = 0;
	long sum_iter_time = 0;
	long sum_error_time = 0;
	while (1) {
		//distribute pageranks according to the links
		//thrust::for_each(links.begin(), links.end(), add_pagerank_host(rank_new, rank_old, degrees, beta, mutexes));
		auto start = std::chrono::high_resolution_clock::now();

		++iter_count;
		

		#pragma omp parallel for num_threads(5)
		for (int i = 0; i < links.size(); ++i) {
			Link l = links[i];
			int t = l.to;
			#pragma omp atomic
			rank_new[t] += static_cast<float>(beta*rank_old[l.from] / degrees[l.from].out);
		}
			
		
		
		thrust::counting_iterator<int> first(0);

		//prune the pageranks of 0 in degree nodes
		thrust::for_each(thrust::host, first, first + sr_count, prune_pagerank<thrust::host_vector<float>, thrust::host_vector<Degree>>(rank_new, degrees));
		float leak = 1.0f - thrust::reduce(rank_new.begin(), rank_new.end(), (float)0.0f, thrust::plus<float>());


		thrust::for_each(thrust::host, first, first + sr_count, add_leaked_pagerank<thrust::host_vector<float>>(rank_new, static_cast<float>(leak / sr_count)));

		// bunlar kullanýlacak
		// auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(rank_new.begin(), rank_old.begin()));
		// auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(rank_new.end(), rank_old.end()));

		auto error_start = std::chrono::high_resolution_clock::now();
		thrust::for_each(thrust::host, first, first + sr_count, calculate_error<thrust::host_vector<float>>(rank_new, rank_old));
		error = thrust::reduce(thrust::host, rank_old.begin(), rank_old.end());
		auto error_stop = std::chrono::high_resolution_clock::now();
		auto error_duration = std::chrono::duration_cast<std::chrono::microseconds>(error_stop - error_start);
		sum_error_time += error_duration.count();

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		sum_iter_time += duration.count();

		std::cout << "Iter: " << iter_count << " Leak: " << leak << " Error: " << error << " Duration:" << duration.count() << "mcs" << " Error Duration:" << error_duration.count() << "mcs" << std::endl;


		if (iter_count >= 100) {
			break;
		}
		if (error > epsilon) {
			rank_old = rank_new;
			thrust::fill(thrust::host, rank_new.begin(), rank_new.end(), 0);
		}
		else break;

	}
	std::cout << "Average Error Calculation: " << sum_error_time / iter_count << " Average Iteration Time: " << sum_iter_time / iter_count << std::endl;

}



void trustRank_host(
	const thrust::host_vector<Link>& links,
	thrust::host_vector<Degree>& degrees,
	int sr_count,
	float epsilon,
	float beta,
	thrust::host_vector<float>& rank_old,
	thrust::host_vector<float>& rank_new,
	thrust::host_vector<int>& trusted_indices,
	int trusted_page_count
)
{
	float error = 0.0f;
	//thrust::device_vector<float> rank_old(sr_count, 1/(float)sr_count);
	//thrust::device_vector<float> rank_new(sr_count, 0.0f);
	long iter_count = 0;

	long sum_iter_time = 0;
	long sum_error_time = 0;

	while (1) {
		auto start = std::chrono::high_resolution_clock::now();

		//distribute pageranks according to the links
		++iter_count;
		//thrust::for_each(links.begin(), links.end(), add_pagerank_host(rank_new, rank_old, degrees, beta, mutexes));
		
		//add_pagerank_host add_pr(rank_new, rank_old, degrees, beta);
		//for (auto& link : links)
		//{
		//	add_pr(link);
		//}
		
		#pragma omp parallel for num_threads(4)
		for (int i = 0; i < links.size(); ++i) {
			Link l = links[i];
			int t = l.to;
			#pragma omp atomic
			rank_new[t] += static_cast<float>(beta * rank_old[l.from] / degrees[l.from].out);

		}
		

		thrust::counting_iterator<int> first(0);
		//prune the pageranks of 0 in degree nodes
		thrust::for_each(thrust::host, first, first + sr_count, prune_pagerank<thrust::host_vector<float>, thrust::host_vector<Degree>>(rank_new, degrees));

		float leak = 1.0f - thrust::reduce(rank_new.begin(), rank_new.end(), (float)0.0f, thrust::plus<float>());


		//as trusted rank count is mostly very small, it is better to 
		//distibute the leakage in CPU
		
		float portion = static_cast<float>(leak / static_cast<float>(trusted_page_count));
		thrust::for_each(thrust::host, first, first + trusted_indices.size(), add_leaked_trustrank<thrust::host_vector<float>, thrust::host_vector<int>>(rank_new, trusted_indices, portion));


		// bunlar kullanýlacak
		// auto startIterator = thrust::make_zip_iterator(thrust::make_tuple(rank_new.begin(), rank_old.begin()));
		// auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(rank_new.end(), rank_old.end()));

		auto error_start = std::chrono::high_resolution_clock::now();

		thrust::for_each(thrust::host, first, first + sr_count, calculate_error<thrust::host_vector<float>>(rank_new, rank_old));
		error = thrust::reduce(rank_old.begin(), rank_old.end());
		auto error_stop = std::chrono::high_resolution_clock::now();
		auto error_duration = std::chrono::duration_cast<std::chrono::microseconds>(error_stop - error_start);
		sum_error_time += error_duration.count();

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		sum_iter_time += duration.count();

		std::cout << "Iter: " << iter_count << " Leak: " << leak << " Error: " << error << " Duration:" << duration.count() << "mcs" << " Error Duration:" << error_duration.count() << "mcs" << std::endl;


		if (iter_count >= 100) {
			break;
		}
		if (error > epsilon) {
			rank_old = rank_new;
			thrust::fill(thrust::host, rank_new.begin(), rank_new.end(), 0);
		}
		else break;

	}

	std::cout << "Average Error Calculation: " << sum_error_time / iter_count << " Average Iteration Time: " << sum_iter_time / iter_count << std::endl;

}




