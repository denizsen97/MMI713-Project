#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <unordered_map>
#include <string>
#include "my_utils.h"
#include <thread>
#include <thrust/sort.h>
#include <random>
#include "omp.h"


#define VERBOSE 0
#define PAGERANK 1
#define TEST 0

int trustrank_device(bool);
int trustrank_host();

std::string run;
float epsilon;
float beta;
int augment;
int trusted_pages;


int main(int argc, char* argv[])
{
#ifdef GLOB_TEST
    openmpTest();
#endif

#ifndef GLOB_TEST

    run = argv[1];
    epsilon = std::atof(argv[2]);
    beta = std::atof(argv[3]);
    augment = std::atoi(argv[4]);
    trusted_pages = std::atoi(argv[5]);

    std::cout << "Run type: " << run << std::endl;
    std::cout << "Epsilon: " << epsilon << std::endl;
    std::cout << "Beta: " << epsilon << std::endl;
    std::cout << "Augment: " << augment << std::endl;
    std::cout << "Trusted Pages: " << trusted_pages << std::endl;


    if (run.compare("device") == 0) {
        bool t;
        std::string isThrust = argv[6];
        if (isThrust.compare("thrust") == 0)
            t = true;
        else t = false;

        return trustrank_device(t);
    } 
    if (run.compare("host") == 0) {
    
        return trustrank_host();
    }
#endif
    return 0;

}




#ifndef GLOB_TEST



int trustrank_host() {
    std::unordered_map<std::string, int> name_index = {};
    int h_sr_count = 0, h_link_count = 0, h_max_char = 0;
    thrust::host_vector<Link> h_links;

    std::cout << "OMP max Thread NUM : " << omp_get_max_threads() << std::endl;
    omp_set_num_threads(2);

    std::cout << "OMP Thread NUM after setting : " << omp_get_num_threads() << std::endl;


    //create the name hash_map and the sparse adjacency matrix
    int hasError = loadTSV("soc-redditHyperlinks-body.tsv", name_index, h_links, h_sr_count, h_link_count, h_max_char, augment);
    if (hasError) return -1;

    hasError = loadTSV("soc-redditHyperlinks-title.tsv", name_index, h_links, h_sr_count, h_link_count, h_max_char, augment);
    if (hasError) return -1;

    auto start_op = std::chrono::high_resolution_clock::now();


    //convert the hash map into a vector
    thrust::host_vector<std::string> sr_names(h_sr_count);
    thrust::for_each(thrust::host, name_index.begin(), name_index.end(), convert_sr_map(sr_names));

#if VERBOSE == 1
    for (auto& a : sr_names) {
        std::cout << a << std::endl;
    }
#endif


    std::cout << "Create degrees" << std::endl;
    thrust::host_vector<Degree> h_degrees(h_sr_count);
    std::cout << "Subreddit Count: " << h_sr_count << std::endl;
    std::cout << "Link Count: " << h_links.size() << std::endl;
    
    //compute in and out degrees of the pages
    #pragma omp parallel for num_threads(10)
    for (int i = 0; i < h_links.size(); ++i) {
        Link l = h_links[i];
        int f = l.from;
        int t = l.to;

        #pragma omp atomic
        h_degrees[f].out += 1;

        #pragma omp atomic
        h_degrees[t].in  += 1;

    }
        
    
#if PAGERANK == 1

#if VERBOSE == 1

    for (auto a : h_degrees)
    {
        std::cout << a.in << " " << a.out << std::endl;
    }
#endif

    //pagerank
    //initilize the vectors here to prevent multiple creation
    std::cout << "Create rank vectors" << std::endl;
    thrust::host_vector<float> rank_old(h_sr_count, 1 / (float)h_sr_count);
    thrust::host_vector<float> rank_new(h_sr_count, 0.0f);

    std::cout << "Start pagerank" << std::endl;
    pageRank_host(h_links, h_degrees, h_sr_count, epsilon, beta,
        rank_old, rank_new
    );
    std::cout << "End pagerank" << std::endl;

    //change the names to indices
    thrust::host_vector<int> h_indices(h_sr_count);
    thrust::counting_iterator<int> first(0);
    thrust::for_each(thrust::host, first, first + h_sr_count, fill_indices<thrust::host_vector<int>>(h_indices));
    

    //sort the pages
    auto start_sort = std::chrono::high_resolution_clock::now();
    thrust::sort_by_key(thrust::host, rank_new.begin(), rank_new.end(), h_indices.begin(), thrust::greater<float>());
    auto stop_sort = std::chrono::high_resolution_clock::now();
    auto duration_sort = std::chrono::duration_cast<std::chrono::microseconds>(stop_sort - start_sort);

    std::cout << "Sorting took " << duration_sort.count() << "microseconds" << std::endl;

#if VERBOSE == 1
    for (auto& a : h_indices) {
        std::cout << a << " ";
    }
#endif
    std::cout << std::endl;

    thrust::host_vector<int> h_trusted_indices(trusted_pages);

    for (int i = 0; i < trusted_pages; ++i) {
        h_trusted_indices[i] = h_indices[i];
        std::cout << sr_names[h_indices[i]] << " " << h_indices[i] << " -- ";
    }
    
    std::cout << std::endl;

    std::cout << "TrustRank-----------------" << std::endl;
    thrust::fill(thrust::host, rank_new.begin(), rank_new.end(), 0.0f);
    thrust::fill(thrust::host, rank_old.begin(), rank_old.end(), 1 / (float)h_sr_count);
    trustRank_host(h_links, h_degrees, h_sr_count, epsilon, beta,
        rank_old, rank_new,
        h_trusted_indices, trusted_pages
    );

    thrust::for_each(thrust::host, first, first + h_sr_count, fill_indices<thrust::host_vector<int>>(h_indices));
    
    

    thrust::sort_by_key(thrust::host, rank_new.begin(), rank_new.end(), h_indices.begin(), thrust::greater<float>());

    std::cout << "Sorting took " << duration_sort.count() << "microseconds" << std::endl;

    for (int i = 0; i < 15; i++) {
        std::cout << "(" << sr_names[h_indices[i]] << ", " << rank_new[i] << ") " ;
    }

    std::cout << std::endl;

    auto stop_op = std::chrono::high_resolution_clock::now();
    auto duration_op = std::chrono::duration_cast<std::chrono::milliseconds>(stop_op - start_op);
    std::cout << "Operation took " << duration_op.count() << "milliseconds" << std::endl;

#endif
    return 0;

}


int trustrank_device(bool isThrust) {
    std::unordered_map<std::string, int> name_index = {};
    int h_sr_count = 0, h_link_count = 0, h_max_char = 0;
    thrust::host_vector<Link> h_links;

    //create the name hash_map and the sparse adjacency matrix
    std::cout << "Read 1" << std::endl;
#if TEST == 1
    int hasError = loadTSV("hyperlink_body_test.tsv", name_index, h_links, h_sr_count, h_link_count, h_max_char);
    if (hasError) return -1;

    std::cout << "Read 2" << std::endl;
    hasError = loadTSV("hyperlink_title_test.tsv", name_index, h_links, h_sr_count, h_link_count, h_max_char);
    if (hasError) return -1;
#else 
    int hasError = loadTSV("soc-redditHyperlinks-body.tsv", name_index, h_links, h_sr_count, h_link_count, h_max_char, augment);
    if (hasError) return -1;

    std::cout << "Read 2" << std::endl;
    hasError = loadTSV("soc-redditHyperlinks-title.tsv", name_index, h_links, h_sr_count, h_link_count, h_max_char, augment);
    if (hasError) return -1;
#endif
    auto start_op = std::chrono::high_resolution_clock::now();

    thrust::host_vector<std::string> sr_names(h_sr_count);
    thrust::for_each(name_index.begin(), name_index.end(), convert_sr_map(sr_names));

#if VERBOSE == 1
    for (auto& a : sr_names) {
        std::cout << a << std::endl;
    }
#endif

    std::cout << "Create degrees" << std::endl;
    thrust::device_vector<Degree> d_degrees(h_sr_count);
    thrust::device_vector<Link> d_links = h_links;
    std::cout << "Subreddit Count: " << h_sr_count << std::endl;
    std::cout << "Link Count: " << h_links.size() << std::endl;  
    thrust::for_each(d_links.begin(), d_links.end(), compute_degrees(d_degrees));
    thrust::host_vector<Degree> h_degrees = d_degrees;

#if PAGERANK == 1

#if VERBOSE == 1

    for (auto a : h_degrees)
    {
        std::cout << a.in << " " << a.out << std::endl;
    }
#endif
    //pagerank
    //initilize the vectors here to prevent multiple creation
    std::cout << "Create rank vectors" << std::endl;
    thrust::device_vector<float> rank_old(h_sr_count, 1 / (float)h_sr_count);
    thrust::device_vector<float> rank_new(h_sr_count, 0.0f);

    std::cout << "Start pagerank" << std::endl;
    pageRank(d_links, d_degrees, h_sr_count, epsilon, beta,
        rank_old, rank_new, isThrust
    );

    std::cout << "End pagerank" << std::endl;
    thrust::device_vector<int> d_indices(h_sr_count);
    thrust::counting_iterator<int> first(0);
    thrust::for_each(first, first + h_sr_count, fill_indices<thrust::device_vector<int>>(d_indices));
    
    
    auto start_sort = std::chrono::high_resolution_clock::now();
    thrust::sort_by_key(rank_new.begin(), rank_new.end(), d_indices.begin(), thrust::greater<float>());
    auto stop_sort = std::chrono::high_resolution_clock::now();
    
    auto duration_sort = std::chrono::duration_cast<std::chrono::microseconds>(stop_sort - start_sort);
    
    std::cout << "Sorting took " << duration_sort.count() << "microseconds" << std::endl;


    thrust::host_vector<int> h_indices = d_indices;
#if VERBOSE == 1
    for (auto& a : h_indices) {
        std::cout << a << " ";
    }
#endif
    std::cout << std::endl;

    thrust::host_vector<int> h_trusted_indices(trusted_pages);

    for (int i = 0; i < trusted_pages; ++i) {
        h_trusted_indices[i] = h_indices[i];
        std::cout << sr_names[h_indices[i]] << " " << h_indices[i] << " -- ";
    }

    std::cout << std::endl;
    thrust::device_vector<int> d_trusted_indices = h_trusted_indices;

    std::cout << "TrustRank-----------------" << std::endl;
    thrust::fill(rank_new.begin(), rank_new.end(), 0.0f);
    thrust::fill(rank_old.begin(), rank_old.end(), 1 / (float)h_sr_count);
    trustRank(d_links, d_degrees, h_sr_count, epsilon, beta,
        rank_old, rank_new,
        d_trusted_indices, trusted_pages, isThrust
    );
    thrust::for_each(first, first + h_sr_count, fill_indices<thrust::device_vector<int>>(d_indices));
    thrust::sort_by_key(rank_new.begin(), rank_new.end(), d_indices.begin(), thrust::greater<float>());
    h_indices = d_indices;

    std::cout << "Create and copy rank vector from device to host" << std::endl;
    thrust::host_vector<float> h_ranks = rank_new;
    std::cout << "End create copy rank vector" << std::endl;

    for (int i = 0; i < 50; i++){
        std::cout << "(" << sr_names[h_indices[i]] << ", " << h_ranks[i] << ") ";
    }

    std::cout << std::endl;

    auto stop_op = std::chrono::high_resolution_clock::now();
    auto duration_op = std::chrono::duration_cast<std::chrono::milliseconds>(stop_op - start_op);
    std::cout << "Operation took " << duration_op.count() << "milliseconds" << std::endl;

#endif
    return 0;

}



#endif // GLOB_TEST
