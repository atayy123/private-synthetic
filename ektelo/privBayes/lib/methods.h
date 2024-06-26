#pragma once
#include <numeric>
#include <vector>
#include <algorithm>
#include <tuple>
#include <set>
#include <iostream>
using namespace std;

#include "table.h"
#include "noise.h"

class base
{
public:
	base(engine&, table&);
	~base();
	virtual vector<double> getCounts(const vector<int>&) = 0;

	engine& eng;
	table& tbl;
};


class bayesian : public base
{
public:
	bayesian(engine&, table&, double, int, int);			//eng, tbl, eps, theta
	~bayesian();

	vector<dependence> greedy(double);
	vector<dependence> pml_select(double, int);
	vector<dependence> greedy_exact(double);
	vector<dependence> naive();

	vector<dependence> S2V(const set<int>&, const set<int>&);
	//vector<vector<attribute>> maximal(set<int>, double);
	vector<vector<attribute>> maximal(set<int>, int);
	void addnoise(double);
	void noNoise(void);
	vector<double> pml_noise(vector<double>, double, int, double);

	void sampling(int, string);

	// tools
	void printo_libsvm(const string&, const int&, const set<int>&);
	string print_model();
	void print_model_txt(string);
	string to_string(const dependence&); 
	string to_string(const attribute&);
	double evaluate();

	// interface
	vector<double> getCounts(const vector<int>&);

	int dim;
	int k;
	double bound;
	vector<dependence> model;
	table syn;
	vector<vector<double>> model_counts;
};


//class laplace : public base
//{
//public:
//	laplace(engine&, table&, double, const vector<vector<int>>&);		//eng, tbl, eps, mrgs
//	~laplace();
//
//	// interface
//	vector<double> getCounts(const vector<int>&);
//
//	map<vector<int>, vector<double>> noisy;
//};
//
//
//class contingency : public base
//{
//public:
//	contingency(engine&, table&, double);
//	~contingency();
//
//	// interface
//	vector<double> getCounts(const vector<int>&);
//
//	table syn;
//};