#include "methods.h"

base::base(engine& eng1, table& tbl1) : eng(eng1), tbl(tbl1) {
}

base::~base() {
}



////////////////////////////// bayesian //////////////////////////////
// 3 modes: 1 for normal, 2 for no privacy, 3 for inf at laplace
bayesian::bayesian(engine& eng1, table& tbl1, double ep, double theta, int mode) : base(eng1, tbl1) {
	dim = tbl.dim; // number of attributes in the dataset
	// Bound for domain size of chosen attributes
	// The first one is the original bound
	// bound =  ep * tbl.size() / (4.0 * dim * theta);		// bound to nr. of cells
	bound = ep * tbl.size() / (4.0 * dim * theta);		// bound to nr. of cells
	// if (mode == 2)
	//  	bound = bound * 100;

	//cout << bound << " bound " << endl;
	// for efficiency
	int sub = log2(bound);
	int count = 0;
	while (tbl.size() * tools::nCk(dim, sub) > 2e12) {		// default 2e9
		sub--;
		bound /= 2;
		count++;
	}
	if (count) cout << "Bound reduced for efficiency: " << count << "." << endl;
	// for efficiency

	// Normal PrivBayes
	if (mode == 1){
		//model = pml_select(0.5 * ep);
		model = greedy(0.5 * ep);
		addnoise(0.5 * ep);
	}
	
	// InfBudget
	if (mode == 2){
		model = greedy_exact(ep);
		noNoise();
	}

	// Build a naive bayes network conditioned on the argument of 'naive()'
	if (mode == 3){
		model = naive(2);
		addnoise(ep);

	}

	// PML with uniform prior
	if (mode == 4){
		model = pml_select(0.5 * ep, 1);
		addnoise(0.5 * ep);
	}

	// PML with real prior
	if (mode == 5){
		model = pml_select(0.5 * ep, 2);
		addnoise(0.5 * ep);
	}

	sampling(tbl.size());

	//print_model();





}


 
bayesian::~bayesian() {
}

// Greedy estimation of the bayesNet with budget ep

/* deps is the bayesNet N
 * V is all the attributes minus the ones we already picked.
 * S is the set of already picked attributes (named "V" in the paper)
 *
 *
 *
 *
 */
vector<dependence> bayesian::greedy(double ep) {
	vector<dependence> model;
	double sens = tbl.sens;

	set<int> S;
	set<int> V = tools::setize(dim);

	for (int t = 0; t < dim; t++) {
		// Pick candidate sets
		vector<dependence> deps = S2V(S, V);		
		//cout << deps.size() << " (ios) \t";
		
		vector<double> quality;
		for (const auto& dep : deps) {
		//	cout << tbl.getScore(dep) << " ";
			quality.push_back(tbl.getScore(dep));
		}

		dependence picked = t ? deps[noise::EM(eng, quality, ep / (dim - 1), sens)] : deps[noise::EM(eng, quality, 1000.0, sens)];
		// first selection is free: all scores are zero.
	//	cout << endl << tbl.getScore(picked);
		// cout << "Dep: " << to_string(picked) << endl;
		// cout << "Cols: ";
		// for (int c : picked.cols) {cout << c << ' ';}
		// cout << endl << "Level: ";
		// for (int c : picked.lvls) {cout << c << ' ';}
		// cout << endl << "Depth: ";
		// for (int c : picked.cols) {cout << tbl.getDepth(c) << ' ';}
		// cout << endl << "Size: ";
		// for (int c : picked.cols) {cout << tbl.getSize(c) << ' ';}
		// cout << endl << "Width: ";
		// for (int i=0; i < picked.cols.size(); i++) {cout << tbl.getWidth(picked.cols[i], picked.lvls[i]) << ' ';}
	//	cout << tbl.getWidth(picked.cols[0], picked.lvls[0]);
	//	cout << endl;

		S.insert(picked.x.first);								
		V.erase(picked.x.first);
		model.push_back(picked);
	//	cout << to_string(picked) << endl;				// debug
	}
	//cout << endl;
	return model;

}

// prior_mode = 1 means uniform prior distribution
// prior_mode = 2 means prior distribution is taken from original data (possible privacy violations)
vector<dependence> bayesian::pml_select(double ep, int prior_mode) {
	vector<dependence> model;
	double sens = tbl.sens;
	double p_min;
	set<int> S;
	set<int> V = tools::setize(dim);
	int picked_index;

	for (int t = 0; t < dim; t++) {
		// Pick candidate sets
		vector<dependence> deps = S2V(S, V);
		
		// If t=0 (first selection), choose the first attribute randomly (do the same procedure as the original code)
		if (t==0) {
			vector<double> quality;
			for (const auto& dep : deps) {
				quality.push_back(tbl.getScore(dep));
			}

			picked_index = noise::EM(eng, quality, 1000.0, sens);

		} else {
			// For the other selections, get the histogram dist and add noise based on PML
			// Then get the mutual information from the noisy distributions
			// Select the dependence based on the noisy mutual information
			vector<double> quality;
			vector<vector<double>> noisy_distributions;
			vector<double> counts;
			vector<int> widths;
			double budget = ep/dim;

			for (const auto& dep : deps) {
				// get the histogram
				counts = tbl.getCounts(dep.cols, dep.lvls);
				widths = tbl.getWidth(dep.cols, dep.lvls);
				// add PML noise based on the mode
				// first, calculate p_min
				if (prior_mode==1) { // uniform
					p_min = 1.0 / accumulate(widths.begin(), widths.end(), 1.0, std::multiplies<double>()); //check this line
			//	    cout << "Uniform prior: " << p_min << " ";
				} else if (prior_mode==2) { // prior from data
					vector<double>::iterator minCount = min_element(counts.begin(), counts.end()); // also these, figure out why it doesnt work
					double sumCounts = accumulate(counts.begin(), counts.end(), 0.0);
					p_min = *minCount / sumCounts;
					cout << "Data prior: " << p_min;
				}
				// write the noise term in terms of p_min
				double noise_scale = 2/(budget+log((1-p_min)/(1-p_min*exp(budget))));
			//	cout << "Noise: " << noise_scale << endl;
				vector<double> noisy_counts;
				// add noise to counted values
				for (double count: counts) {
					noisy_counts.push_back(count + noise::nextLaplace(eng, noise_scale));
				}
				// store the noisy distributions for each dependence
				//noisy_distributions.push_back(noisy_counts);
				// now, convert the noisy counts to mutual information and store
				quality.push_back(tbl.getI(noisy_counts, widths)[dep.ptr]);
			}
			picked_index = distance(quality.begin(), max_element(quality.begin(), quality.end()));
		}

		dependence picked = deps[picked_index];
		
		S.insert(picked.x.first);								
		V.erase(picked.x.first);
		model.push_back(picked);
		cout << to_string(picked) << endl;
		// we want to return the selected dependencies and their noisy distributions
	}
	return model;
}

// Learn a naive bayesian network (for free)
vector<dependence> bayesian::naive(int root_id) {
	vector<dependence> model;
	
	// Add the root to the naive bayes model. Root should be conditioned on Null
	dependence root = dependence(vector<attribute>(), attribute(root_id, 0));
	model.push_back(root);

	vector<attribute> root_attribute;
	root_attribute.push_back(attribute(root_id, 0));
	for (int t = 0; t < dim; t++) {
		if (t == root_id)
			continue;
		dependence dep = dependence(root_attribute, attribute(t, 0));
		model.push_back(dep);
		//cout << to_string(picked) << endl;				// debug
	}
	//cout << endl;
	return model;

}


// Greedy estimation of the BayesNet without any differential privacy
vector<dependence> bayesian::greedy_exact(double ep) {
	vector<dependence> model;
	double sens = tbl.sens;

	set<int> S;
	set<int> V = tools::setize(dim);

	for (int t = 0; t < dim; t++) {
		vector<dependence> deps = S2V(S, V);		
		vector<double> quality;
		for (const auto& dep : deps) 
			quality.push_back(tbl.getScore(dep));

		dependence picked = deps[max_element(quality.begin(), quality.end()) - quality.begin()];

		//cout << to_string(picked) << " exact " << endl;

		S.insert(picked.x.first);								
		V.erase(picked.x.first);
		model.push_back(picked);
		//cout << to_string(picked) << endl;				// debug

	}
	return model;
}
// // Construct a naive bayes model. 
// vector<dependence> bayesian::naive(double ep) {
// 	vector<dependence> model;
// 	double sens = tbl.sens;

// 	set<int> S;
// 	set<int> V = tools::setize(dim);

// 	for (int t = 0; t < dim; t++) {
// 		// Pick candidate sets
// 		vector<dependence> deps = S2V(S, V);		
// 		cout << deps.size() << " (ios) \t";
		
// 		vector<double> quality;
// 		for (const auto& dep : deps) 
// 			quality.push_back(tbl.getScore(dep));

// 		dependence picked = t ? deps[noise::EM(eng, quality, ep / (dim - 1), sens)] : deps[noise::EM(eng, quality, 1000.0, sens)];
// 		// first selection is free: all scores are zero.

// 		S.insert(picked.x.first);								
// 		V.erase(picked.x.first);
// 		model.push_back(picked);
// 		//cout << to_string(picked) << endl;				// debug
// 	}
// 	cout << endl;
// 	return model;

// }




/*
	In the first call S is empty and V is all the attributes
*/
vector<dependence> bayesian::S2V(const set<int>& S, const set<int>& V) {
	vector<dependence> ans;
	for (int x : V) {
		set<vector<attribute>> exist;
		vector<vector<attribute>> parents = maximal(S, bound / tbl.getWidth(x));

		for (const vector<attribute>& p : parents)
			if (exist.find(p) == exist.end()) {
				exist.insert(p);
				ans.push_back(dependence(p, attribute(x, 0)));
			}
		if (exist.empty()) ans.push_back(dependence(vector<attribute>(), attribute(x, 0)));
	}
	return ans;
}

vector<vector<attribute>> bayesian::maximal(set<int> S, double tau) {
	vector<vector<attribute>> ans;
	if (tau < 1) return ans;
	if (S.empty()) {
		ans.push_back(vector<attribute>());
		return ans;
	}

	int last = *(--S.end());
	S.erase(--S.end());
	int depth = tbl.getDepth(last);
	set<vector<attribute>> exist;

	// with 'last' at a certain level
	for (int l = 0; l < depth; l++) {
		attribute att(last, l);
		vector<vector<attribute>> maxs = maximal(S, tau / tbl.getWidth(att));
		for (vector<attribute> z : maxs)
			if (exist.find(z) == exist.end()) {
				exist.insert(z);
				z.push_back(att);
				ans.push_back(z);
			}
	}

	// without 'last'
	vector<vector<attribute>> maxs = maximal(S, tau);
	for (vector<attribute> z : maxs)
		if (exist.find(z) == exist.end()) {
			exist.insert(z);
			ans.push_back(z);
		}

	return ans;
}

void bayesian::addnoise(double ep) {
	syn.initialize(tbl);
	for (const dependence& dep : model) {
		vector<double>& counts_syn = syn.margins[dep.cols].counts[dep.lvls];
		for (double count : tbl.getCounts(dep.cols, dep.lvls))
			counts_syn.push_back(count + noise::nextLaplace(eng, 2.0 * dim / ep));
	}

}

void bayesian::noNoise(){
	syn.initialize(tbl);
	for (const dependence& dep : model) {
		vector<double>& counts_syn = syn.margins[dep.cols].counts[dep.lvls];
		for (double count : tbl.getCounts(dep.cols, dep.lvls))
			counts_syn.push_back(count);
	}
}

void bayesian::sampling(int num) {
	for (int i = 0; i < num; i++) {
		vector<int> tuple(dim, 0);
		for (const dependence& dep : model) {
			vector<int> pre = tbl.generalize(
				tools::projection(tuple, dep.cols), 
				dep.cols, 
				dep.lvls);

			vector<double> conditional = syn.getConditional(dep, pre);
			tuple[dep.x.first] = noise::sample(eng, conditional);
		}
		syn.data.push_back(tuple);
	}
	syn.margins.clear();
}

string bayesian::print_model(){
	string ans;
	for (const dependence& dep : model) {
		ans += to_string(dep) + "\n";
	}
	return ans;
}

string bayesian::to_string(const dependence& dep) {
	string ans = to_string(dep.x);
	for (const auto& p : dep.p)
		ans += "," + to_string(p);
	return ans;
}

string bayesian::to_string(const attribute& att) {
	//return std::to_string(att.first) + "(" + std::to_string(att.second) + ")";
	return  std::to_string(att.first) + "," + std::to_string(tbl.getWidth(att.first, att.second));
}

void bayesian::printo_libsvm(const string& filename, const int& col, const set<int>& positives) {
	syn.printo_libsvm(filename, col, positives);
}

double bayesian::evaluate() {
	double sum = 0.0;
	for (const dependence& dep : model) sum += tbl.getMutual(dep);
	return sum;
}

// interface
vector<double> bayesian::getCounts(const vector<int>& mrg) {
	return syn.getCounts(mrg);
}






////////////////////////////	laplace //////////////////////////////
//laplace::laplace(engine& eng1, table& tbl1, double ep, const vector<vector<int>>& mrgs) : base(eng1, tbl1) {
//	double scale = 2.0 * mrgs.size() / ep;
//	for (const auto& mrg : mrgs) {
//		vector<double> counts = tbl.getCounts(mrg);
//		for (double& val : counts) val = max(0.0, val + noise::nextLaplace(eng, scale));
//		noisy[mrg] = counts;
//	}
//}
//
//laplace::~laplace() {
//}
//
//// interface
//vector<double> laplace::getCounts(const vector<int>& mrg) {
//	return noisy[mrg];
//}
//
//
//
////////////////////////////	contingency //////////////////////////////
//contingency::contingency(engine& eng1, table& tbl1, double ep) : base(eng1, tbl1) {
//	vector<int> hist = tbl.getHistogram();
//	vector<int> cells = tbl.cells(tbl.dimset());
//
//	vector<double> noisy(hist.begin(), hist.end());
//	for (double& val : noisy) val += noise::nextLaplace(eng, 2.0 / ep);
//
//	syn.copySettings(tbl);
//	vector<int> sampled = noise::sample(eng, noisy, tbl.size());
//	for (const int item : sampled)
//		syn.data().push_back(tools::decode(item, cells));
//}
//
//contingency::~contingency() {
//}
//
//// interface
//vector<double> contingency::getCounts(const vector<int>& mrg) {
//	return syn.getCounts(mrg);
//}
