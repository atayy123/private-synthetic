#include "methods.h"

base::base(engine& eng1, table& tbl1) : eng(eng1), tbl(tbl1) {
}

base::~base() {
}



////////////////////////////// bayesian //////////////////////////////
// 3 modes: 1 for normal, 2 for no privacy, 3 for inf at laplace
bayesian::bayesian(engine& eng1, table& tbl1, double ep, int max_k, int mode) : base(eng1, tbl1) {
	// double theta: variable deleted
	dim = tbl.dim; // number of attributes in the dataset
	// Bound for domain size of chosen attributes
	// The first one is the original bound
	// bound =  ep * tbl.size() / (4.0 * dim * theta);		// bound to nr. of cells
	// bound = ep * tbl.size() / (4.0 * dim * theta);		// bound to nr. of cells
	// if (mode == 2)
	//  	bound = bound * 100;
	vector<vector<double>> model_counts;
	k = max_k;
	//cout << bound << " bound " << endl;
	// // for efficiency
	// int sub = log2(bound);
	// int count = 0;
	// while (tbl.size() * tools::nCk(dim, sub) > 2e12) {		// default 2e9
	// 	sub--;
	// 	bound /= 2;
	// 	count++;
	// }
	// if (count) cout << "Bound reduced for efficiency: " << count << "." << endl;
	// // for efficiency

	string filename = "Ep"+std::to_string(ep)+"Mode"+std::to_string(mode);

	// Normal PrivBayes
	if (mode == 1){
		//model = pml_select(0.5 * ep);
		//cout << "DP Model Sel: " << 0.5*ep << endl;
		model = greedy(0.3 * ep);
		//cout << "Noise: " << 0.5*ep << endl;
		addnoise(0.7 * ep);
	}
	
	// InfBudget
	if (mode == 2){
		model = greedy_exact(ep);
		noNoise();
	}

	// Build a naive bayes network conditioned on the argument of 'naive()'
	if (mode == 3){
		model = naive();
		addnoise(ep);

	}

	// PML with uniform prior
	if (mode == 4){
		model = pml_select(ep, 1);
	}

	// PML with real prior
	if (mode == 5){
		model = pml_select(ep, 2);
	}

	sampling(round(tbl.size()*0.33), filename+".csv");
	//print_model_txt(filename+".txt");
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
	set<int> S;
	set<int> V = tools::setize(dim);
	int picked_index;
	syn.initialize(tbl);
	double budget = ep/dim;
	//cout << "PML Sel and Noise: " << ep << endl;
	//cout << "budget: " << budget << endl;

	for (int t = 0; t < dim; t++) {
		// Pick candidate sets
		vector<dependence> deps = S2V(S, V);
		vector<double> quality;
		vector<double> counts;
		
		// If t=0 (first selection), choose the first attribute randomly (do the same procedure as the original code)
		if (t==0) {
			for (const auto& dep : deps) {
				quality.push_back(tbl.getScore(dep));
			}
			picked_index = noise::EM(eng, quality, 1000.0, sens);
			// get the noisy counts of the chosen dependence
			dependence dp = deps[picked_index];
			counts = tbl.getCounts(dp.cols, dp.lvls);
			double sumCounts = accumulate(counts.begin(), counts.end(), 0.0);
			vector<double> noisy_counts = pml_noise(counts, sumCounts, prior_mode, budget);
			model_counts.push_back(noisy_counts);
			syn.margins[dp.cols].counts[dp.lvls] = noisy_counts;

		} else {
			// For the other selections, get the histogram dist and add noise based on PML
			// Then get the mutual information from the noisy distributions
			// Select the dependence based on the noisy mutual information
			vector<double> quality;
			vector<vector<double>> noisy_distributions;
			vector<int> widths;

			for (const auto& dep : deps) {
				// get the histogram
				counts = tbl.getCounts(dep.cols, dep.lvls);
				// cout << "Counts: ";
				// for (double c : counts) {cout << c << " ";}
				// cout << endl;
				double sumCounts = accumulate(counts.begin(), counts.end(), 0.0);
				widths = tbl.getWidth(dep.cols, dep.lvls);
				
				vector<double> noisy_counts = pml_noise(counts, sumCounts, prior_mode, budget);
				// cout << "Noisy: ";
				// for (double c : noisy_counts) {cout << c << " ";}
				// cout << endl;
				// normalize the noisy counts vector so that the total counts are equal
				// (pdf normalization)
				//double sumNoisy = accumulate(noisy_counts.begin(), noisy_counts.end(), 0.0);
				//for (int ind = 0; ind< noisy_counts.size(); ind++) {
				//	noisy_counts[ind] = (sumCounts * noisy_counts[ind]); / sumNoisy;
				//}
				// cout << "Noisy Normalized: ";
				// for (double c : noisy_counts) {cout << c << " ";}
				// cout << endl;

				// store the noisy distributions for each dependence,
				// we want to return the counts for picked ones
			    noisy_distributions.push_back(noisy_counts);
				
				// now, convert the noisy counts to mutual information
				
			//	cout << tbl.getI(noisy_counts, widths)[dep.ptr] << " ";//db
				quality.push_back(tbl.getI(noisy_counts, widths)[dep.ptr]);
			}
			// find the index of largest MI
			picked_index = distance(quality.begin(), max_element(quality.begin(), quality.end()));
			dependence dp = deps[picked_index];
			syn.margins[dp.cols].counts[dp.lvls] = noisy_distributions[picked_index];
			model_counts.push_back(noisy_distributions[picked_index]);
		}

		dependence picked = deps[picked_index];
		
		S.insert(picked.x.first);								
		V.erase(picked.x.first);
		model.push_back(picked);
	//	cout << to_string(picked) << endl;
		// we want to return the selected dependencies and their noisy distributions
	}
	//for (auto& dep : model) {cout << dep.x.first << endl;}
	return model;
}

// add PML noise based on the mode
vector<double> bayesian::pml_noise(vector<double> counts, double sumCounts, int prior_mode, double budget) {
	double p_min;

	if (prior_mode==1) { // uniform
		p_min = 1.0 / counts.size();
//	    cout << "Uniform prior: " << p_min << " ";//db
	} else if (prior_mode==2) { // prior from data
		// create a copy of count vector and erase the 0 values
		vector<double> count_no_zeros;
		count_no_zeros = counts;
		count_no_zeros.erase(
			remove(count_no_zeros.begin(), count_no_zeros.end(), 0.0),
			count_no_zeros.end()
		);
		// find non-zero minimum count value
		vector<double>::iterator minCount = min_element(count_no_zeros.begin(), count_no_zeros.end()); 
		
		p_min = *minCount / sumCounts;
//		cout << "Data prior: " << p_min;//db
	}
	// cout << "p_min: " << p_min << endl;
	vector<double> noisy_counts;
	// check if budget < -log(p_min), it is a PML property 
	if (budget <= -log(p_min)) {
		// we add noise
		// write the noise term in terms of p_min
		double noise_scale = 2/(budget+log((1-p_min)/(1-p_min*exp(budget))));
		// cout << "DP noise: " << 2/budget << endl;
		// cout << "Pml noise: " << noise_scale << endl;
		for (double count: counts) {
			double noisy_count = count + noise::nextLaplace(eng, noise_scale);
			// equalize the values smaller than 0 to zero.
			if (noisy_count < 0) {noisy_count = 0;}
			noisy_counts.push_back(noisy_count);
		} 
	} else {
		// no noise
		noisy_counts = counts;
	}
	return noisy_counts;
}

// Learn a naive bayesian network (for free)
vector<dependence> bayesian::naive() {
	vector<dependence> model;
	set<int> S;
	set<int> V = tools::setize(dim);
	double sens = tbl.sens;
	vector<dependence> deps = S2V(S, V);		
	vector<double> quality;
	for (const auto& dep : deps)
		quality.push_back(tbl.getScore(dep));
	// Add the root to the naive bayes model. Root should be conditioned on Null
	int root_index = noise::EM(eng, quality, 1000.0, sens);
	dependence root = deps[root_index];
	int root_id = stoi(to_string(root.x));
	model.push_back(deps[root_index]);

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
	int picked_index;

	set<int> S;
	set<int> V = tools::setize(dim);

	for (int t = 0; t < dim; t++) {
		vector<dependence> deps = S2V(S, V);		
		vector<double> quality;
		for (const auto& dep : deps) 
			quality.push_back(tbl.getScore(dep));

		if (t==0) {
			// choose root randomly
			picked_index = noise::EM(eng, quality, 1000.0, sens);
		} else {
			// choose parent-node pairs based on max quality
			picked_index = max_element(quality.begin(), quality.end()) - quality.begin();
		}

		//cout << to_string(picked) << " exact " << endl;
		dependence picked = deps[picked_index];
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
	//	cout << k;
		vector<vector<attribute>> parents = maximal(S, k);
		// maximal(S, bound / tbl.getWidth(x));

		for (const vector<attribute>& p : parents)
			if (exist.find(p) == exist.end()) {
				exist.insert(p);
				ans.push_back(dependence(p, attribute(x, 0)));
			}
		if (exist.empty()) ans.push_back(dependence(vector<attribute>(), attribute(x, 0)));
	}
	return ans;
}

// vector<vector<attribute>> bayesian::maximal(set<int> S, double tau) {
// 	vector<vector<attribute>> ans;
// 	cout << "Tau: " << tau << endl;
// 	if (tau < 1) return ans;
// 	if (S.empty()) {
// 		ans.push_back(vector<attribute>());
// 		return ans;
// 	}

// 	int last = *(--S.end());
// 	S.erase(--S.end());
// 	int depth = tbl.getDepth(last);
// 	set<vector<attribute>> exist;

// 	// with 'last' at a certain level
// 	for (int l = 0; l < depth; l++) {
// 		attribute att(last, l);
// 		vector<vector<attribute>> maxs = maximal(S, tau / tbl.getWidth(att));
// 		cout << "Tau/width: " << tau / tbl.getWidth(att) << endl;
// 		for (vector<attribute> z : maxs)
// 			if (exist.find(z) == exist.end()) {
// 				exist.insert(z);
// 				z.push_back(att);
// 				ans.push_back(z);
// 			}
// 	}

// 	// without 'last'
// 	vector<vector<attribute>> maxs = maximal(S, tau);
// 	for (vector<attribute> z : maxs)
// 		if (exist.find(z) == exist.end()) {
// 			exist.insert(z);
// 			ans.push_back(z);
// 		}

// 	return ans;
// }


vector<vector<attribute>> bayesian::maximal(set<int> S, int k) {
    vector<vector<attribute>> ans;
    // Check if k is less than or equal to 0
    if (k <= 0) return ans;
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
        vector<vector<attribute>> maxs = maximal(S, k - 1); // Decrease k by 1
        for (vector<attribute> z : maxs) {
            if (z.size() < k) { // Check if number of parents is less than k
                exist.insert(z);
                z.push_back(att);
                ans.push_back(z);
            }
        }
    }

    // without 'last'
    vector<vector<attribute>> maxs = maximal(S, k); // Recursive call without adding last
    for (vector<attribute> z : maxs)
        if (z.size() < k && exist.find(z) == exist.end()) { // Check if number of parents is less than k and not already present
            exist.insert(z);
            ans.push_back(z);
        }

    return ans;
}


void bayesian::addnoise(double ep) {
	syn.initialize(tbl);
	for (const dependence& dep : model) {
		vector<double>& counts_syn = syn.margins[dep.cols].counts[dep.lvls];
		for (double count : tbl.getCounts(dep.cols, dep.lvls)) {
			//double res = count + noise::nextLaplace(eng, 2.0 * dim / ep); // debug
			//if (res < 0) {res=0;} //debug
			counts_syn.push_back(count + noise::nextLaplace(eng, 2.0 * dim / ep));
		}
		model_counts.push_back(counts_syn);
	}

}

void bayesian::noNoise(){
	syn.initialize(tbl);
	for (const dependence& dep : model) {
		vector<double>& counts_syn = syn.margins[dep.cols].counts[dep.lvls];
		for (double count : tbl.getCounts(dep.cols, dep.lvls))
			counts_syn.push_back(count);
		model_counts.push_back(counts_syn);
	}
}

void bayesian::sampling(int num, string filename) {
	ofstream csv_file;
	csv_file.open(filename);

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
		csv_file << i;
		for (int val : tuple) {csv_file << "," << val;}
		csv_file << "\n";
	}
	csv_file.close();

	syn.margins.clear();
}

string bayesian::print_model(){
	string ans;
	int i = 0;
	for (const dependence& dep : model) {
		ans += to_string(dep) + "\n";
		for (double c : model_counts[i]) {
			ans += std::to_string(c) + " ";
		}
		ans += "\n";
		i++;
	}
	return ans;
}

void bayesian::print_model_txt(string filename){
	ofstream txt_file;
	txt_file.open(filename);
	int i = 0;
	for (const dependence& dep : model) {
		txt_file << to_string(dep) << "\n";
		for (double c : model_counts[i]) {
			txt_file << std::to_string(c) << " ";
		}
		txt_file << "\n";
		i++;
	}
	txt_file.close();
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
