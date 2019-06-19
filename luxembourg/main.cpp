#include <algorithm>
#include <cassert>
#include <csignal>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <vector>

#include "GaussianProcess.h"
#include "InformativeVectorMachine.h"
#include "GaussianModelTree.h"

using internal_t = float;

template <typename T>
T var(std::vector<Dataset<internal_t,T,internal_t>> const &splits, size_t fold) {
	T mean = 0;
	size_t cnt = 0;
	for (size_t j = 0; j < splits.size(); ++j) {
		if (fold != j) {
			for (size_t i = 0; i < splits[j].size(); ++i) {
				mean += splits[j][i].label;
				++cnt;
			}
		}
	}
	mean /= cnt;

	T sum2 = 0;
	for (size_t j = 0; j < splits.size(); ++j) {
		if (fold != j) {
			for (size_t i = 0; i < splits[j].size(); ++i) {
				sum2 += (splits[j][i].label-mean)*(splits[j][i].label-mean);
			}
		}
	}

	return sum2/cnt;
}

void testModel(
		std::function<BatchLearner<internal_t, internal_t, internal_t> *()> createModel,
		Dataset<internal_t, internal_t, internal_t> D, std::string const &model_name,
		bool with_header = false, size_t folds = 5) {
	std::vector<Dataset<internal_t,internal_t,internal_t>> splits = D.k_fold(folds);

	std::vector<internal_t> error(folds, 0.0);
	for (size_t i = 0; i < folds; ++i) {
		BatchLearner<internal_t, internal_t, internal_t> *model = createModel();
		bool fitted = model->fit(splits[i]);
		if (!fitted) {
			error[i] = std::numeric_limits<internal_t>::quiet_NaN();
		} else {
			internal_t variance = var(splits, i);

			for (size_t j = 0; j < folds; ++j) {
				if (i != j) {
					error[i] += model->error(splits[j], [variance](std::vector<internal_t> const &pred, internal_t label) -> internal_t {
						return (pred[0]-label)*(pred[0]-label)/variance;
					});
				}
			}
			error[i] /= (folds - 1);
			delete model;
		}
	}

	internal_t smse = std::accumulate(error.begin(), error.end(), 0.0) / error.size();
    
	internal_t xval_var = 0.0;
	std::for_each (error.begin(), error.end(), [&](const internal_t e) {
        xval_var += (e - smse) * (e - smse);
    });
    xval_var /= error.size();

	std::stringstream ss;
	ss << "\"" << model_name << "\"";
	for (size_t i = 0; i < folds; ++i) {
		ss << Logger::sep << error[i];
	}

	if (with_header) {
		std::stringstream header;
		header << "model_name,";
		for (size_t i = 0; i < folds; ++i) {
			header << Logger::sep << "SMSE_" << i;
		}
		Logger::log("xval", header.str());
	}

	Logger::log("xval", ss.str());
	std::cout << model_name << " SMSE is: " << smse << " +/-" << xval_var << std::endl;
//	std::cout << "SMSE is: " << error / (static_cast<internal_t>(NTest)*var<internal_t>(YTest,NTest)) << std::endl << std::endl;
}

void readCSV(std::string const &path, std::vector<internal_t> &X, std::vector<internal_t> &Y) {
	if (!file_exists(path)) {
		throw std::runtime_error("File does not exist: " + path);
	}
	std::ifstream file(path);
    
    std::string header;
    getline(file, header);

	while (!file.eof()) {
		std::string line;

		while (getline(file, line)) {
			std::stringstream ss(line);
			std::string entry;
			size_t i = 0;

			while (getline(ss, entry, ',')) {
				if (i == 0 || i == 1) {
					X.push_back(static_cast<internal_t>(atof(entry.c_str()))*100.0f);
				} else if (i == 2) {
					Y.push_back(static_cast<internal_t>(atof(entry.c_str()))*100.0f);
				} else {
					break;
				}
				++i;
			}
		}
	}
	file.close();
}

int main(int argc, char const* argv[]) {
    std::cout << "Reading files" << std::endl;
	std::vector<internal_t> XVec;
	std::vector<internal_t> YVec;
    auto pathToData = "../data.csv";
	readCSV(pathToData, XVec, YVec);

	size_t N = YVec.size();
	size_t dim = 2;
	internal_t * X = &XVec[0];
	internal_t * Y = &YVec[0];

    std::cout << "=== DATA READ ===" << std::endl;
	std::cout << "N = " << N << std::endl;
	std::cout << "dim = " << dim << std::endl;
	for (unsigned int i = 0;i < 10; ++i) {
		for (unsigned int j = 0; j < dim; ++j) {
			std::cout << X[i*dim+j] << " ";
		}
		std::cout << "- " << Y[i] << std::endl;
	}
    std::cout << "======" << std::endl << std::endl;

	normalize<internal_t>(X, N, dim);
	std::map<internal_t, std::map<internal_t, std::vector<internal_t>>> coord_map;

    for (size_t i = 0; i < N; ++i) {
		internal_t c1 = X[i*dim + 0];
		internal_t c2 = X[i*dim + 1];
        if (!coord_map.count(c1)) {
			coord_map[c1] = std::map<internal_t, std::vector<internal_t>>();
        }
        if (!coord_map[c1].count(c2)) {
			coord_map[c1][c2] = std::vector<internal_t>();
        }
        coord_map[c1][c2].push_back(Y[i]);
    }

	std::vector<internal_t> _X;
	std::vector<internal_t> _Y;

    for (auto& [c1, inner_map]: coord_map) {
        for (auto& [c2, value]: inner_map) {
            _X.push_back(c1);
            _X.push_back(c2);
            
			internal_t mean = 0;
            for (auto e : value) {
                mean += e;
            }
            _Y.push_back(mean / value.size());
        }
    }

    std::cout << "=== DATA AGGREGATED ===" << std::endl;
	std::cout << "N = " << _Y.size() << std::endl;
	std::cout << "dim = " << dim << std::endl;
	for (unsigned int i = 0;i < 10; ++i) {
		for (unsigned int j = 0; j < dim; ++j) {
			std::cout << _X[i*dim+j] << " ";
		}
		std::cout << "- " << _Y[i] << std::endl;
	}
    std::cout << "======" << std::endl << std::endl;

	Dataset<internal_t, internal_t, internal_t> D(&_X[0], &_Y[0], _Y.size(), dim);
	bool print_header = true;

	for (auto eps : {0.01, 0.05, 0.1, 0.5}) {
		for (auto l1 : {1.0,2.0,5.0}) {
			for (auto l2 : {1.0,2.0,5.0}) {
				internal_t kparam[2] = {l1,l2};
				ARDKernel<internal_t,internal_t> k(kparam, dim);
				//DotKernel<internal_t, internal_t> k;
				for (auto p : {200,500,1000,2000}) {
					testModel(
						[p,eps,&k]() {return new GaussianProcess<internal_t, internal_t, internal_t>(p, eps, k);},
						D,
						"GP(" + std::to_string(p) + "," + std::to_string(eps) + "," + std::to_string(l1) + "," + std::to_string(l2) + ")",
						print_header,
						5
					);
					print_header = false;
				}
				for (auto p : {200,500}) {
					testModel(
						[p,eps,&k]() -> BatchLearner<internal_t, internal_t, internal_t>* {return new InformativeVectorMachine<internal_t, internal_t, internal_t>(p, eps, k);},
						D,
						"IVM(" + std::to_string(p) + "," + std::to_string(eps) + "," + std::to_string(l1) + "," + std::to_string(l2) + ")",
						print_header,
						5
					);
	//				testModel(
	//					[p,&k]() -> BatchLearner<internal_t, internal_t, internal_t>* {return new GaussianModelTree<internal_t, internal_t, internal_t>(50, p, 0, 0.1, k);},
	//					D,
	//					"GMT(" + std::to_string(l1) + "," + std::to_string(l2) + "," + std::to_string(p) + ")",
	//					print_header,
	//					5
	//				);
				}
	        }
	    }
    }

    /*
    std::vector<std::vector<unsigned int> > train;
    std::vector<std::vector<unsigned int> > test;

    unsigned int const XVAL = 5;
    xval(XVAL, X.size(), train, test);
    std::vector<internal_t> epsilons = { 1e-5, 0.001, 0.1, 1, 2, 5 };
    std::vector<unsigned int> Ks = { 50, 200 };
    std::vector<unsigned int> optIter = { 0, 25 };

    std::ofstream statistics("../experiments/luxembourg/luxembourg.csv");
    statistics << "xval-run," << ExperimentConfiguration::csvHeader() << ","
               << Experiment::csvHeader() << ",filename" << std::endl;

    for (auto K : Ks) {
        for (auto epsilon : epsilons) {
            for (auto iter : optIter) {
                internal_t avgSMSE = 0.0f;
                internal_t avgRMSE = 0.0f;
                for (unsigned int i = 0; i < XVAL; ++i) {
                    std::vector<std::vector<internal_t> > XTrain;
                    std::vector<internal_t> YTrain;
                    std::vector<std::vector<internal_t> > XTest;
                    std::vector<internal_t> YTest;

                    for (auto j : train[i]) {
                        XTrain.push_back(X[j]);
                        YTrain.push_back(Y[j]);
                    }

                    for (auto j : test[i]) {
                        XTest.push_back(X[j]);
                        YTest.push_back(Y[j]);
                    }
                    std::vector<internal_t> min;
                    std::vector<internal_t> max;
                    minMaxData(XTrain, min, max);
                    normalizeData(XTrain, min, max);
                    normalizeData(XTest, min, max);

                    std::cout << "XTrain.size(): " << XTrain.size() << std::endl;
                    std::cout << "XTest.size(): " << XTest.size() << std::endl;
                    std::cout << "TRAIN mean: " << mean(YTrain) << std::endl;
                    std::cout << "TRAIN variance: " << variance(YTrain) << std::endl;
                    std::cout << "TEST mean: " << mean(YTest) << std::endl;
                    std::cout << "TEST variance: " << variance(YTest) << std::endl;

                    auto fRegion = "XVAL_" + std::to_string(i) + "K" + std::to_string(K) + "_e"
                        + std::to_string(epsilon) + "_iter" + std::to_string(iter) + ".csv";
                    std::ofstream region("../experiments/luxembourg/" + fRegion);
                    ExperimentConfiguration config(XTrain, YTrain, XTest, YTest);
                    config.K = K;
                    config.noise = 0.05;
                    config.epsilon = epsilon;
                    config.numIter = iter;
                    Experiment exp = performExperiment(config);
                    statistics << i << "," << config.csv() << "," << exp.csv() << "," << fRegion
                               << std::endl;
                    region << Experiment::csvRegionHeader() << std::endl;
                    region << exp.csvRegions();
                    avgRMSE += exp.rmse;
                    avgSMSE += exp.smse;
                    getch();
                    internal_t mseBaseline = 0.0;
                    internal_t baseline = mean(YTrain);

                    // For reference
                    for (unsigned int i = 0; i < XTest.size(); ++i) {
                        mseBaseline += (baseline - YTest[i]) * (baseline - YTest[i]);
                    }

                    std::cout << "SMSE-BASELINE: " << (mseBaseline / XTest.size()) / variance(YTest)
                              << std::endl;
                    std::cout << "RMSE-BASELINE: " << std::sqrt(mseBaseline / XTest.size())
                              << std::endl;
                }
                std::cout << "XVAL DONE" << std::endl;
                std::cout << "RMSE: " << avgRMSE / XVAL << std::endl;
                std::cout << "SMSE: " << avgSMSE / XVAL << std::endl;
                std::cout << std::endl;
                // getch();
            }
        }
    } */

}
