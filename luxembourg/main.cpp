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
T var(Dataset<internal_t,T,internal_t> const &D) {
	T mean = 0;
	size_t cnt = 0;
	for (size_t i = 0; i < D.size(); ++i) {
		mean += D[i].label;
		++cnt;
	}
	mean /= cnt;

	T sum2 = 0;
	for (size_t i = 0; i < D.size(); ++i) {
		sum2 += (D[i].label-mean)*(D[i].label-mean);
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
		Dataset<internal_t,internal_t,internal_t> DTrain(splits, i);

		bool fitted = model->fit(DTrain);
		if (!fitted) {
			error[i] = std::numeric_limits<internal_t>::quiet_NaN();
		} else {
			internal_t variance = var(splits[i]);

			error[i] = model->error(splits[i], [variance](std::vector<internal_t> const &pred, internal_t label) -> internal_t {
				return (pred[0]-label)*(pred[0]-label)/variance;
			});
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

void readCSV(std::string const &path, std::vector<size_t> &coords, std::vector<internal_t> &density) {
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
					coords.push_back(static_cast<size_t>(static_cast<float>(atof(entry.c_str()))*100.0f));
				} else if (i == 2) {
					density.push_back(static_cast<internal_t>(atof(entry.c_str())));
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
	std::vector<size_t> coords;
	std::vector<internal_t> density;
    auto pathToData = "../data.csv";
	readCSV(pathToData, coords, density);

	size_t dim = 2;

    std::cout << "=== DATA READ ===" << std::endl;
//	std::cout << "RAW N = " << counts.size() << std::endl;
//	std::cout << "dim = " << dim << std::endl;
//	for (unsigned int i = 0;i < counts.size(); ++i) {
//		for (unsigned int j = 0; j < dim; ++j) {
//			std::cout << coords[i*dim+j] << " ";
//		}
//		std::cout << "- " << counts[i] << std::endl;
//	}
//    std::cout << "======" << std::endl << std::endl;
	std::map<size_t, std::map<size_t, std::vector<internal_t>>> coord_map;

	for (size_t i = 0; i < density.size(); ++i) {
		size_t c1 = coords[i*dim + 0];
		size_t c2 = coords[i*dim + 1];
        if (!coord_map.count(c1)) {
			coord_map[c1] = std::map<size_t, std::vector<internal_t>>();
        }
        if (!coord_map[c1].count(c2)) {
			coord_map[c1][c2] = std::vector<internal_t>();
        }
		coord_map[c1][c2].push_back(density[i]);
    }

	std::vector<internal_t> X;
	std::vector<internal_t> Y;

    for (auto& [c1, inner_map]: coord_map) {
        for (auto& [c2, value]: inner_map) {
			X.push_back(c1);
			X.push_back(c2);
            
			internal_t mean = 0;
            for (auto e : value) {
                mean += e;
            }
			Y.push_back(mean / value.size());
			//std::cout << "value.size() = " << value.size() << std::endl;
        }
    }

	size_t N = Y.size();
//	std::cout << "Y.size() = " << Y.size() << std::endl;
//	std::cout << "X.size() = " << X.size() << std::endl;
//	std::cout << "=== DATA PROCESSED ===" << std::endl;
//	for (unsigned int i = 0;i < N; ++i) {
//		for (unsigned int j = 0; j < dim; ++j) {
//			std::cout << X[i*dim+j] << " ";
//		}
//		std::cout << "- " << Y[i] << std::endl;
//	}
//	std::cout << "======" << std::endl << std::endl;

	normalize<internal_t>(&X[0], N, dim);

	std::cout << "=== DATA NORMALIZED ===" << std::endl;
	for (unsigned int i = 0;i < N; ++i) {
		for (unsigned int j = 0; j < dim; ++j) {
			std::cout << X[i*dim+j] << " ";
		}
		std::cout << "- " << Y[i] << std::endl;
	}
    std::cout << "======" << std::endl << std::endl;

	Dataset<internal_t, internal_t, internal_t> D(&X[0], &Y[0], N, dim);
	bool print_header = true;
	const size_t xval_runs = 5;

	for (auto eps : {0.01, 0.05, 0.1, 0.5}) {
		for (auto l1 : {0.5,1.0,2.0,5.0}) {
			for (auto l2 : {0.5, 1.0,2.0,5.0}) {
				internal_t kparam[2] = {static_cast<internal_t>(l1),static_cast<internal_t>(l2)};
				ARDKernel<internal_t,internal_t> k(kparam, dim);
				//DotKernel<internal_t, internal_t> k;
//				for (auto p : {200, 500, 1000}) {
//					testModel(
//						[p,eps,&k]() {return new GaussianProcess<internal_t, internal_t, internal_t>(p, eps, k);},
//						D,
//						"GP(" + std::to_string(p) + "," + std::to_string(eps) + "," + std::to_string(l1) + "," + std::to_string(l2) + ")",
//						print_header,
//						xval_runs
//					);
//					print_header = false;
//				}
//				for (auto p : {50, 100, 200}) {
//					testModel(
//						[p,eps,&k]() -> BatchLearner<internal_t, internal_t, internal_t>* {return new InformativeVectorMachine<internal_t, internal_t, internal_t>(p, eps, k);},
//						D,
//						"IVM(" + std::to_string(p) + "," + std::to_string(eps) + "," + std::to_string(l1) + "," + std::to_string(l2) + ")",
//						print_header,
//						xval_runs
//					);
//				}
				for (auto split_points: {50, 100, 200}) {
					for (auto gp_points : {500, 1000}) {
						testModel(
							[split_points, gp_points, eps, &k]() -> BatchLearner<internal_t, internal_t, internal_t>* {return new GaussianModelTree<internal_t, internal_t, internal_t>(split_points, gp_points, 0, eps, k);},
							D,
							"GMT(" + std::to_string(split_points) + "," + std::to_string(gp_points) + "," + std::to_string(eps) + "," + std::to_string(l1) + "," + std::to_string(l2) + ")",
							print_header,
							xval_runs
						);
					}
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
