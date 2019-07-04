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
#include <chrono>

#include "GaussianProcess.h"
#include "InformativeVectorMachine.h"
#include "GaussianModelTree.h"
#include "DecisionStump.h"

using internal_t = float;

#ifdef USE_TORCH
#include "TorchWrapper.h"

class Net : public TorchNet<internal_t, internal_t, internal_t> {
public:
	Net(size_t dim)
		: fc1(dim, 20), fc2(20, 20), fc3(20, 1) {
		register_module("fc1", fc1);
		register_module("fc2", fc2);
		register_module("fc3", fc3);
	}

	size_t size() const {
		return 0;
	}

	std::string str() const {
		return "FCNN(dim->20->20->1)";
	}

	torch::Tensor predict(torch::Tensor x) {
		x = torch::relu(fc1->forward(x));
		x = torch::relu(fc2->forward(x));
		x = fc3->forward(x);
		return x;
	}

	bool fit(TorchDataset<internal_t, internal_t, internal_t> &D) {
		torch::Device device(torch::kCPU);
		this->to(device);
		float const lr = 0.01f;
		float const mom = 0.0f;
		unsigned int const bSize = 32;
		torch::optim::SGD optimizer(parameters(), torch::optim::SGDOptions(lr).momentum(mom));
		auto tmp = D.map(torch::data::transforms::Stack<>());
		auto train_loader =
				torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(tmp), bSize);
		for (size_t i = 0; i < 20; ++i) {
			float mean_loss = 0;
			size_t batch_cnt = 0;
			for (auto &batch : *train_loader) {
				auto examples = batch.data.to(device);
				auto targets = batch.target.slice(1, 0, 1, 1); // dim, start, end, step
				auto weights = batch.target.slice(1, 1, 2, 1);
//				std::cout << "original_targets = " << batch.target << std::endl;
				//std::cout << "targets = " << targets << std::endl;
				//std::cout << "examples = " << examples << std::endl;
//				std::cout << "weights = " << weights << std::endl;
				try {
					//batch.data()
					targets = targets.to(device);
					weights = weights .to(device);

					optimizer.zero_grad();
					//std::cout << data << std::endl;
					auto output = predict(examples);
					//std::cout << "output = " << output << std::endl;
					//auto loss = torch::binary_cross_entropy(output, targets);
					auto loss = torch::mse_loss(output, targets);
					//loss = (loss * weights  / weights .sum()).sum();
					//loss = loss.mean();

					loss.backward(); // TODO MAYBE NORMALIZE HERE
					//std::cout << "loss = " << loss << std::endl;
					mean_loss += loss.template item<float>();
					batch_cnt++;
					optimizer.step();
				} catch (std::runtime_error &e) {
					std::cout << "error was " << e.what() << std::endl;
					std::cout << "targets " << targets << std::endl;
					std::cout << "weights " << weights << std::endl;
					return false;
				}
			}
//			std::cout << "loss: " << mean_loss / batch_cnt << std::endl;
		}
		return true;
	}

	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;
};


#endif

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

void test_model(
		std::function<BatchLearner<internal_t, internal_t, internal_t> *()> createModel,
		Dataset<internal_t, internal_t, internal_t> D,
		std::vector<std::pair<std::string, std::string>> const &params,
		bool with_header = false,
		size_t folds = 5) {
	std::vector<Dataset<internal_t,internal_t,internal_t>> splits = D.k_fold(folds);

	std::vector<internal_t> mse_error(folds, 0.0);
	std::vector<internal_t> smse_error(folds, 0.0);
	std::vector<internal_t> mae_error(folds, 0.0);
	std::vector<long> test_time(folds);
	std::vector<long> fit_time(folds);

	std::chrono::high_resolution_clock::time_point start, end;

#pragma omp parallel for
	for (size_t i = 0; i < folds; ++i) {
		BatchLearner<internal_t, internal_t, internal_t> *model = createModel();
		Dataset<internal_t,internal_t,internal_t> DTrain(splits, i);

		start = std::chrono::high_resolution_clock::now();
		bool fitted = model->fit(DTrain);
		end = std::chrono::high_resolution_clock::now();
		auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		fit_time[i] = milliseconds.count();

		start = std::chrono::high_resolution_clock::now();
		if (!fitted) {
			mae_error[i] = std::numeric_limits<internal_t>::quiet_NaN();
			mse_error[i] = std::numeric_limits<internal_t>::quiet_NaN();
			smse_error[i] = std::numeric_limits<internal_t>::quiet_NaN();
		} else {
			mae_error[i] = model->error(splits[i], [](std::vector<internal_t> const &pred, internal_t label) -> internal_t {
				return std::abs(pred[0]-label);
			});
			mse_error[i] = model->error(splits[i], [](std::vector<internal_t> const &pred, internal_t label) -> internal_t {
				return (pred[0]-label)*(pred[0]-label);
			});
			smse_error[i] = mse_error[i]/var(splits[i]);
			delete model;
		}
		end = std::chrono::high_resolution_clock::now();
		milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		test_time[i] = milliseconds.count();
	}

	internal_t smse = std::accumulate(smse_error.begin(), smse_error.end(), 0.0) / smse_error.size();

	/*internal_t xval_var = 0.0;
	std::for_each (smse_error.begin(), smse_error.end(), [&](const internal_t e) {
		xval_var += (e - smse) * (e - smse);
	});
	xval_var /= smse_error.size();*/

	if (with_header) {
		std::stringstream header;
		for (auto &p : params) {
			header << "\"" << p.first  << "\"" << Logger::sep;
		}
		for (size_t i = 0; i < folds; ++i) {
			header << "MAE_" << i << Logger::sep;
		}
		for (size_t i = 0; i < folds; ++i) {
			header << "MSE_" << i << Logger::sep;
		}
		for (size_t i = 0; i < folds; ++i) {
			header << "SMSE_" << i << Logger::sep;
		}
		for (size_t i = 0; i < folds; ++i) {
			header << "fit_time" << i << Logger::sep;
		}
		for (size_t i = 0; i < folds; ++i) {
			header << "testtime_" << i;

			if (i != folds - 1) {
				header << Logger::sep;
			}
		}
		std::cout << header.str() << std::endl;
		Logger::log("xval", header.str());
	}

	std::stringstream ss;
	for (auto &p : params) {
		ss << "\"" << p.second  << "\"" << Logger::sep;
	}
	for (size_t i = 0; i < folds; ++i) {
		ss << mae_error[i] << Logger::sep;
	}
	for (size_t i = 0; i < folds; ++i) {
		ss << mse_error[i] << Logger::sep;
	}
	for (size_t i = 0; i < folds; ++i) {
		ss << smse_error[i] << Logger::sep;
	}
	for (size_t i = 0; i < folds; ++i) {
		ss << fit_time[i] << Logger::sep;
	}
	for (size_t i = 0; i < folds; ++i) {
		ss << test_time[i];

		if (i != folds - 1) {
			ss << Logger::sep;
		}
	}
	Logger::log("xval", ss.str());
	std::cout << ss.str() << std::endl;
	//"\t " << smse << " +/-" << xval_var << std::endl;
//	std::cout << model_name << " SMSE is: " << smse << " +/-" << xval_var << std::endl;
//	std::cout << "SMSE is: " << error / (static_cast<internal_t>(NTest)*var<internal_t>(YTest,NTest)) << std::endl << std::endl;
}

size_t read_csv(std::string const &path, std::vector<internal_t> &X, std::vector<internal_t> &Y) {
	std::string line;
	std::ifstream file(path);
	if (!file_exists(path)) {
		throw std::runtime_error("File not found " + path);
	}
	std::string header;
	std::getline(file, header);

	size_t dim = 0;
	if (file.is_open()) {
		while (std::getline(file, line)) {
			if (line.size() > 0) {
				std::stringstream ss(line);
				std::string entry;

				std::vector<std::string> tmp;
				while (std::getline(ss, entry, ',')) {
					if (entry.size() > 0) {
						tmp.push_back(entry);
					}
				}
				dim = tmp.size() - 4;
				for (size_t i = 0; i < tmp.size(); ++i) {
					if (i == 0 || i == 5 || i == 6) continue;
					if (i == 7) {
						//Y.push_back(std::log(static_cast<internal_t>(std::stof(tmp[i]))));
						Y.push_back(static_cast<internal_t>(std::stof(tmp[i])));
					} else {
						X.push_back(static_cast<internal_t>(std::stof(tmp[i])));
					}
				}
			}
		}
		file.close();
	}
	return dim;
}

void test_all_models(Dataset<internal_t, internal_t, internal_t> D,
					 Kernel<internal_t,internal_t> &k, std::string const &kname, std::string const &k_l1, std::string const &k_l2,
					 bool with_header = false,
					 size_t folds = 5) {

	for (auto eps : {0.1}) {
		for (auto p : {500, 1000, 5000, 10000}) {
			test_model(
				[p,eps,&k]() {return new GaussianProcess<internal_t, internal_t, internal_t>(p, eps, k);},
				D,
				{
					std::make_pair("name","GP"),
					std::make_pair("kernel",kname),
					std::make_pair("eps",std::to_string(eps)),
					std::make_pair("k_l1",k_l1),
					std::make_pair("k_l2",k_l2),
					std::make_pair("gp_points",std::to_string(p)),
					std::make_pair("ivm_points","None"),
				},
				with_header,
				folds
			);
			with_header = false;
		}

		for (auto p : {50, 100, 200}) {
			test_model(
				[p,eps,&k]() -> BatchLearner<internal_t, internal_t, internal_t>* {return new InformativeVectorMachine<internal_t, internal_t, internal_t>(p, eps, k);},
				D,
				{
					std::make_pair("name","IVM"),
					std::make_pair("kernel",kname),
					std::make_pair("eps",std::to_string(eps)),
					std::make_pair("k_l1",k_l1),
					std::make_pair("k_l2",k_l2),
					std::make_pair("gp_points","None"),
					std::make_pair("ivm_points",std::to_string(p)),
				},
				with_header,
				folds
			);
		}

		for (auto split_points: {50, 100}) {
			for (auto gp_points : {500, 1000, 5000, 10000}) {
				test_model(
					[split_points, gp_points, eps, &k]() -> BatchLearner<internal_t, internal_t, internal_t>* {return new GaussianModelTree<internal_t, internal_t, internal_t>(split_points, gp_points, 0, eps, k);},
					D,
					{
						std::make_pair("name","GMT"),
						std::make_pair("kernel",kname),
						std::make_pair("eps",std::to_string(eps)),
						std::make_pair("k_l1",k_l1),
						std::make_pair("k_l2",k_l2),
						std::make_pair("gp_points",std::to_string(gp_points)),
						std::make_pair("ivm_points",std::to_string(split_points)),
					},
					with_header,
					folds
				);
				test_model(
					[split_points, gp_points, eps, &k, &D]() -> BatchLearner<internal_t, internal_t, internal_t>* {
						return new ModelTree<internal_t, internal_t, internal_t>(
							[&D](size_t h) -> BatchLearner<internal_t, internal_t, internal_t>* {
								return new TorchWrapper(new Net(D.dimension()));
							},
							[split_points, eps, &k](
									unsigned int height) -> Splitter<internal_t, internal_t, internal_t> * { //-> DTSplitter<FT,LABEL_TYPE::BINARY>*
								return new IVMSplitter<internal_t, internal_t, internal_t>(split_points, eps, k);
							},
							gp_points,
							0);
					},
					D,
					{
						std::make_pair("name","GMT-NN"),
						std::make_pair("kernel",kname),
						std::make_pair("eps",std::to_string(eps)),
						std::make_pair("k_l1",k_l1),
						std::make_pair("k_l2",k_l2),
						std::make_pair("gp_points",std::to_string(gp_points)),
						std::make_pair("ivm_points",std::to_string(split_points)),
					},
					with_header,
					folds
				);
			}
		}
	}
}

int main(int argc, char const* argv[]) {
    std::cout << "Reading files" << std::endl;
	std::vector<internal_t> X;
	std::vector<internal_t> Y;
	auto pathToData = "../Datasets/Range-Queries-Aggregates.csv";
	size_t dim = read_csv(pathToData, X,  Y);
	size_t N = Y.size();

	std::cout << "=== DATA READ ===" << std::endl;
	for (unsigned int i = 0;i < 10; ++i) {
		for (unsigned int j = 0; j < dim; ++j) {
			std::cout << X[i*dim+j] << " ";
		}
		std::cout << "- " << Y[i] << std::endl;
	}
	std::cout << "======" << std::endl << std::endl;

	normalize<internal_t>(&X[0], N, dim);

	std::cout << "=== DATA NORMALIZED ===" << std::endl;
	for (unsigned int i = 0;i < 10; ++i) {
		for (unsigned int j = 0; j < dim; ++j) {
			std::cout << X[i*dim+j] << " ";
		}
		std::cout << "- " << Y[i] << std::endl;
	}
    std::cout << "======" << std::endl << std::endl;

	Dataset<internal_t, internal_t, internal_t> D(&X[0], &Y[0], N, dim);
	bool print_header = true;
	const size_t xval_runs = 10;

	for (auto l : {0.01, 0.1, 1.0}) {
		Matern3_2<internal_t, internal_t> k32(l);
		test_all_models(D, k32, "M32", std::to_string(l), "None", print_header, xval_runs);
		print_header = false;
		Matern5_2<internal_t, internal_t> k52(l);
		test_all_models(D, k52, "M52", std::to_string(l), "None", print_header, xval_runs);
		print_header = false;
		RBFKernel<internal_t,internal_t> kRBF(l, dim);
		test_all_models(D, kRBF, "RBF", std::to_string(l), "None", print_header, xval_runs);
	}
}
