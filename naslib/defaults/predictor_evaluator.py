import codecs
import time
import json
import logging
import os
import numpy as np
import copy
import torch
from scipy import stats
from sklearn import metrics

from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import utils

logger = logging.getLogger(__name__)


class PredictorEvaluator(object):
    """
    Evaluate a chosen predictor.
    """

    def __init__(self, predictor, config=None):
        """
        Initializes the Evaluator.
        """
        self.predictor = predictor
        self.config = config
        self.experiment_type = config.experiment_type

        self.test_size = config.test_size
        self.train_size_single = config.train_size_single
        self.train_size_list = config.train_size_list        
        self.fidelity_single = config.fidelity_single
        self.fidelity_list = config.fidelity_list

        self.dataset = config.dataset
        self.metric = Metric.VAL_ACCURACY
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = [config]
        self.learn_hyper = False

    def adapt_search_space(self, search_space, load_labeled, scope=None, dataset_api=None):
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.predictor.set_ss_type(self.search_space.get_type())
        self.load_labeled = load_labeled
        self.dataset_api = dataset_api
        
        # todo: see if we can query 'flops', 'latency', 'params' in darts
        if self.search_space.get_type() == 'nasbench101':
            self.full_lc = False
            self.hyperparameters = False
        elif self.search_space.get_type() == 'nasbench201':
            self.full_lc = True
            self.hyperparameters = True
        elif self.search_space.get_type() == 'darts':
            self.full_lc = True
            self.hyperparameters = False
        else:
            raise NotImplementedError('This search space is not yet implemented in PredictorEvaluator.')

    def load_dataset(self, load_labeled=False, data_size=10):
        """
        There are two ways to load an architecture.
        load_labeled=False: sample a random architecture from the search space.
        This works on NAS benchmarks where we can query any architecture (nasbench101/201/301)
        load_labeled=True: sample a random architecture from a set of evaluated architectures.
        When we only have data on a subset of the search space (e.g., the set of 5k DARTS
        architectures that have the full training info).
        
        After we load an architecture, query the final val accuracy.
        If the predictor requires extra info such as partial learning curve info, query that too.
        """
        xdata = []
        ydata = []
        info = []
        train_times = []
        for _ in range(data_size):
            if not load_labeled:
                arch = self.search_space.clone()
                arch.sample_random_architecture(dataset_api=self.dataset_api)
            else:
                arch = self.search_space.clone()
                arch.load_labeled_architecture(dataset_api=self.dataset_api)
            
            accuracy = arch.query(metric=self.metric, 
                                  dataset=self.dataset, 
                                  dataset_api=self.dataset_api)
            train_time = arch.query(metric=Metric.TRAIN_TIME, 
                                    dataset=self.dataset, 
                                    dataset_api=self.dataset_api)
            data_reqs = self.predictor.get_data_reqs()
            if data_reqs['requires_partial_lc']:
                # add partial learning curve if applicable
                assert self.full_lc, 'This predictor requires learning curve info'
                lc = arch.query(metric=data_reqs['metric'],
                                full_lc=True,
                                dataset=self.dataset, 
                                dataset_api=self.dataset_api)
                info_dict = {'lc':lc}
                if data_reqs['requires_hyperparameters']:
                    assert self.hyperparameters, 'This predictor requires querying arch hyperparams'                
                    for hp in data_reqs['hyperparams']:
                        info_dict[hp] = arch.query(Metric.HP, dataset=self.dataset, 
                                                   dataset_api=self.dataset_api)[hp]
                info.append(info_dict)
            xdata.append(arch)
            ydata.append(accuracy)
            train_times.append(train_time)
        return xdata, ydata, info, train_times

    def single_evaluate(self, train_data, test_data, fidelity):
        xtrain, ytrain, train_info, train_times = train_data
        xtest, ytest, test_info, _ = test_data
        train_size = len(xtrain)
        
        data_reqs = self.predictor.get_data_reqs()
    
        logger.info("Fit the predictor")
        if data_reqs['requires_partial_lc']:
            """
            todo: distinguish between predictors that need LC info
            at training time vs test time
            """
            train_info = copy.deepcopy(train_info)
            test_info = copy.deepcopy(test_info)
            for info_dict in train_info:
                info_dict['lc'] = info_dict['lc'][:fidelity]
            for info_dict in test_info:
                info_dict['lc'] = info_dict['lc'][:fidelity]

        # check whether need to run HPO:
        # the first condition check whether the predictor needs separate HPO
        # the second condition check whether we want to run HPO at current fidelity or train_size
        self.learn_hyper = True # TODO for debug --> remove this line later
        if self.predictor.need_separate_hpo and self.learn_hyper:
            self.run_hpo(xtrain, ytrain, iters=100, score_metric='pearson') #TODO iters= specify how many hps to be tried
            self.learn_hyper = False

        fit_time_start = time.time()
        self.predictor.fit(xtrain, ytrain, train_info)
        fit_time_end = time.time()
        test_pred = self.predictor.query(xtest, test_info)
        query_time_end = time.time()

        #If the predictor is an ensemble, take the mean
        if len(test_pred.shape) > 1:
            test_pred = np.mean(test_pred, axis=0)

        logger.info("Compute evaluation metrics")
        results_dict = self.compare(ytest, test_pred)
        results_dict['train_size'] = train_size
        results_dict['fidelity'] = fidelity
        results_dict['train_time'] = np.sum(train_times)
        results_dict['fit_time'] = fit_time_end - fit_time_start
        results_dict['query_time'] = (query_time_end - fit_time_end) / len(xtest)
        # print abridged results on one line:
        logger.info("train_size: {}, fidelity: {}, pearson correlation {}"
                    .format(train_size, fidelity, np.round(results_dict['pearson'], 4)))
        # print entire results dict:
        print_string = ''
        for key in results_dict:
            print_string += key + ': {}, '.format(np.round(results_dict[key], 4))
        logger.info(print_string)
        self.results.append(results_dict)
        """
        Todo: query_time currently does not include the time taken to train a partial learning curve
        """

    def evaluate(self):
        self.predictor.pre_process()

        logger.info("Load the test set")
        test_data = self.load_dataset(load_labeled=self.load_labeled, data_size=self.test_size)

        if self.experiment_type == 'single':
            train_size = self.train_size_single
            fidelity = self.fidelity_single
            logger.info("Load the training set")
            train_data = self.load_dataset(load_labeled=self.load_labeled,
                                           data_size=train_size)
            self.single_evaluate(train_data, test_data, fidelity=fidelity)

        elif self.experiment_type == 'vary_train_size':
            logger.info("Load the training set")
            full_train_data = self.load_dataset(load_labeled=self.load_labeled,
                                                data_size=self.train_size_list[-1])
            fidelity = self.fidelity_single

            for train_size in self.train_size_list:
                # if train_size % n == 0:
                #     self.learn_hyper = True
                train_data = [data[:train_size] for data in full_train_data]
                self.single_evaluate(train_data, test_data, fidelity=fidelity)

        elif self.experiment_type == 'vary_fidelity':
            train_size = self.train_size_single
            logger.info("Load the training set")
            train_data = self.load_dataset(load_labeled=self.load_labeled,
                                           data_size=self.train_size_single)

            for fidelity in self.fidelity_list:
                # if fidelity % k == 0:
                #     self.learn_hyper = True
                self.single_evaluate(train_data, test_data, fidelity=fidelity)

        elif self.experiment_type == 'vary_both':
            logger.info("Load the training set")
            full_train_data = self.load_dataset(load_labeled=self.load_labeled,
                                                data_size=self.train_size_list[-1])

            for train_size in self.train_size_list:
                train_data = [data[:train_size] for data in full_train_data]

                for fidelity in self.fidelity_list:
                    # if train_size % n ==0 or fidelity % k == 0:
                    #     self.learn_hyper = True
                    self.single_evaluate(train_data, test_data, fidelity=fidelity)                

        else:
            raise NotImplementedError()

        self._log_to_json()

    def compare(self, ytest, test_pred):
        ytest = np.array(ytest)
        test_pred = np.array(test_pred)
        metrics_dict = {}
        metrics_dict['mae'] = np.mean(abs(test_pred - ytest))
        metrics_dict['rmse'] = metrics.mean_squared_error(ytest, test_pred, squared=False)
        metrics_dict['pearson'] = np.abs(np.corrcoef(ytest, test_pred)[1,0])
        metrics_dict['spearman'] = stats.spearmanr(ytest, test_pred)[0]
        metrics_dict['kendalltau'] = stats.kendalltau(ytest, test_pred)[0]
        metrics_dict['kt_2dec'] = stats.kendalltau(ytest, np.round(test_pred, decimals=2))[0]
        metrics_dict['kt_1dec'] = stats.kendalltau(ytest, np.round(test_pred, decimals=1))[0]
        for k in [10, 20]:
            top_ytest = np.array([y > sorted(ytest)[max(-len(ytest),-k-1)] for y in ytest])
            top_test_pred = np.array([y > sorted(test_pred)[max(-len(test_pred),-k-1)] for y in test_pred])
            metrics_dict['precision_{}'.format(k)] = sum(top_ytest & top_test_pred) / k

        return metrics_dict

    def _log_to_json(self):
        """log statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        with codecs.open(os.path.join(self.config.save, 'errors.json'), 'w', encoding='utf-8') as file:
            json.dump(self.results, file, separators=(',', ':'))

    def run_hpo(self, xtrain, ytrain, iters=3, score_metric='pearson'):
        logger.info(f'Perform HPO: run CV among {iters} hyperparams and scored by {score_metric}')
        n_train = len(xtrain)
        split_indices = utils.generate_kFold(n_train, 3)
        predictor = copy.deepcopy(self.predictor)

        # use score metric: maximise score metric
        best_score = -1e6
        best_hyperparams = None

        for i in range(iters):
            hyperparams = predictor.get_random_hyperparams()
            print(f'hyper_i = {hyperparams}') # for debug TODO remove this line later
            # check whether the predictor needs epoch training: if so, reduce the training epoch number during HPO
            if 'epochs' in hyperparams.keys():
                hyperparams['epochs'] = int(hyperparams['epochs']/10)
            predictor.hyperparams = hyperparams
            cv_score = utils.cross_validation(xtrain, ytrain, predictor, split_indices, score_metric)
            if cv_score > best_score:
                best_hyperparams = hyperparams
                best_score = cv_score

        if best_score is np.nan:
            best_hyperparams = predictor.default_hyperparams
        elif 'epochs' in best_hyperparams.keys():
            # best_hyperparams['epochs'] = predictor.default_hyperparams['epochs']
            best_hyperparams['epochs'] = int(best_hyperparams['epochs'] * 10)
        logger.info(f'Perform HPO: Best hyperparams = {best_hyperparams} with CV score = {best_score}')
        self.predictor.hyperparams = best_hyperparams

