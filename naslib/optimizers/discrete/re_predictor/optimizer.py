"""
author: Felix Jablonski <33359018+jabb0@users.noreply.github.com>

Notes:
    - This implementation does not support zero cost and semi performance predictors yet
    - All models in the initial population are sampled using real runs and not the performance predictor
    - The interval in which a model is evaluated actually is given by config.train_interval
        and defaults to DEFAULT_TRAIN_INTERVAL
"""
import logging
import torch
import copy
import numpy as np

from naslib.predictors.ensemble import Ensemble  # This supports all performance predictors
from naslib.optimizers.discrete.bananas.acquisition_functions import acquisition_function
from naslib.optimizers import RegularizedEvolution as RE
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)


class RegularizedEvolutionWithPredictor(RE):
    DEFAULT_TRAIN_INTERVAL = 10

    def __init__(self, config):
        super().__init__(config)
        # Add the performance predictor options
        self.num_ensemble = config.search.num_ensemble
        self.predictor_type = config.search.predictor_type
        self.encoding_type = config.search.encoding_type  # currently not implemented
        self.num_ensemble = config.search.num_ensemble
        self.predictor_type = config.search.predictor_type
        self.acq_fn_type = config.search.acq_fn_type

        # Note: Zerocost and semi are not implemented
        self.zc = ('omni' in self.predictor_type)
        self.semi = ('semi' in self.predictor_type)
        if self.zc or self.semi:
            raise NotImplementedError("omni and semi predictors are not implemented for RE with predictor")

        self.train_data = []  # The training data for the predictor
        # The interval after which to train an actual model instead of using the predictor
        self.real_train_interval = config.get("train_interval", self.DEFAULT_TRAIN_INTERVAL)

        # Set later in adapt_search_space()
        self.ss_type = None

        self.ensemble = None
        self.acq_fn = None

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        super().adapt_search_space(search_space, scope=scope, dataset_api=dataset_api)
        self.ss_type = search_space.get_type()

    def _train_ensemble(self):
        # train a neural predictor
        xtrain = [m.arch for m in self.train_data]
        ytrain = [m.accuracy for m in self.train_data]
        ensemble = Ensemble(num_ensemble=self.num_ensemble,
                            ss_type=self.ss_type,
                            predictor_type=self.predictor_type,
                            config=self.config)
        train_error = ensemble.fit(xtrain, ytrain)
        logger.info(f"Fitted ensemble with train error of {train_error}")
        acq_fn = acquisition_function(ensemble=self.ensemble,
                                      ytrain=ytrain,
                                      acq_fn_type=self.acq_fn_type)
        self.ensemble = ensemble
        self.acq_fn = acq_fn

    def _add_initial_model(self, epoch):
        """
        Draw and add a model to the initial population
        This will do a full performance evaluation of the architecture instead of using the predictor
        """
        # If there is no scope defined, let's use the search space default one
        model = torch.nn.Module()  # hacky way to get arch and accuracy checkpointable
        model.arch = self.search_space.clone()
        model.arch.sample_random_architecture(dataset_api=self.dataset_api)
        model.accuracy = model.arch.query(self.performance_metric,
                                          self.dataset,
                                          dataset_api=self.dataset_api)

        log_every_n_seconds(logging.INFO, "Population size {}".format(len(self.population)))
        # Note: For zero-cost estimators one needs to compute the zero cost score of the model here too

        # Store every model for that a full performance evaluation has been performed as training data
        # Add the model and its performance to the training data of the performance predictor
        self.train_data.append(model)
        return model

    def _self_predict_accuracy(self, arch):
        """
        Predict the accuracy using the current acquisition function and ensemble given the architecture encoding
        """
        return self.acq_fn(arch)

    def _add_genetic_model(self, epoch):
        """
        Select a new model based on regularized evolution.

        """
        sample = []
        while len(sample) < self.sample_size:
            candidate = np.random.choice(list(self.population))
            sample.append(candidate)

        parent = max(sample, key=lambda x: x.accuracy)

        # Child is a new model
        child = torch.nn.Module()  # hacky way to get arch and accuracy checkpointable
        child.arch = self.search_space.clone()
        child.arch.mutate(parent.arch, dataset_api=self.dataset_api)

        # Select the accuracy based on the real evaluation every of self.real_train_interval model
        if (epoch % self.real_train_interval) == 0:
            logger.info("Using real performance of the model")
            child.accuracy = child.arch.query(self.performance_metric,
                                              self.dataset,
                                              dataset_api=self.dataset_api)
            # Add the child for which we have correct data to the training set
            self.train_data.append(child)
            # Retrain the ensemble
            self._train_ensemble()
        # Evaluate the model accuracy using the predictor
        else:
            child.accuracy = self._self_predict_accuracy(child.arch.clone())

        return child

    def new_epoch(self, epoch):
        # We sample as many architectures as we need
        if epoch < self.population_size:
            logger.debug("Drawing model for the initial population")
            model = self._add_initial_model(epoch)
            if epoch == self.population_size - 1:
                self._train_ensemble()
        # If we have a full population and want to start the evolvement process
        else:
            model = self._add_genetic_model(epoch)

        self.population.append(model)
        self._update_history(model)

    def get_op_optimizer(self):
        raise NotImplementedError()
