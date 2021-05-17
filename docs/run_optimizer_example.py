import logging
import os

from naslib.defaults.trainer import Trainer
from naslib.optimizers import RegularizedEvolutionWithPredictor
from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.search_spaces import NasBench201SearchSpace as NB201

# instantiate the search space object
search_space = NB201()


def run_optimizer(config_file="./docs/re_pred_run_0/cifar10/configs/nas_predictors/config_re_pred_gp_0.yaml",
                  nas_optimizer=RegularizedEvolutionWithPredictor) -> None:
    # add all the utilities, such as config file reading, logging as before.
    # afterwards instantiate the search space, optimizer, trainer and run the search + evaluation

    # NOTE: I highly recommend building a second function to do this.
    # E.g. allow to modify/patch an existing config based on other configs as a method
    # Maybe the current implementation suits the need already as this tools will mostly be used in command line
    # in this case a docstring for the programmatic use will be enough

    config = utils.get_config_from_args(args=None, config_type="nas_predictor", merge_file=config_file)

    # Setup seed for reproducibility
    utils.set_seed(config.seed)
    # Log the config parameters as good practice
    utils.log_args(config)
    # Set the logging to the output directory
    logger = setup_logger(os.path.join(config.save, "log.log"))
    logger.setLevel(logging.INFO)
    # Initialize the desired NAS optimizer with the desired config
    optimizer = nas_optimizer(config)
    # Define the search space to search on
    search_space = NB201()
    # As this is a tabular benchmark we need to load the dataset. This depends on the searchspace
    # TODO: Why is config.search_space required here?: The loader type depends on the dataset type
    dataset_api = get_dataset_api(config.search_space, config.dataset)
    # Adapt the optimizer to the search space
    optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

    # Build a NAS Trainer based on the config and optimizer
    trainer = Trainer(optimizer, config, lightweight_output=True)
    # Run the trainer
    trainer.search()
    # Evaluate the trainer
    trainer.evaluate(dataset_api=dataset_api)


if __name__ == '__main__':
    print(os.getcwd())
    run_optimizer()
