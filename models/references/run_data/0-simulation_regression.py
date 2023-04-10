
import os
import json
import copy
import random
import argparse

import torch
import numpy as np
import pandas as pd

from rich.console import Console
from torch.utils.data import Subset

from datasets.simulation import RegressionExample
from utils_datasets.utils import DictionaryDataset

from models.linear_regression import LinearRegression
from models.irm import IRMRegression
from models.robust import GroupDRORegression
from models.robust import VRExRegression
from models.heckman.linear_models_deprecated import HeckmanRegression
from models.heckman.linear_models_deprecated import HeckmanDGRegression
from models.heckman.linear_models_deprecated import TrueHeckmanDGRegression
from models.heckman.linear_models_deprecated import SafeHeckmanDGRegression

from utils.misc import fix_random_seed


def parse_arguments(description: str = 'HeckmanDG, regression, linear case.') -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--num_train_domains', type=int,
                        default=2,
                        help='Number of training domains (default: 2)')
    parser.add_argument('--num_test_domains', type=int,
                        default=1,
                        help='Number of testing domains (default: 1)')
    
    parser.add_argument('--input_features', type=int,
                        default=2,
                        help='Number of input covariates (default: 2)')
    parser.add_argument('--selection_feature_index', nargs='+', type=int,
                        default=[1, ],
                        help='')
    
    parser.add_argument('--alpha_prior_mean', type=float,
                        default=5.0,
                        help='Mean of alpha coefficients for selection models (default: 5.0)')
    parser.add_argument('--alpha_prior_sigma', type=float,
                        default=3.0,
                        help='Standard deviation of alpha coefficients for selection models (default: 3.0)')
    
    parser.add_argument('--sigma_s', type=float,
                        default=1.0,
                        help='Standard deviation of selection error terms (default: 1.0)')
    parser.add_argument('--sigma_y', type=float,
                        default=1.0,
                        help='Standard deviation of outcome error term (default: 1.0)')
    
    parser.add_argument('--rho', type=float,
                        default=0.8,
                        help='Correlation between error terms (default: 0.8)')
    
    parser.add_argument('--population_size', type=int,
                        default=100000,
                        help='True population size (default: 100K)')
    parser.add_argument('--sampling_rate_per_domain', type=float,
                        default=0.01,
                        help='Sampling rate for each domain (default: 0.01)')
    
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', ],
                        default='sgd',
                        help='Optimizer (default: sgd)')
    
    parser.add_argument('--learning_rate', type=float,
                        default=0.001,
                        help='Learning rate (default: 0.001)')
    
    parser.add_argument('--weight_decay', type=float,
                        default=0.0001,
                        help='Weight decay (default=0.0001)')
    
    parser.add_argument('--epochs', type=int,
                        default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int,
                        default=256,
                        help='Training batch size. If None or -1, triggers full batch training. (default: 256)')
    parser.add_argument('--unlabeled_batch_size', type=int,
                        default=None,
                        help="Training batch size for unlabeled data, If None, use `batch_size` (default: None)")
    
    parser.add_argument('--device', type=str,
                        default='cpu',
                        help='Training device (default: cpu)')
    
    parser.add_argument('--num_trials', type=int,
                        default=30,
                        help='Number of repeated experiments (default: 30)')
    
    parser.add_argument('--write_dir', type=str,
                        default='./results',
                        help='Base directory to write result files.')

    parser.add_argument('--skip_erm', action='store_true')
    parser.add_argument('--skip_heckman', action='store_true')
    parser.add_argument('--skip_irm', action='store_true')
    parser.add_argument('--skip_groupdro', action='store_true')
    parser.add_argument('--skip_coral', action='store_true')
    parser.add_argument('--skip_vrex', action='store_true')

    return parser.parse_args()


def main():
    """Main function."""

    # Parse command-line arguments
    args = parse_arguments()

    # Rich console for printing
    console = Console()
    console.print(vars(args))

    # Specify output directory
    os.makedirs(args.write_dir, exist_ok=True)
    subfolder: str = os.path.join(
        args.write_dir,
        f"inputs={args.input_features}_domain={args.num_train_domains}_" + \
        f"rate={args.sampling_rate_per_domain:.2f}_rho={args.rho:.1f}_" + \
        f"mean={args.alpha_prior_mean}_sigma={args.alpha_prior_sigma:.1f}"
    )
    os.makedirs(subfolder, exist_ok=True)
    
    # Specify output file (raise error if exists)
    write_file: str = os.path.join(subfolder, 'results.csv')
    if os.path.exists(write_file):
        raise FileExistsError(f"{write_file} exists. Try a different name.")

    for t in range(args.num_trials):

        # buffer for dictionaries, which is later converted into a pandas dataframe
        results = list()

        # (TODO) fix random seed for reproduction
        fix_random_seed(s=2022 + t)

        # start trial
        console.print(f":thread: Trial {t:,}")

        """
            1-1. Training dataset.
        """
        dataset = RegressionExample(
            input_covariance_matrix=torch.eye(args.input_features),
            outcome_intercept=torch.tensor([1.0, ]),
            outcome_coef=torch.tensor([1.5, 3.0]),
            num_train_domains=args.num_train_domains,
            num_test_domains=args.num_test_domains,
            alpha_prior_mean=args.alpha_prior_mean,
            alpha_prior_sigma=args.alpha_prior_sigma,
            selection_feature_index=args.selection_feature_index,
            sigma_s=args.sigma_s,
            sigma_y=args.sigma_y,
            rho=args.rho,
            population_size=args.population_size,
            sampling_rate_per_domain=args.sampling_rate_per_domain,
        )
        console.print(dataset)

        """
            1-2. In-distribution test set, created upon dataset initialization by default.
        """
        id_test_set = DictionaryDataset(dataset.test_data)
        console.print(f"In-distribution test set: {len(id_test_set):,}")

        """
            1-3. Randomly sampled test set, reflecting the true population.
        """
        _random_sample_size = int(args.population_size * args.sampling_rate_per_domain)
        random_test_set = DictionaryDataset(dataset.randomly_sample_data(size=_random_sample_size), )
        console.print(f"Random test set: {len(random_test_set):,}")


        """
            1-4. Moderate out-of-distribution test set using a different prior  for alpha coefficients.
                if args.alpha_prior_mean == 0, mean += {5,-5} (a pos/neg mean shift)
                if args.alpha_prior_mean != 0, mean *= 2 (same direction, but mean shift)
        """
        if len(dataset.selection_feature_index) > 1:
            raise NotImplementedError(
                f"Current does not support the scenario "
                f"with {len(dataset.selection_feature_index)} selection variables."
            )
        
        moderate_ood_alpha_prior_mean = torch.zeros_like(dataset.alpha_prior_mean)
        for i, _a_mean in enumerate(dataset.alpha_prior_mean):
            if _a_mean == 0:
                # mean shift; +5
                moderate_ood_alpha_prior_mean[i] =  5.0
            else:
                # mean shift in the same direction
                moderate_ood_alpha_prior_mean[i] = 2 * _a_mean
        
        # (moderate) generate selection model based on prior specified above
        moderate_ood_selection_coef, moderate_ood_selection_intercept = \
            dataset.get_selection_models(
                num_domains=args.num_test_domains,
                alpha_prior_mean=moderate_ood_alpha_prior_mean,
                alpha_prior_sigma=dataset.alpha_prior_sigma,
                selection_feature_index=dataset.selection_feature_index,
            )

        # (moderate) sample data according to the selection model
        moderate_ood_test_set = DictionaryDataset(
            dataset.sample_data(
                selection_coef=moderate_ood_selection_coef,
                selection_intercept=moderate_ood_selection_intercept,
                mode='moderate_ood_test'  # any string will do except 'train'
            )
        )
        console.print(f"Moderate out-of-distribution test set: {len(moderate_ood_test_set):,}")

        """
            1-5. Extreme out-of-distribution test set using a different prior for alpha coefficients
                if args.alpha_prior_mean == 0, mean += {10, -10} (a pos/neg mean shift)
                if args.alpha_prior_mean != 0, mean *= -1 (opposite sign)
        """
        if len(dataset.selection_feature_index) > 1:
            raise NotImplementedError(
                f"Current does not support the scenario "
                f"with {len(dataset.selection_feature_index)} selection variables."
            )

        extreme_ood_alpha_prior_mean = torch.zeros_like(dataset.alpha_prior_mean)
        for i, _a_mean in enumerate(dataset.alpha_prior_mean):
            if _a_mean == 0:
                # mean shift; {+10}
                extreme_ood_alpha_prior_mean[i] = 10.
            else:
                # opposite sign
                extreme_ood_alpha_prior_mean[i] = torch.neg(_a_mean)

        # (extreme) generate selection model based on prior specified above
        extreme_ood_selection_coef, extreme_ood_selection_intercept = dataset.get_selection_models(
            num_domains=args.num_test_domains,
            alpha_prior_mean=extreme_ood_alpha_prior_mean,
            alpha_prior_sigma=dataset.alpha_prior_sigma,
            selection_feature_index=dataset.selection_feature_index,
        )

        # (extreme) sample data according to the selection model
        extreme_ood_test_set = DictionaryDataset(
            dataset.sample_data(
                selection_coef=extreme_ood_selection_coef,
                selection_intercept=extreme_ood_selection_intercept,
                mode='extreme_ood_test'  # any string will do except 'train'
            )
        )
        console.print(f"Extreme out-of-distribution test set: {len(extreme_ood_test_set):,}")

        """
            1-6. Compliment test set; all the unselected observations
        """
        compliment_test_set = DictionaryDataset(dataset.compliment_data)
        console.print(f"Compliment test set: {len(compliment_test_set):,}")

        """
            Common logging information shared across algorithms.
        """
        _results_base = dict(
            trial=t,
            num_train_domains=args.num_train_domains,
            num_test_domains=args.num_test_domains,
            num_input_features=args.input_features,
            selection_feature_index=args.selection_feature_index,
            alpha_prior_mean=args.alpha_prior_mean,
            alpha_prior_sigma=args.alpha_prior_sigma,
            moderate_ood_alpha_prior_mean=moderate_ood_alpha_prior_mean[0].item(),    # FIXME: will not work for more than one selection variable
            extreme_ood_alpha_prior_mean=extreme_ood_alpha_prior_mean[0].item(),      # FIXME: will not work for more than one selection variable
            sigma_s=args.sigma_s,
            sigma_y=args.sigma_y,
            rho=args.rho,
            population_size=args.population_size,
            sampling_rate_per_domain=args.sampling_rate_per_domain,
            unique_train_counts=dataset.unique_train_counts,
            train_test_overlap=dataset.train_test_overlap,
            train_size=len(dataset),
            id_test_size=len(id_test_set),
            moderate_ood_test_size=len(moderate_ood_test_set),
            extreme_ood_test_size=len(extreme_ood_test_set),
            random_test_size=len(random_test_set),
            compliment_test_size=len(compliment_test_set),
            epochs=args.epochs,
            batch_size=args.batch_size,
            unlabeled_batch_size=args.unlabeled_batch_size,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Save beta coefficients & intercept
        _results_base['beta_true_0'] = dataset.outcome_intercept.item()
        for j in range(args.input_features):
            _results_base[f'beta_true_{j+1}'] = dataset.outcome_coef[j].item()
        
        # Save the alpha coefficients & intercepts
        for k in range(args.num_train_domains):
            _results_base[f'alpha_true_0_{k}'] = dataset.selection_intercept.squeeze()[k].item()
            for j in args.selection_feature_index:
                _results_base[f'alpha_true_{j+1}_{k}'] = dataset.selection_coef[j, k].item()

        console.print(_results_base)

        # Save base info to json file
        with open(os.path.join(subfolder, f'configs_{t}.json'), 'w') as fp:
            json.dump(_results_base, fp, indent=2)

        if not args.skip_erm:
            """
                A-0) Empirical Risk Minimization with randomly sampled training data.
            """
            trainer = LinearRegression(
                input_features=args.input_features,
                hparams={
                    'optimizer': args.optimizer,
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                },
                device=args.device
            )
            train_history: dict = trainer.fit(
                train_set=DictionaryDataset(
                    dataset.randomly_sample_data(size=int(args.population_size * args.sampling_rate_per_domain)),
                ),
                epochs=args.epochs,
                batch_size=args.batch_size,
                description="ERM (data = random, oracle, linear regression) "
            )
            train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
            id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
            moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
            extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
            random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
            compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

            _model_results = copy.deepcopy(_results_base)
            _model_results['model'] = 'ERM (random)'
            _model_results['loss'] = train_history['loss'][-1].item()

            _model_results['beta_hat_0'] = trainer.model.linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'beta_hat_{j+1}'] = trainer.model.linear.weight.data[0][j].item()
            _model_results['sigma_hat'] = None
            _model_results['rho_hat'] = None

            _model_results['train_mae'] = train_metrics['mae']
            _model_results['id_test_mae'] = id_test_metrics['mae']
            _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
            _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
            _model_results['random_test_mae'] = random_test_metrics['mae']
            _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

            _model_results['train_mse'] = train_metrics['mse']
            _model_results['id_test_mse'] = id_test_metrics['mse']
            _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
            _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
            _model_results['random_test_mse'] = random_test_metrics['mse']
            _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

            _model_results['train_mape'] = train_metrics['mape']
            _model_results['id_test_mape'] = id_test_metrics['mape']
            _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
            _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
            _model_results['random_test_mape'] = random_test_metrics['mape']
            _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

            _model_results['train_r2'] = train_metrics['r2']
            _model_results['id_test_r2'] = id_test_metrics['r2']
            _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
            _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
            _model_results['random_test_r2'] = random_test_metrics['r2']
            _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

            results.append(_model_results); del _model_results

            """
                A-1) Pooled Empirical Risk Minimization, using the pooled data but ignoring domain information.
            """
            trainer = LinearRegression(
                input_features=args.input_features,
                hparams={
                    'optimizer': args.optimizer,
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                },
                device=args.device
            )
            train_history: dict = trainer.fit(
                train_set=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                description="ERM (data = pooled, linear regression) ",
            )

            train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
            id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
            moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
            extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
            random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
            compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

            _model_results = copy.deepcopy(_results_base)
            _model_results['model'] = 'ERM (pooled)'
            _model_results['loss'] = train_history['loss'][-1].item()

            _model_results['beta_hat_0'] = trainer.model.linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'beta_hat_{j+1}'] = trainer.model.linear.weight.data[0][j].item()
            _model_results['sigma_hat'] = None
            _model_results['rho_hat'] = None

            _model_results['train_mae'] = train_metrics['mae']
            _model_results['id_test_mae'] = id_test_metrics['mae']
            _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
            _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
            _model_results['random_test_mae'] = random_test_metrics['mae']
            _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

            _model_results['train_mse'] = train_metrics['mse']
            _model_results['id_test_mse'] = id_test_metrics['mse']
            _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
            _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
            _model_results['random_test_mse'] = random_test_metrics['mse']
            _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

            _model_results['train_mape'] = train_metrics['mape']
            _model_results['id_test_mape'] = id_test_metrics['mape']
            _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
            _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
            _model_results['random_test_mape'] = random_test_metrics['mape']
            _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

            _model_results['train_r2'] = train_metrics['r2']
            _model_results['id_test_r2'] = id_test_metrics['r2']
            _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
            _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
            _model_results['random_test_r2'] = random_test_metrics['r2']
            _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

            results.append(_model_results); del _model_results

            """
                A-2. Domain-Separate Empirical Risk Minimization, using single domain data only.
            """
            for k in range(dataset.num_train_domains):
                trainer = LinearRegression(
                    input_features=args.input_features,
                    hparams={
                        'optimizer': args.optimizer,
                        'learning_rate': args.learning_rate,
                        'weight_decay': args.weight_decay,
                    },
                    device=args.device
                )
                train_history: dict = trainer.fit(
                    train_set=Subset(
                        dataset=dataset,
                        indices=dataset.data['domain'].eq(k).nonzero(as_tuple=True)[0],
                    ),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    description=f"ERM (data = domain {k}, linear regression) "
                )

                train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
                id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
                moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
                extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
                random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
                compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

                _model_results = copy.deepcopy(_results_base)
                _model_results['model'] = f'ERM (domain={k})'
                _model_results['loss'] = train_history['loss'][-1].item()

                _model_results['beta_hat_0'] = trainer.model.linear.bias.data[0].item()
                for j in range(args.input_features):
                    _model_results[f'beta_hat_{j+1}'] = trainer.model.linear.weight.data[0][j].item()
                _model_results['sigma_hat'] = None
                _model_results['rho_hat'] = None

                _model_results['train_mae'] = train_metrics['mae']
                _model_results['id_test_mae'] = id_test_metrics['mae']
                _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
                _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
                _model_results['random_test_mae'] = random_test_metrics['mae']
                _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

                _model_results['train_mse'] = train_metrics['mse']
                _model_results['id_test_mse'] = id_test_metrics['mse']
                _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
                _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
                _model_results['random_test_mse'] = random_test_metrics['mse']
                _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

                _model_results['train_mape'] = train_metrics['mape']
                _model_results['id_test_mape'] = id_test_metrics['mape']
                _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
                _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
                _model_results['random_test_mape'] = random_test_metrics['mape']
                _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

                _model_results['train_r2'] = train_metrics['r2']
                _model_results['id_test_r2'] = id_test_metrics['r2']
                _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
                _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
                _model_results['random_test_r2'] = random_test_metrics['r2']
                _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

                results.append(_model_results); del _model_results


        if not args.skip_irm:
            """
                A-3) Invariant Risk Minimization.
            """
            trainer = IRMRegression(
                input_features=args.input_features,
                hparams={
                    'optimizer': args.optimizer,
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'lambda': 1e-3,
                    'penalty_anneal_iters': 0.,
                },
                device=args.device,
            )
            train_history: dict = trainer.fit(
                train_set=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                description="IRM (linear regression) ",
                group_ids=None,
            )

            train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
            id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
            moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
            extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
            random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
            compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

            _model_results = copy.deepcopy(_results_base)
            _model_results['model'] = 'IRM'
            _model_results['loss'] = train_history['loss'][-1].item()
            
            _model_results['beta_hat_0'] = trainer.model.linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'beta_hat_{j+1}'] = trainer.model.linear.weight.data[0][j].item()

            _model_results['train_mae'] = train_metrics['mae']
            _model_results['id_test_mae'] = id_test_metrics['mae']
            _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
            _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
            _model_results['random_test_mae'] = random_test_metrics['mae']
            _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

            _model_results['train_mse'] = train_metrics['mse']
            _model_results['id_test_mse'] = id_test_metrics['mse']
            _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
            _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
            _model_results['random_test_mse'] = random_test_metrics['mse']
            _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

            _model_results['train_mape'] = train_metrics['mape']
            _model_results['id_test_mape'] = id_test_metrics['mape']
            _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
            _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
            _model_results['random_test_mape'] = random_test_metrics['mape']
            _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

            _model_results['train_r2'] = train_metrics['r2']
            _model_results['id_test_r2'] = id_test_metrics['r2']
            _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
            _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
            _model_results['random_test_r2'] = random_test_metrics['r2']
            _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

            results.append(_model_results); del _model_results

        if not args.skip_groupdro:
            """
                A-4) Group Dristributionally Robust Optimization.
            """
            trainer = GroupDRORegression(
                num_groups=args.num_train_domains,
                input_features=args.input_features,
                hparams={
                    'optimizer': args.optimizer,
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'eta': 0.001,
                },
                device=args.device,
            )
            train_history: dict = trainer.fit(
                train_set=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                description="GroupDRO (linear regression) ",
                group_ids=None,
            )

            train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
            id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
            moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
            extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
            random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
            compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

            _model_results = copy.deepcopy(_results_base)
            _model_results['model'] = 'GroupDRO'
            _model_results['loss'] = train_history['loss'][-1].item()
            
            _model_results['beta_hat_0'] = trainer.model.linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'beta_hat_{j+1}'] = trainer.model.linear.weight.data[0][j].item()

            _model_results['train_mae'] = train_metrics['mae']
            _model_results['id_test_mae'] = id_test_metrics['mae']
            _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
            _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
            _model_results['random_test_mae'] = random_test_metrics['mae']
            _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

            _model_results['train_mse'] = train_metrics['mse']
            _model_results['id_test_mse'] = id_test_metrics['mse']
            _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
            _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
            _model_results['random_test_mse'] = random_test_metrics['mse']
            _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

            _model_results['train_mape'] = train_metrics['mape']
            _model_results['id_test_mape'] = id_test_metrics['mape']
            _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
            _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
            _model_results['random_test_mape'] = random_test_metrics['mape']
            _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

            _model_results['train_r2'] = train_metrics['r2']
            _model_results['id_test_r2'] = id_test_metrics['r2']
            _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
            _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
            _model_results['random_test_r2'] = random_test_metrics['r2']
            _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

            results.append(_model_results); del _model_results

        if not args.skip_vrex:
            """
            A-5) Variance Risk Extrapolation.
            """
            trainer = VRExRegression(
                num_groups=args.num_train_domains,
                input_features=args.input_features,
                hparams={
                    'optimizer': args.optimizer,
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'lambda': 0.001,
                    'n_groups_per_batch': 2,
                },
                device=args.device,
            )
            train_history: dict = trainer.fit(
                train_set=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                description="VREx (linear regression) ",
                group_ids=None,
            )

            train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
            id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
            moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
            extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
            random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
            compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

            _model_results = copy.deepcopy(_results_base)
            _model_results['model'] = 'VREx'
            _model_results['loss'] = train_history['loss'][-1].item()
            
            _model_results['beta_hat_0'] = trainer.model.linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'beta_hat_{j+1}'] = trainer.model.linear.weight.data[0][j].item()

            _model_results['train_mae'] = train_metrics['mae']
            _model_results['id_test_mae'] = id_test_metrics['mae']
            _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
            _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
            _model_results['random_test_mae'] = random_test_metrics['mae']
            _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

            _model_results['train_mse'] = train_metrics['mse']
            _model_results['id_test_mse'] = id_test_metrics['mse']
            _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
            _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
            _model_results['random_test_mse'] = random_test_metrics['mse']
            _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

            _model_results['train_mape'] = train_metrics['mape']
            _model_results['id_test_mape'] = id_test_metrics['mape']
            _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
            _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
            _model_results['random_test_mape'] = random_test_metrics['mape']
            _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

            _model_results['train_r2'] = train_metrics['r2']
            _model_results['id_test_r2'] = id_test_metrics['r2']
            _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
            _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
            _model_results['random_test_r2'] = random_test_metrics['r2']
            _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

            results.append(_model_results); del _model_results

        """
            B-1) Heckman-1.
                Uses unselected data as unobserved data.
        """
        trainer = HeckmanRegression(
            input_features=args.input_features,
            method='mle',
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        )
        train_history: dict = trainer.fit(
            labeled_set=dataset,
            unlabeled_set=compliment_test_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            unlabeled_batch_size=args.unlabeled_batch_size,
            description="Heckman-1 (linear regression) "
        )

        train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
        id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
        moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
        extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
        random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
        compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

        _model_results = copy.deepcopy(_results_base)
        _model_results['model'] = 'Heckman-1'
        _model_results['loss'] = train_history['outcome/loss'][-1].item()
        
        _model_results['beta_hat_0'] = trainer.outcome_model.linear.bias.data[0].item()
        for j in range(args.input_features):
            _model_results[f'beta_hat_{j+1}'] = trainer.outcome_model.linear.weight.data[0][j].item()
        
        _model_results['alpha_hat_0'] = trainer.selection_model.linear.bias.data[0].item()
        for j in range(args.input_features):
            _model_results[f'alpha_hat_{j+1}'] = trainer.selection_model.linear.weight.data[0][j].item()
        
        _model_results['sigma_hat'] = train_history['outcome/sigma'][-1].item()
        _model_results['rho_hat'] = train_history['outcome/rho'][-1].item()
        _model_results['beta_lambda'] = None

        _model_results['selection/accuracy'] = train_history['selection/accuracy'][-1].item()
        _model_results['selection/precision'] = train_history['selection/precision'][-1].item()
        _model_results['selection/recall'] = train_history['selection/recall'][-1].item()
        _model_results['selection/brier_score'] = train_history['selection/brier_score'][-1].item()

        _model_results['train_mae'] = train_metrics['mae']
        _model_results['id_test_mae'] = id_test_metrics['mae']
        _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
        _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
        _model_results['random_test_mae'] = random_test_metrics['mae']
        _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

        _model_results['train_mse'] = train_metrics['mse']
        _model_results['id_test_mse'] = id_test_metrics['mse']
        _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
        _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
        _model_results['random_test_mse'] = random_test_metrics['mse']
        _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

        _model_results['train_mape'] = train_metrics['mape']
        _model_results['id_test_mape'] = id_test_metrics['mape']
        _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
        _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
        _model_results['random_test_mape'] = random_test_metrics['mape']
        _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

        _model_results['train_r2'] = train_metrics['r2']
        _model_results['id_test_r2'] = id_test_metrics['r2']
        _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
        _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
        _model_results['random_test_r2'] = random_test_metrics['r2']
        _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

        results.append(_model_results); del _model_results

        """
        B-2) Heckman-2.
            Uses unselected data as unobserved data.
        """

        trainer = HeckmanRegression(
            input_features=args.input_features,
            method='two-step',
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        )
        train_history: dict = trainer.fit(
            labeled_set=dataset,
            unlabeled_set=compliment_test_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            unlabeled_batch_size=args.unlabeled_batch_size,
            description="Heckman-2 (linear regression) "
        )

        train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
        id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
        moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
        extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
        random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
        compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

        _model_results = copy.deepcopy(_results_base)
        _model_results['model'] = 'Heckman-2'
        _model_results['loss'] = train_history['outcome/loss'][-1].item()
        
        _model_results['beta_hat_0'] = trainer.outcome_model.linear.bias.data[0].item()
        for j in range(args.input_features):
            _model_results[f'beta_hat_{j+1}'] = trainer.outcome_model.linear.weight.data[0][j].item()
        _model_results['alpha_hat_0'] = trainer.selection_model.linear.bias.data[0].item()
        for j in range(args.input_features):
            _model_results[f'alpha_hat_{j+1}'] = trainer.selection_model.linear.weight.data[0][j].item()
        
        _model_results['sigma_hat'] = None
        _model_results['rho_hat'] = None
        _model_results['beta_lambda'] = train_history['outcome/beta_lambda'][-1].item()

        _model_results['selection/accuracy'] = train_history['selection/accuracy'][-1].item()
        _model_results['selection/precision'] = train_history['selection/precision'][-1].item()
        _model_results['selection/recall'] = train_history['selection/recall'][-1].item()
        _model_results['selection/brier_score'] = train_history['selection/brier_score'][-1].item()

        _model_results['train_mae'] = train_metrics['mae']
        _model_results['id_test_mae'] = id_test_metrics['mae']
        _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
        _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
        _model_results['random_test_mae'] = random_test_metrics['mae']
        _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

        _model_results['train_mse'] = train_metrics['mse']
        _model_results['id_test_mse'] = id_test_metrics['mse']
        _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
        _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
        _model_results['random_test_mse'] = random_test_metrics['mse']
        _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

        _model_results['train_mape'] = train_metrics['mape']
        _model_results['id_test_mape'] = id_test_metrics['mape']
        _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
        _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
        _model_results['random_test_mape'] = random_test_metrics['mape']
        _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

        _model_results['train_r2'] = train_metrics['r2']
        _model_results['id_test_r2'] = id_test_metrics['r2']
        _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
        _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
        _model_results['random_test_r2'] = random_test_metrics['r2']
        _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

        results.append(_model_results); del _model_results

        """
        C-1) HeckmanDG-1
             Uses out-of-domain data as unobserved data, for fitting each domain's selection model.
        """
        trainer = HeckmanDGRegression(
            train_domains=args.num_train_domains,
            input_features=args.input_features,
            method='mle',
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        )
        train_history: dict = trainer.fit(
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            description="HeckmanDG-1 (linear regression)"
        )

        train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
        id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
        moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
        extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
        random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
        compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

        _model_results = copy.deepcopy(_results_base)
        _model_results['model'] = 'HeckmanDG-1'
        _model_results['loss'] = train_history['outcome/loss'][-1].item()

        _model_results['beta_hat_0'] = trainer.outcome_model.linear.bias.data[0].item()
        for j in range(args.input_features):
            _model_results[f'beta_hat_{j+1}'] = trainer.outcome_model.linear.weight.data[0][j].item()
        for k in range(len(trainer.selection_models)):
            _model_results[f'alpha_hat_0_{k}'] = trainer.selection_models[k].linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'alpha_hat_{j+1}_{k}'] = trainer.selection_models[k].linear.weight.data[0][j].item()

        _model_results['sigma_hat'] = train_history['outcome/sigma'][-1].item()
        _model_results['rho_hat'] = train_history['outcome/rho'][-1].item()
        _model_results['beta_lambda'] = None

        _model_results['selection/accuracy'] = train_history['selection/accuracy'][-1].item()
        _model_results['selection/precision'] = train_history['selection/precision'][-1].item()
        _model_results['selection/recall'] = train_history['selection/recall'][-1].item()
        _model_results['selection/brier_score'] = train_history['selection/brier_score'][-1].item()

        _model_results['train_mae'] = train_metrics['mae']
        _model_results['id_test_mae'] = id_test_metrics['mae']
        _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
        _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
        _model_results['random_test_mae'] = random_test_metrics['mae']
        _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

        _model_results['train_mse'] = train_metrics['mse']
        _model_results['id_test_mse'] = id_test_metrics['mse']
        _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
        _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
        _model_results['random_test_mse'] = random_test_metrics['mse']
        _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

        _model_results['train_mape'] = train_metrics['mape']
        _model_results['id_test_mape'] = id_test_metrics['mape']
        _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
        _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
        _model_results['random_test_mape'] = random_test_metrics['mape']
        _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

        _model_results['train_r2'] = train_metrics['r2']
        _model_results['id_test_r2'] = id_test_metrics['r2']
        _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
        _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
        _model_results['random_test_r2'] = random_test_metrics['r2']
        _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

        results.append(_model_results); del _model_results

        """
            C-2) HeckmanDG-2
                 Uses out-of-domain data as unobserved data, for fitting each domain's selection model.
        """
        trainer = HeckmanDGRegression(
            train_domains=args.num_train_domains,
            input_features=args.input_features,
            method='two-step',
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        )
        
        train_history: dict = trainer.fit(
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            description="HeckmanDG-2 (linear regression)"
        )

        train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
        id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
        moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
        extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
        random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
        compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

        _model_results = copy.deepcopy(_results_base)
        _model_results['model'] = 'HeckmanDG-2'
        _model_results['loss'] = train_history['outcome/loss'][-1].item()

        _model_results['beta_hat_0'] = trainer.outcome_model.linear.bias.data[0].item()
        for j in range(args.input_features):
            _model_results[f'beta_hat_{j+1}'] = trainer.outcome_model.linear.weight.data[0][j].item()
        for k in range(len(trainer.selection_models)):
            _model_results[f'alpha_hat_0_{k}'] = trainer.selection_models[k].linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'alpha_hat_{j+1}_{k}'] = trainer.selection_models[k].linear.weight.data[0][j].item()

        _model_results['sigma_hat'] = None
        _model_results['rho_hat'] = None
        _model_results['beta_lambda'] = train_history['outcome/beta_lambda'][-1].item()

        _model_results['selection/accuracy'] = train_history['selection/accuracy'][-1].item()
        _model_results['selection/precision'] = train_history['selection/precision'][-1].item()
        _model_results['selection/recall'] = train_history['selection/recall'][-1].item()
        _model_results['selection/brier_score'] = train_history['selection/brier_score'][-1].item()

        _model_results['train_mae'] = train_metrics['mae']
        _model_results['id_test_mae'] = id_test_metrics['mae']
        _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
        _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
        _model_results['random_test_mae'] = random_test_metrics['mae']
        _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

        _model_results['train_mse'] = train_metrics['mse']
        _model_results['id_test_mse'] = id_test_metrics['mse']
        _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
        _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
        _model_results['random_test_mse'] = random_test_metrics['mse']
        _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

        _model_results['train_mape'] = train_metrics['mape']
        _model_results['id_test_mape'] = id_test_metrics['mape']
        _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
        _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
        _model_results['random_test_mape'] = random_test_metrics['mape']
        _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

        _model_results['train_r2'] = train_metrics['r2']
        _model_results['id_test_r2'] = id_test_metrics['r2']
        _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
        _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
        _model_results['random_test_r2'] = random_test_metrics['r2']
        _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

        results.append(_model_results); del _model_results

        """
            D-1) True HeckmanDG-1
                 Uses both out-of-domain & out-of-distribution data as unobserved data.
        """
        trainer = TrueHeckmanDGRegression(
            train_domains=args.num_train_domains,
            input_features=args.input_features,
            method='mle',
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device
        )

        train_history: dict = trainer.fit(
            labeled_set=dataset,
            unlabeled_set=compliment_test_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            unlabeled_batch_size=args.unlabeled_batch_size,
            description="True HeckmanDG-1 (linear regression)"
        )

        train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
        id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
        moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
        extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
        random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
        compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

        _model_results = copy.deepcopy(_results_base)
        _model_results['model'] = 'True-HeckmanDG-1'
        _model_results['loss'] = train_history['outcome/loss'][-1].item()

        _model_results['beta_hat_0'] = trainer.outcome_model.linear.bias.data[0].item()
        for j in range(args.input_features):
            _model_results[f'beta_hat_{j+1}'] = trainer.outcome_model.linear.weight.data[0][j].item()
        for k in range(len(trainer.selection_models)):
            _model_results[f'alpha_hat_0_{k}'] = trainer.selection_models[k].linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'alpha_hat_{j+1}_{k}'] = trainer.selection_models[k].linear.weight.data[0][j].item()

        _model_results['sigma_hat'] = train_history['outcome/sigma'][-1].item()
        _model_results['rho_hat'] = train_history['outcome/rho'][-1].item()
        _model_results['beta_lambda'] = None

        _model_results['selection/accuracy'] = train_history['selection/accuracy'][-1].item()
        _model_results['selection/precision'] = train_history['selection/precision'][-1].item()
        _model_results['selection/recall'] = train_history['selection/recall'][-1].item()
        _model_results['selection/brier_score'] = train_history['selection/brier_score'][-1].item()

        _model_results['train_mae'] = train_metrics['mae']
        _model_results['id_test_mae'] = id_test_metrics['mae']
        _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
        _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
        _model_results['random_test_mae'] = random_test_metrics['mae']
        _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

        _model_results['train_mse'] = train_metrics['mse']
        _model_results['id_test_mse'] = id_test_metrics['mse']
        _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
        _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
        _model_results['random_test_mse'] = random_test_metrics['mse']
        _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

        _model_results['train_mape'] = train_metrics['mape']
        _model_results['id_test_mape'] = id_test_metrics['mape']
        _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
        _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
        _model_results['random_test_mape'] = random_test_metrics['mape']
        _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

        _model_results['train_r2'] = train_metrics['r2']
        _model_results['id_test_r2'] = id_test_metrics['r2']
        _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
        _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
        _model_results['random_test_r2'] = random_test_metrics['r2']
        _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

        results.append(_model_results); del _model_results

        """
            D-2) True Heckman DG-2.
        """
        trainer = TrueHeckmanDGRegression(
            train_domains=args.num_train_domains,
            input_features=args.input_features,
            method='two-step',
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device
        )

        train_history: dict = trainer.fit(
            labeled_set=dataset,
            unlabeled_set=compliment_test_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            unlabeled_batch_size=args.unlabeled_batch_size,
            description="True HeckmanDG-2 (linear regression)"
        )

        train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
        id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
        moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
        extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
        random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
        compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

        _model_results = copy.deepcopy(_results_base)
        _model_results['model'] = 'True-HeckmanDG-2'
        _model_results['loss'] = train_history['outcome/loss'][-1].item()

        _model_results['beta_hat_0'] = trainer.outcome_model.linear.bias.data[0].item()
        for j in range(args.input_features):
            _model_results[f'beta_hat_{j+1}'] = trainer.outcome_model.linear.weight.data[0][j].item()
        for k in range(len(trainer.selection_models)):
            _model_results[f'alpha_hat_0_{k}'] = trainer.selection_models[k].linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'alpha_hat_{j+1}_{k}'] = trainer.selection_models[k].linear.weight.data[0][j].item()

        _model_results['sigma_hat'] = None
        _model_results['rho_hat'] = None
        _model_results['beta_lambda'] = train_history['outcome/beta_lambda'][-1].item()

        _model_results['selection/accuracy'] = train_history['selection/accuracy'][-1].item()
        _model_results['selection/precision'] = train_history['selection/precision'][-1].item()
        _model_results['selection/recall'] = train_history['selection/recall'][-1].item()
        _model_results['selection/brier_score'] = train_history['selection/brier_score'][-1].item()

        _model_results['train_mae'] = train_metrics['mae']
        _model_results['id_test_mae'] = id_test_metrics['mae']
        _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
        _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
        _model_results['random_test_mae'] = random_test_metrics['mae']
        _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

        _model_results['train_mse'] = train_metrics['mse']
        _model_results['id_test_mse'] = id_test_metrics['mse']
        _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
        _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
        _model_results['random_test_mse'] = random_test_metrics['mse']
        _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

        _model_results['train_mape'] = train_metrics['mape']
        _model_results['id_test_mape'] = id_test_metrics['mape']
        _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
        _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
        _model_results['random_test_mape'] = random_test_metrics['mape']
        _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

        _model_results['train_r2'] = train_metrics['r2']
        _model_results['id_test_r2'] = id_test_metrics['r2']
        _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
        _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
        _model_results['random_test_r2'] = random_test_metrics['r2']
        _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

        results.append(_model_results); del _model_results

        """
            E-1) Safe Domain-Aware Heckman Correction w/ Joint Maximum Likelihood Estimation.
                Uses out-of-distribution data only as unobserved data.
        """
        trainer = SafeHeckmanDGRegression(
            train_domains=args.num_train_domains,
            input_features=args.input_features,
            method='mle',
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        )

        train_history: dict = trainer.fit(
            labeled_set=dataset,
            unlabeled_set=compliment_test_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            unlabeled_batch_size=args.unlabeled_batch_size,
            description="Safe-HeckmanDG-1 (linear regression) ",
        )

        train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
        id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
        moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
        extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
        random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
        compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

        _model_results = copy.deepcopy(_results_base)
        _model_results['model'] = 'Safe-HeckmanDG-1'
        _model_results['loss'] = train_history['outcome/loss'][-1].item()

        _model_results['beta_hat_0'] = trainer.outcome_model.linear.bias.data[0].item()
        for j in range(args.input_features):
            _model_results[f'beta_hat_{j+1}'] = trainer.outcome_model.linear.weight.data[0][j].item()
        for k in range(len(trainer.selection_models)):
            _model_results[f'alpha_hat_0_{k}'] = trainer.selection_models[k].linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'alpha_hat_{j+1}_{k}'] = trainer.selection_models[k].linear.weight.data[0][j].item()

        _model_results['sigma_hat'] = train_history['outcome/sigma'][-1].item()
        _model_results['rho_hat'] = train_history['outcome/rho'][-1].item()
        _model_results['beta_lambda'] = None

        _model_results['selection/accuracy'] = train_history['selection/accuracy'][-1].item()
        _model_results['selection/precision'] = train_history['selection/precision'][-1].item()
        _model_results['selection/recall'] = train_history['selection/recall'][-1].item()
        _model_results['selection/brier_score'] = train_history['selection/brier_score'][-1].item()

        _model_results['train_mae'] = train_metrics['mae']
        _model_results['id_test_mae'] = id_test_metrics['mae']
        _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
        _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
        _model_results['random_test_mae'] = random_test_metrics['mae']
        _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

        _model_results['train_mse'] = train_metrics['mse']
        _model_results['id_test_mse'] = id_test_metrics['mse']
        _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
        _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
        _model_results['random_test_mse'] = random_test_metrics['mse']
        _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

        _model_results['train_mape'] = train_metrics['mape']
        _model_results['id_test_mape'] = id_test_metrics['mape']
        _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
        _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
        _model_results['random_test_mape'] = random_test_metrics['mape']
        _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

        _model_results['train_r2'] = train_metrics['r2']
        _model_results['id_test_r2'] = id_test_metrics['r2']
        _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
        _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
        _model_results['random_test_r2'] = random_test_metrics['r2']
        _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

        results.append(_model_results); del _model_results

        """
        E-2) Safe Domain-Aware Heckman Correction w/ Two-Step Estimation.
             Uses out-of-distribution data only as unobserved data.
        """
        trainer = SafeHeckmanDGRegression(
            train_domains=args.num_train_domains,
            input_features=args.input_features,
            method='two-step',
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        )

        train_history: dict = trainer.fit(
            labeled_set=dataset,
            unlabeled_set=compliment_test_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            unlabeled_batch_size=args.unlabeled_batch_size,
            description="Safe-HeckmanDG-2 (linear regression) ",
        )

        train_metrics: dict = trainer.evaluate(dataset, batch_size=args.batch_size)
        id_test_metrics: dict = trainer.evaluate(id_test_set, batch_size=None)
        moderate_ood_test_metrics: dict = trainer.evaluate(moderate_ood_test_set, batch_size=None)
        extreme_ood_test_metrics: dict = trainer.evaluate(extreme_ood_test_set, batch_size=None)
        random_test_metrics: dict = trainer.evaluate(random_test_set, batch_size=None)
        compliment_test_metrics: dict = trainer.evaluate(compliment_test_set, batch_size=None)

        _model_results = copy.deepcopy(_results_base)
        _model_results['model'] = 'Safe-HeckmanDG-2'
        _model_results['loss'] = train_history['outcome/loss'][-1].item()

        _model_results['beta_hat_0'] = trainer.outcome_model.linear.bias.data[0].item()
        for j in range(args.input_features):
            _model_results[f'beta_hat_{j+1}'] = trainer.outcome_model.linear.weight.data[0][j].item()
        for k in range(len(trainer.selection_models)):
            _model_results[f'alpha_hat_0_{k}'] = trainer.selection_models[k].linear.bias.data[0].item()
            for j in range(args.input_features):
                _model_results[f'alpha_hat_{j+1}_{k}'] = trainer.selection_models[k].linear.weight.data[0][j].item()

        _model_results['sigma_hat'] = None
        _model_results['rho_hat'] = None
        _model_results['beta_lambda'] = train_history['outcome/beta_lambda'][-1].item()

        _model_results['selection/accuracy'] = train_history['selection/accuracy'][-1].item()
        _model_results['selection/precision'] = train_history['selection/precision'][-1].item()
        _model_results['selection/recall'] = train_history['selection/recall'][-1].item()
        _model_results['selection/brier_score'] = train_history['selection/brier_score'][-1].item()

        _model_results['train_mae'] = train_metrics['mae']
        _model_results['id_test_mae'] = id_test_metrics['mae']
        _model_results['moderate_ood_test_mae'] = moderate_ood_test_metrics['mae']
        _model_results['extreme_ood_test_mae'] = extreme_ood_test_metrics['mae']
        _model_results['random_test_mae'] = random_test_metrics['mae']
        _model_results['compliment_test_mae'] = compliment_test_metrics['mae']

        _model_results['train_mse'] = train_metrics['mse']
        _model_results['id_test_mse'] = id_test_metrics['mse']
        _model_results['moderate_ood_test_mse'] = moderate_ood_test_metrics['mse']
        _model_results['extreme_ood_test_mse'] = extreme_ood_test_metrics['mse']
        _model_results['random_test_mse'] = random_test_metrics['mse']
        _model_results['compliment_test_mse'] = compliment_test_metrics['mse']

        _model_results['train_mape'] = train_metrics['mape']
        _model_results['id_test_mape'] = id_test_metrics['mape']
        _model_results['moderate_ood_test_mape'] = moderate_ood_test_metrics['mape']
        _model_results['extreme_ood_test_mape'] = extreme_ood_test_metrics['mape']
        _model_results['random_test_mape'] = random_test_metrics['mape']
        _model_results['compliment_test_mape'] = compliment_test_metrics['mape']

        _model_results['train_r2'] = train_metrics['r2']
        _model_results['id_test_r2'] = id_test_metrics['r2']
        _model_results['moderate_ood_test_r2'] = moderate_ood_test_metrics['r2']
        _model_results['extreme_ood_test_r2'] = extreme_ood_test_metrics['r2']
        _model_results['random_test_r2'] = random_test_metrics['r2']
        _model_results['compliment_test_r2'] = compliment_test_metrics['r2']

        results.append(_model_results); del _model_results


        """
        F-2) Positive-Unlabeled Domain-Aware Heckman Correction w/ Two-Step Estimation.
             Uses out-of-distribution data only as unobserved data.
        """

        # Write results to .csv file
        df = pd.DataFrame.from_dict(results)
        if os.path.exists(write_file):
            df.to_csv(write_file, mode='a', index=False, header=False)
        else:
            df.to_csv(write_file, mode='w', index=False, header=True)


if __name__ == '__main__':
    try:
        _ = main()
    except KeyboardInterrupt:
        pass
