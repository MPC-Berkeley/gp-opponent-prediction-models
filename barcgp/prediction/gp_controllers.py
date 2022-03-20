import array
import copy
import sys
from typing import Type, List

import time
import numpy as np
import torch
import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood
from matplotlib import pyplot as plt
from barcgp.common.utils.scenario_utils import SampleGenerator, Sample
from barcgp.common.pytypes import VehicleState, ParametricPose, VehicleActuation, VehiclePrediction
from barcgp.prediction.gpytorch_models import ExactGPModel, MultitaskGPModel, MultitaskGPModelApproximate, \
    IndependentMultitaskGPModelApproximate
from barcgp.prediction.abstract_gp_controller import GPController
from torch.utils.data import DataLoader
from tqdm import tqdm


class GPControllerExact(GPController):
    def __init__(self, sample_generator: SampleGenerator, model_class: Type[gpytorch.models.GP],
                 likelihood, input_size: int, output_size: int, enable_GPU=False):
        super().__init__(sample_generator, model_class, likelihood, input_size, output_size, enable_GPU)
        if self.model_class == ExactGPModel:
            self.independent = True
            likelihoods = []
            models = []
            for i in range(output_size):
                likelihoods.append(likelihood())
                models.append(ExactGPModel(self.train_x, self.train_y[:, i], likelihoods[i]))
            self.model = gpytorch.models.IndependentModelList(*models)
            self.likelihood = gpytorch.likelihoods.LikelihoodList(*[i.likelihood for i in models])  # model.likelihood?
        else:
            self.independent = False
            self.model = self.model_class(self.train_x, self.train_y, self.likelihood)

    def predict(self, input):
        '''if not self.independent:
            return super().predict(input)'''

        if self.model == None:
            raise Exception('X', 'XX')
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if self.independent:
                model_output = self.model(*([self.standardize(input)] * self.output_size))
                prediction = self.likelihood(*model_output)
            else:
                model_output = self.model(self.standardize(input))
                prediction = self.likelihood(model_output)
            mean = torch.zeros((1, self.output_size))  # [ego_state | tv_state]
            cov = np.zeros((self.output_size, self.output_size))  # [ego_state | tv_state]
        for i in range(len(prediction)):
            mean[0, i] = prediction[i].mean.cpu()
            cov[i, i] = np.sqrt((prediction[i].variance * (self.stds_y[0, i] ** 2)).detach().numpy())  # correct?
        retv = self.outputToReal(mean)
        return retv, cov

    def sample_gp(self, ego_state: VehicleState, target_state: VehicleState):
        test_x = torch.zeros((1, self.input_size))
        test_x[0] = self.state_to_tensor(ego_state, target_state)
        if self.enable_GPU:
            test_x = test_x.cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if self.independent:
                model_output = self.model(*([self.standardize(test_x) for _ in range(self.output_size)]))
                prediction = self.likelihood(*model_output)
                px = torch.distributions.Normal(torch.FloatTensor([a.mean for a in prediction]),
                                                torch.FloatTensor([a.stddev for a in model_output]))
                if self.enable_GPU:
                    sampled = px.sample().cuda()
                sampled = sampled[None, :]
            else:
                model_output = self.model(self.standardize(test_x))
                prediction = self.likelihood(model_output)
                px = torch.distributions.Normal(prediction.mean, model_output.stddev)
                sampled = px.sample().cpu()
        return self.outputToReal(sampled)

    def evaluate(self):
        if not self.independent:
            super().evaluate()
        else:
            self.set_evaluation_mode()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # This contains predictions for both outcomes as a list
                predictions = self.likelihood(*self.model(*([self.test_x] * self.output_size)))
            f, axs = plt.subplots(self.output_size, 1, figsize=(15, 10))
            i = 0
            titles = ['s', 'x_tran', 'e_psi', 'v_long', 'w_psi']
            for submodel, prediction, ax, title in zip(self.model.models, predictions, axs, titles):
                mean = self.stds_y[0, i] * prediction.mean + self.means_y[0, i]
                # mean =  prediction.mean
                mean = mean.cpu().detach().numpy()
                cov = np.sqrt((prediction.variance.cpu() * (self.stds_y[0, i] ** 2)).detach().numpy())
                '''lower, upper = prediction.confidence_region()
                lower = lower.detach().numpy()
                upper = upper.detach().numpy()'''
                lower = mean - 2 * cov
                upper = mean + 2 * cov
                tr_y = self.stds_y[0, i] * self.test_y[:, i] + self.means_y[0, i]
                # tr_y = self.test_y[:, i]
                i += 1
                # Plot training data as black stars
                ax.plot(tr_y, 'k*')
                # Predictive mean as blue line
                ax.plot(mean, 'b')
                # Shade in confidence
                ax.fill_between(np.arange(len(mean)), lower, upper, alpha=0.5)
                ax.legend(['Observed Data', 'Mean', 'Confidence'])
                ax.set_title(title)
            plt.show()

    def train(self):
        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.test_x = self.test_x.cuda()

        # Find optimal model hyper-parameters
        self.model.train()
        self.likelihood.train()

        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)  # Includes GaussianLikelihood parameters

        # GP marginal log likelihood
        # p(y | x, X, Y) = ∫p(y|x, f)p(f|X, Y)df
        if self.independent:
            mll = SumMarginalLogLikelihood(self.likelihood, self.model)
        else:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        last_loss = 20
        i = 0
        not_done = True
        j = 0
        best_model = None
        best_likelihood = None
        while not_done:
            not_done=False
            self.model.train()
            self.likelihood.train()
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            if self.independent:
                output = self.model(*self.model.train_inputs)
                # Calc loss and backprop gradients
                loss = -mll(output, *self.model.train_targets)
            else:
                output = self.model(self.train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            self.model.eval()
            self.likelihood.eval()
            if self.independent:
                output_val = self.model(*([(self.test_x,) for i in range(self.output_size)]))
                loss_val = -mll(output_val, *([self.test_y[:, i] for i in range(self.output_size)]))
            if round(loss_val.item(), 3) >= last_loss:
                # train until loss is not improving
                best_model = copy.copy(self.model)
                best_likelihood = copy.copy(self.likelihood)
                j += 1
                if j > 20:
                    not_done = False
            else:
                j = 0
                last_loss = round(loss_val.item(), 3)
            print('Iter %d - Loss: %.3f -  Val-Loss: %.3f' % (i + 1, loss.item(), loss_val.item()))
            i += 1
        self.model = best_model
        self.likelihood = best_likelihood
        for param_name, param in self.model.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.item()}')

class GPControllerApproximate(GPController):
    def __init__(self, sample_generator: SampleGenerator, model_class: Type[gpytorch.models.GP],
                 likelihood: gpytorch.likelihoods.Likelihood, input_size: int, output_size: int, inducing_points: int,
                 enable_GPU=False):
        super().__init__(sample_generator, model_class, likelihood, input_size, output_size, enable_GPU)
        if self.model_class == IndependentMultitaskGPModelApproximate:
            self.model = IndependentMultitaskGPModelApproximate(inducing_points_num=inducing_points,
                                                                input_dim=self.input_size,
                                                                num_tasks=self.output_size)  # Independent
            self.independent = True
        elif self.model_class == MultitaskGPModelApproximate:
            self.model = MultitaskGPModelApproximate(inducing_points_num=inducing_points, input_dim=self.input_size,
                                                     num_latents=16, num_tasks=self.output_size)  # Correlated
            self.independent = False
        else:
            raise ('Model not found!')

    def train(self):
        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.train_x = self.train_x.cuda()
            self.train_y = self.train_y.cuda()
            self.test_x = self.test_x.cuda()
            self.test_y = self.test_y.cuda()

        # Find optimal model hyper-parameters
        self.model.train()
        self.likelihood.train()

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.train_x), torch.tensor(self.train_y)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.test_x), torch.tensor(self.test_y)
        )
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=150 if self.enable_GPU else 100,
                                      shuffle=True,  # shuffle?
                                      num_workers=0 if self.enable_GPU else 8)
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=25,
                                      shuffle=False,  # shuffle?
                                      num_workers=0 if self.enable_GPU else 8)
        # Use the Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.005)  # Includes GaussianLikelihood parameters

        # GP marginal log likelihood
        # p(y | x, X, Y) = ∫p(y|x, f)p(f|X, Y)df
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.train_y.numel())

        epochs = 100
        last_loss = np.inf
        no_progress_epoch = 0
        not_done = True
        epoch = 0
        best_model = None
        best_likeli = None
        sys.setrecursionlimit(100000)
        while not_done:
        # for _ in range(epochs):
            train_dataloader = tqdm(train_dataloader)
            valid_dataloader = tqdm(valid_dataloader)
            train_loss = 0
            valid_loss = 0
            c_loss = 0
            for step, (train_x, train_y) in enumerate(train_dataloader):
                # Within each iteration, we will go over each minibatch of data
                optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                train_loss += loss.item()
                train_dataloader.set_postfix(log={'train_loss': f'{(train_loss / (step + 1)):.5f}'})
                loss.backward()
                optimizer.step()
            for step, (train_x, train_y) in enumerate(valid_dataloader):
                optimizer.zero_grad()
                output = self.model(train_x)
                loss = -mll(output, train_y)
                valid_loss += loss.item()
                c_loss = valid_loss / (step + 1)
                valid_dataloader.set_postfix(log={'valid_loss': f'{(c_loss):.5f}'})
            if c_loss > last_loss:
                if no_progress_epoch >= 15:
                    not_done = False
            else:
                best_model = copy.copy(self.model)
                best_likeli = copy.copy(self.likelihood)
                last_loss = c_loss
                no_progress_epoch = 0

            no_progress_epoch += 1
        self.model = best_model
        self.likelihood = best_likeli

    def evaluate(self):
        if not self.independent:
            super().evaluate()
        else:
            self.set_evaluation_mode()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # This contains predictions for both outcomes as a list
                predictions = self.likelihood(self.likelihood(self.model(self.test_x[:50])))

            mean = predictions.mean.cpu()
            variance = predictions.variance.cpu()
            self.means_x = self.means_x.cpu()
            self.means_y = self.means_y.cpu()
            self.stds_x = self.stds_x.cpu()
            self.stds_y = self.stds_y.cpu()
            self.test_y = self.test_y.cpu()

            f, ax = plt.subplots(self.output_size, 1, figsize=(15, 10))
            titles = ['s', 'x_tran', 'e_psi', 'v_long', 'w_psi']
            for i in range(self.output_size):
                unnormalized_mean = self.stds_y[0, i] * mean[:, i] + self.means_y[0, i]
                unnormalized_mean = unnormalized_mean.detach().numpy()

                cov = np.sqrt((variance[:, i] * (self.stds_y[0, i] ** 2)))
                cov = cov.detach().numpy()
                '''lower, upper = prediction.confidence_region()
                lower = lower.detach().numpy()
                upper = upper.detach().numpy()'''
                lower = unnormalized_mean - 2 * cov
                upper = unnormalized_mean + 2 * cov
                tr_y = self.stds_y[0, i] * self.test_y[:50, i] + self.means_y[0, i]
                # Plot training data as black stars
                ax[i].plot(tr_y, 'k*')
                # Predictive mean as blue line
                # ax[i].scatter(np.arange(len(unnormalized_mean)), unnormalized_mean)
                ax[i].errorbar(np.arange(len(unnormalized_mean)), unnormalized_mean, yerr=cov, fmt="o", markersize=4, capsize=8)
                # Shade in confidence
                # ax[i].fill_between(np.arange(len(unnormalized_mean)), lower, upper, alpha=0.5)
                ax[i].legend(['Observed Data', 'Predicted Data'])
                ax[i].set_title(titles[i])
            plt.show()


class GPControllerTrained(GPController):
    def __init__(self, name, enable_GPU, model=None):
        if model is not None:
            self.load_model_from_object(model)
        else:
            self.load_model(name)
        self.enable_GPU = enable_GPU
        if self.enable_GPU:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.means_x = self.means_x.cuda()
            self.means_y = self.means_y.cuda()
            self.stds_x = self.stds_x.cuda()
            self.stds_y = self.stds_y.cuda()
        else:
            self.model.cpu()
            self.likelihood.cpu()
            self.means_x = self.means_x.cpu()
            self.means_y = self.means_y.cpu()
            self.stds_x = self.stds_x.cpu()
            self.stds_y = self.stds_y.cpu()