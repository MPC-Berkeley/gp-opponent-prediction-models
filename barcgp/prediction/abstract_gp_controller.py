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
from abc import abstractmethod, ABC
from tqdm import tqdm
from torch.utils.data import DataLoader
from barcgp.common.utils.file_utils import *
from barcgp.common.utils.scenario_utils import SampleGenerator, Sample
from barcgp.common.pytypes import VehicleState, ParametricPose, VehicleActuation, VehiclePrediction
from barcgp.common.tracks.radius_arclength_track import RadiusArclengthTrack


class GPController(ABC):
    '''
    Abstract GP Class the reads samples for training and makes predictions based
    on a trained model. Always instantiate GPControllers of inheriting classes instead
    of this.
    '''

    def __init__(self, sample_generator: SampleGenerator,
                 model_class: Type[gpytorch.models.GP],
                 likelihood: gpytorch.likelihoods.Likelihood,
                 input_size: int,
                 output_size: int,
                 enable_GPU=False):
        self.sample_generator = sample_generator
        self.model_class = model_class  # uninitialized model
        self.likelihood = likelihood
        self.input_size = input_size
        self.output_size = output_size
        self.enable_GPU = enable_GPU
        self.model = None
        self.independent = False
        self.means_x = None
        self.means_y = None
        self.stds_x = None
        self.stds_y = None
        self.pull_samples()  # will initialize model

    def pull_samples(self, holdout=150):
        self.train_x = torch.zeros((self.sample_generator.getNumSamples() - holdout, self.input_size))  # [ego_state | tv_state]
        self.test_x = torch.zeros((holdout, self.input_size))  # [ego_state | tv_state]
        self.train_y = torch.zeros([self.sample_generator.getNumSamples() - holdout, self.output_size])  # [tv_actuation]
        self.test_y = torch.zeros([holdout, self.output_size])  # [tv_actuation]

        # Sampling should be done on CPU
        self.train_x = self.train_x.cpu()
        self.test_x = self.test_x.cpu()
        self.train_y = self.train_y.cpu()
        self.test_y = self.test_y.cpu()

        not_done = True
        sample_idx = 0
        while not_done:
            samp = self.sample_generator.nextSample()
            if samp is not None:
                x_n, y_n = self.convert_to_tensor(samp)
                if sample_idx < holdout:
                    self.test_x[sample_idx] = x_n
                    self.test_y[sample_idx] = y_n
                else:
                    self.train_x[sample_idx - holdout] = x_n
                    self.train_y[sample_idx - holdout] = y_n
                sample_idx += 1
            else:
                print('Finished')
                not_done = False

        self.means_x = self.train_x.mean(dim=0, keepdim=True)
        self.stds_x = self.train_x.std(dim=0, keepdim=True)

        for i in range(self.stds_x.shape[1]):
            if self.stds_x[0, i] == 0:
                self.stds_x[0, i] = 1
        self.train_x = (self.train_x - self.means_x) / self.stds_x
        self.test_x = (self.test_x - self.means_x) / self.stds_x

        self.means_y = self.train_y.mean(dim=0, keepdim=True)
        self.stds_y = self.train_y.std(dim=0, keepdim=True)
        for i in range(self.stds_y.shape[1]):
            if self.stds_y[0, i] == 0:
                self.stds_y[0, i] = 1
        self.train_y = (self.train_y - self.means_y) / self.stds_y
        self.test_y = (self.test_y - self.means_y) / self.stds_y
        print(f"train_x shape: {self.train_x.shape}")
        print(f"train_y shape: {self.train_y.shape}")

    def train(self):
        pass

    def evaluate(self):
        self.set_evaluation_mode()

        f, ax = plt.subplots(4, 1, figsize=(4, 3))

        # make predictions using the same test points
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(self.test_x))

        mean = self.outputToReal(predictions.mean)
        # Get upper and lower confidence bounds
        lower, upper = predictions.confidence_region()
        lower = self.outputToReal(lower)
        upper = self.outputToReal(upper)
        print(mean)
        print(self.test_y)
        # Plot training data as black stars
        ax[0].plot(mean.numpy()[:, 0], 'k*')
        ax[0].plot(self.test_y.numpy()[:, 0], 'b')
        ax[1].plot(mean.numpy()[:, 1], 'k*')
        ax[1].plot(self.test_y.numpy()[:, 1], 'b')
        ax[2].plot(mean.numpy()[:, 2], 'k*')
        ax[2].plot(self.test_y.numpy()[:, 2], 'b')
        ax[3].plot(mean.numpy()[:, 3], 'k*')
        ax[3].plot(self.test_y.numpy()[:, 3], 'b')
        # Plot predictive means as blue line
        '''ax.plot(test_x.numpy(), mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])'''
        plt.show()

    def get_single_prediction(self, ego_state: VehicleState, target_state: VehicleState):
        test_x = torch.zeros((1, self.input_size))
        test_x[0] = self.state_to_tensor(ego_state, target_state)
        if self.enable_GPU:
            test_x = test_x.cuda()
        return self.predict(test_x)

    def standardize(self, input_):
        if self.means_x is not None:
            return (input_ - self.means_x) / self.stds_x
        else:
            return input_

    def outputToReal(self, output):
        if self.means_y is not None:
            return output * self.stds_y + self.means_y
        else:
            return output

    def predict(self, input):
        if self.model == None:
            raise Exception('X', 'XX')
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model_output = self.model(self.standardize(input))
            prediction = self.likelihood(model_output)
        if self.enable_GPU:
            return self.outputToReal(prediction.mean.cuda()), model_output.covariance_matrix.cuda() * (
                np.diag(np.power(self.stds_y[0], 2)))
        else:
            return self.outputToReal(prediction.mean.cpu()), model_output.covariance_matrix.detach().numpy() * (
                np.diag(np.power(self.stds_y[0], 2)))

    def get_full_prediction(self, ego_state: VehicleState, target_state: VehicleState,
                            ego_prediction: VehiclePrediction, track: RadiusArclengthTrack):
        """
        get an estimate of the N-step prediction of the target vehicle. Uses mean and covariance
        of to predict and re-feeds that to evolve a full trajectory. Doesn't account for
        uncertainty propagation through follow-up states
        input:
            ego_state: VehicleState of ego vehicle
            target_state: VehicleState of tar vehicle (the one that is to be predicted)
            ego_prediction: VehiclePrediction, prediction of ego vehicle, used to get
                                tar_prediction
            track: Simulation track, needed to get track curvature at future predicted states
        """

        self.set_evaluation_mode()
        target_prediction = VehiclePrediction()
        target_prediction.s = []
        target_prediction.x_tran = []
        target_prediction.e_psi = []
        target_prediction.v_long = []
        target_prediction.sey_cov = []
        horizon = len(ego_prediction.x)
        evolving_state = target_state.copy()
        lookahead_ego = ego_state.copy()
        target_prediction.s.append(evolving_state.p.s)
        target_prediction.x_tran.append(evolving_state.p.x_tran)
        target_prediction.e_psi.append(evolving_state.p.e_psi)
        target_prediction.v_long.append(evolving_state.v.v_long)
        target_prediction.sey_cov.append(np.zeros((2, 2)).flatten())
        for i in range(horizon - 1):
            next_pred, covariance = self.get_single_prediction(lookahead_ego, evolving_state)
            lookahead_ego.p.s = ego_prediction.s[i + 1]
            lookahead_ego.p.x_tran = ego_prediction.x_tran[i + 1]
            lookahead_ego.p.e_psi = ego_prediction.e_psi[i + 1]
            lookahead_ego.v.v_long = ego_prediction.v_long[i + 1]
            evolving_state.p.s += next_pred[0, 0].cpu().numpy()
            evolving_state.p.x_tran += next_pred[0, 1].cpu().numpy()
            evolving_state.p.e_psi += next_pred[0, 2].cpu().numpy()
            evolving_state.v.v_long += next_pred[0, 3].cpu().numpy()
            track.update_curvature(evolving_state)
            target_prediction.s.append(evolving_state.p.s)
            target_prediction.x_tran.append(evolving_state.p.x_tran)
            target_prediction.e_psi.append(evolving_state.p.e_psi)
            target_prediction.v_long.append(evolving_state.v.v_long)

            target_prediction.sey_cov.append(covariance[:2, :2].flatten())
        return target_prediction

    def sample_gp(self, ego_state: VehicleState, target_state: VehicleState):
        test_x = torch.zeros((1, self.input_size))
        test_x[0] = self.state_to_tensor(ego_state, target_state)
        if self.enable_GPU:
            test_x = test_x.cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model_output = self.model(self.standardize(test_x))
            prediction = self.likelihood(model_output)
            px = torch.distributions.Normal(prediction.mean, model_output.stddev)
            if self.enable_GPU:
                sampled = px.sample().cuda()
            else:
                sampled = px.sample()
            # sampled = sampled[None, :]
        return self.outputToReal(sampled)

    def sample_gp_par_vec(self, ego_state, target_states):
        """
        Samples the gp given multiple tv state predictions together with one ego_state for the same time-step.
        Input states are lists instead of vehicle state objects reducing the computational costs
        Inputs:
            ego_state: ego state for current time step
                [s, x_tran, e_psi, v_long]
            target_state: list of target states for current time step
                each target state: [s, x_tran, e_psi, v_long, w_psi, curv[0], curv[1], curv[2]]
        Outputs:
            return_sampled: sampled delta tv states for each input tv state
                each delta tv state: [ds, dx_tran, de_psi, dv_long, dw_psi]
        """
        test_x = torch.zeros((len(target_states), self.input_size))
        for i in range(len(target_states)):
            test_x[i] = self.state_to_tensor_vec(ego_state, target_states[i])
        if self.enable_GPU:
            test_x = test_x.cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.model(self.standardize(test_x))
            mean = prediction.mean
            stddev = prediction.stddev
            sampled = torch.distributions.Normal(mean, stddev).sample()
        return_sampled = self.outputToReal(sampled)
        return return_sampled, None

    def sample_gp_par_multi_vec(self, ego_states, target_states, n_hyp):
        """
        Samples the gp given multiple tv state predictions together with one ego_state for the same time-step.
        Input states are lists instead of vehicle state objects reducing the computational costs
        Inputs:
            ego_state: Hypothesis ego states for current time step
                [s, x_tran, e_psi, v_long]
            target_state: list of target states for current time step
                each target state: [s, x_tran, e_psi, v_long, w_psi, curv[0], curv[1], curv[2]]
        Outputs:
            return_sampled: sampled delta tv states for each input tv state
                each delta tv state: [ds, dx_tran, de_psi, dv_long, dw_psi]
        """
        test_x = torch.zeros((len(target_states), self.input_size))
        for i in range(int(len(target_states)/n_hyp)):
            for j in range(n_hyp):
                test_x[i*n_hyp + j] = self.state_to_tensor_vec(ego_states[j], target_states[i*n_hyp+j])
        if self.enable_GPU:
            test_x = test_x.cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model_output = self.model(self.standardize(test_x))
            prediction = self.likelihood(model_output)
            sampled = prediction.sample()
        return_sampled = self.outputToReal(sampled)
        return return_sampled, None

    def sample_gp_par(self, ego_states, target_states):
        test_x = torch.zeros((len(ego_states), self.input_size))
        for i in range(len(ego_states)):
            test_x[i] = self.state_to_tensor(ego_states[i], target_states[i])
        if self.enable_GPU:
            test_x = test_x.cuda()
        return_sampled = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model_output = self.model(self.standardize(test_x))
            prediction = self.likelihood(model_output)
            mean = prediction.mean
            # stddev = model_output.stddev
            # cov = model_output.covariance_matrix
            sampled = model_output.rsample()
            '''for i in range(len(ego_states)):
                #px = torch.distributions.Normal(mean[i], stddev[i])

                sampled = np.random.multivariate_normal(mean[i], cov[(i)*5:(i+1)*5, (i)*5:(i+1)*5], 1)
                if self.enable_GPU:
                    sampled = px.sample().cuda()
                else:
                    sampled = px.sample()
                return_sampled.append(self.outputToReal(torch.Tensor(sampled)).detach().numpy())'''
            return_sampled = self.outputToReal(sampled)
        # TODO rescale stddev?
        return return_sampled, model_output.covariance_matrix.detach().numpy()

    def xx(self, mu_x):
        a = self.likelihood(self.model(mu_x))
        t = torch.zeros((5, 2))
        self.mm = a.mean[0]
        t[:, 0] = self.mm
        c = a.covariance_matrix
        self.c = torch.zeros((5, 1))
        for i in range(5):
            self.c[i, 0] = c[i, i]
        t[:, 1] = self.c[:, 0]
        return t

    def xxx(self, mu_x):
        J = torch.autograd.functional.jacobian(self.xx, mu_x, create_graph=True)
        self.J = J
        return J[:, 1, 0, :]

    def get_predicted_dist(self, mu_x, sigma_x):
        """
        Get predicted distribution as described in
        https://www.researchgate.net/publication/2561895_Gaussian_Process_priors_with_Uncertain_Inputs_Multiple-Step-Ahead_Prediction
        Inputs:
            mu_x: Mean vector of random GP Input x
            sigma_x: Covariance matrix of random GP Input x TODO Make sure this is actually cov, not dev
        Outputs:
            m_x: Predicted mean of random GP Input x
            v_x: Predicted variance of random GP Input x
        """
        mu_x = self.standardize(mu_x).float()
        mu_x.requires_grad = True
        with gpytorch.settings.fast_pred_var():
            '''model_output = self.model(mu_x)
            prediction = self.likelihood(model_output)
            mean, cov = prediction.mean, model_output.covariance_matrix'''
            H = torch.autograd.functional.jacobian(self.xxx, mu_x)
            dmu_dx = self.J[:, 0, 0, :]
            dsigma2_ddx = H[:, :, 0, :]
        with torch.no_grad():
            m_x = self.mm
            v_x = torch.zeros((5, 1))
            # TOOO Speedup
            for i in range(5):
                v_x[i] = self.c[i, 0] + np.trace(
                    sigma_x[i, 0] * (0.5 * dsigma2_ddx[i, :, :] + dmu_dx[i, :] * dmu_dx[i, :]))
            return m_x.detach(), v_x.detach()

    def analytic_traj_pred(self, ego_state, target_state, ego_prediction, track):
        target_prediction = VehiclePrediction()
        target_prediction.s = []
        target_prediction.x_tran = []
        target_prediction.e_psi = []
        target_prediction.v_long = []
        target_prediction.sey_cov = []
        horizon = len(ego_prediction.x)
        evolving_state = target_state.copy()
        lookahead_ego = ego_state.copy()
        target_prediction.s.append(evolving_state.p.s)
        target_prediction.x_tran.append(evolving_state.p.x_tran)
        target_prediction.e_psi.append(evolving_state.p.e_psi)
        target_prediction.v_long.append(evolving_state.v.v_long)
        sigma_x = torch.zeros((5, 1))
        cov_scale = np.power(self.stds_y, 2)
        for i in range(horizon - 1):
            mean, cov = self.get_predicted_dist(self.state_to_tensor(lookahead_ego, evolving_state), sigma_x)
            mean = self.outputToReal(mean)
            sigma_x = cov
            cov = cov[:, 0] * cov_scale[:, 0]
            lookahead_ego.p.s = ego_prediction.s[i + 1]
            lookahead_ego.p.x_tran = ego_prediction.x_tran[i + 1]
            lookahead_ego.p.e_psi = ego_prediction.e_psi[i + 1]
            lookahead_ego.v.v_long = ego_prediction.v_long[i + 1]

            evolving_state.p.s += mean[0, 0].cpu().numpy()
            evolving_state.p.x_tran += mean[0, 1].cpu().numpy()
            evolving_state.p.e_psi += mean[0, 2].cpu().numpy()
            evolving_state.v.v_long += mean[0, 3].cpu().numpy()
            track.update_curvature(evolving_state)
            target_prediction.s.append(evolving_state.p.s)
            target_prediction.x_tran.append(evolving_state.p.x_tran)
            target_prediction.e_psi.append(evolving_state.p.e_psi)
            target_prediction.v_long.append(evolving_state.v.v_long)
            target_prediction.sey_cov.append(
                np.array([[cov[0], 0], [0, cov[1]]])[:2, :2].flatten())
            print(target_prediction.sey_cov[-1])
        return target_prediction

    def get_analytic_pred(self, ego_state: VehicleState, target_state: VehicleState,
                          ego_prediction: VehiclePrediction, track: RadiusArclengthTrack):
        # Set GP model to eval-mode
        self.set_evaluation_mode()
        preds = self.analytic_traj_pred(ego_state, target_state, ego_prediction, track)
        return preds

    def predict_gp_par(self, ego_states, target_states):
        test_x = torch.zeros((len(ego_states), self.input_size))
        for i in range(len(ego_states)):
            test_x[i] = self.state_to_tensor(ego_states[i], target_states[i])
        if self.enable_GPU:
            test_x = test_x.cuda()
        means = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model_output = self.model(self.standardize(test_x))
            prediction = self.likelihood(model_output)
            mean = prediction.mean
        if len(ego_states) == 1:
            return [self.outputToReal(mean).detach().numpy()], prediction.covariance_matrix.detach().numpy()
        for i in range(len(ego_states)):
            means.append(self.outputToReal(mean[i]).detach().numpy())
        return means, prediction.covariance_matrix.detach().numpy()

    def UKF_traj_gp_par(self, ego_state: VehicleState, target_state: VehicleState,
                        ego_prediction: VehiclePrediction, track: RadiusArclengthTrack):
        # TODO Make sure uncertainty is correctly propagated, SEE PAPER --> INCOMPLETE
        # Include Observation covariance
        M_ = 5
        M = 2 * M_ + 1
        alpha = 1.2
        beta = 1.2
        kappa = 0
        target_prediction = VehiclePrediction()
        target_prediction.s = []
        target_prediction.x_tran = []
        target_prediction.e_psi = []
        target_prediction.v_long = []
        target_prediction.sey_cov = []
        horizon = len(ego_prediction.x)

        evolving_states = [None] * M
        sigma_points = [None] * M
        lookahead_egos = [None] * M

        for i in range(M):
            evolving_states[i] = target_state.copy()
            lookahead_egos[i] = ego_state.copy()

        # means
        target_prediction.s.append(target_state.p.s)
        target_prediction.x_tran.append(target_state.p.x_tran)
        target_prediction.e_psi.append(target_state.e.psi)
        target_prediction.v_long.append(target_state.v.v_long)
        target_prediction.sey_cov.append(np.array([[0, 0], [0, 0]])[:2, :2].flatten())

        lamb = alpha ** 2 * (M_ + kappa) - M_
        w_a = [0] * M  # list of first order weights
        w_c = [0] * M  # list of second order weights
        w_a[0] = lamb / (M_ + lamb)
        w_c[0] = w_a[0] + 1 - alpha ** 2 + beta
        pred, covs = self.predict_gp_par([ego_state.copy()], [target_state.copy()])
        pred = pred[0]
        covs = np.diag(self.stds_y[0]) * covs * np.diag(self.stds_y[0])  # rescale covariance
        target_prediction.s.append(target_prediction.s[-1] + pred[0, 0])
        target_prediction.x_tran.append(target_prediction.x_tran[-1] + pred[0, 1])
        target_prediction.e_psi.append(target_prediction.e_psi[-1] + pred[0, 2])
        target_prediction.v_long.append(target_prediction.v_long[-1] + pred[0, 3])
        target_prediction.sey_cov.append(
            np.array([[covs[0, 0], 0], [0, covs[1, 1]]])[:2, :2].flatten())  # TODO square to get cov

        A = np.linalg.cholesky(covs)  # TODO replace by sqrt

        sigma_points[0] = pred
        for i in range(1, M_ + 1):
            sigma_points[i] = sigma_points[0] + np.sqrt(M_ + lamb) * A[:, i - 1]
            sigma_points[i + M_] = sigma_points[0] - np.sqrt(M_ + lamb) * A[:, i - 1]

            lookahead_egos[i].p.s = ego_prediction.s[1]
            lookahead_egos[i].p.x_tran = ego_prediction.x_tran[1]
            lookahead_egos[i].p.e_psi = ego_prediction.e_psi[1]
            lookahead_egos[i].v.v_long = ego_prediction.v_long[1]

            evolving_states[i].p.s = evolving_states[0].p.s + sigma_points[i][0, 0]
            evolving_states[i].p.x_tran = evolving_states[0].p.x_tran + sigma_points[i][0, 1]
            evolving_states[i].p.e_psi = evolving_states[0].p.e_psi + sigma_points[i][0, 2]
            evolving_states[i].v.v_long = evolving_states[0].v.v_long + sigma_points[i][0, 3]
            evolving_states[i].w.w_psi = evolving_states[0].w.w_psi + sigma_points[i][0, 4]

            evolving_states[i + M_].p.s = evolving_states[0].p.s + sigma_points[i + M_][0, 0]
            evolving_states[i + M_].p.x_tran = evolving_states[0].p.x_tran + sigma_points[i + M_][0, 1]
            evolving_states[i + M_].p.e_psi = evolving_states[0].p.e_psi + sigma_points[i + M_][0, 2]
            evolving_states[i + M_].v.v_long = evolving_states[0].v.v_long + sigma_points[i + M_][0, 3]
            evolving_states[i + M_].w.w_psi = evolving_states[0].w.w_psi + sigma_points[i + M_][0, 4]

            track.update_curvature(evolving_states[i])
            track.update_curvature(evolving_states[i + M_])

            w_a[i] = 1 / (2 * (M_ + lamb))
            w_a[i + M_] = w_a[i]
            w_c[i + M_] = w_a[i]
            w_c[i] = w_a[i]

        evolving_states[0].p.s += sigma_points[0][0, 0]
        evolving_states[0].p.x_tran += sigma_points[0][0, 1]
        evolving_states[0].p.e_psi += sigma_points[0][0, 2]
        evolving_states[0].v.v_long += sigma_points[0][0, 3]
        evolving_states[0].w.w_psi += sigma_points[0][0, 4]
        ysm = np.power(np.diag(self.stds_y[0]), 2)
        for i in range(horizon - 1):
            mean_dev = covs[0:5, 0:5] @ ysm
            next_preds, covs = self.predict_gp_par(lookahead_egos, evolving_states)
            diffs = []
            m_l = None
            d_l = None
            for k in range(M):
                diff = next_preds[k] + np.array(
                    [evolving_states[k].p.s, evolving_states[k].p.x_tran, evolving_states[k].p.e_psi,
                     evolving_states[k].v.v_long, evolving_states[k].w.w_psi])
                if m_l is None:
                    m_l = w_a[k] * diff
                else:
                    m_l += (w_a[k] * diff)
                diffs.append(diff)
            for k in range(M):
                d_t = diffs[k] - m_l
                if d_l is None:
                    d_l = w_c[k] * d_t.transpose() @ d_t
                else:
                    d_l += w_c[k] * d_t.transpose() @ d_t
            pred = m_l
            dev = d_l
            # dev = np.sum([w_c[k] * (
            #             (next_preds[k] - pred).transpose() @ (next_preds[k] - pred)) for k
            #               in range(M)], axis=0) + mean_dev
            target_prediction.s.append(pred[0, 0])
            target_prediction.x_tran.append(pred[0, 1])
            target_prediction.e_psi.append(pred[0, 2])
            target_prediction.v_long.append(pred[0, 3])
            target_prediction.sey_cov.append(np.array([[dev[0][0], 0], [0, dev[1][1]]])[:2, :2].flatten())
            A = np.linalg.cholesky(dev)
            sigma_points[0] = pred
            for j in range(1, M_ + 1):
                sigma_points[j] = sigma_points[0] + np.sqrt(M_ + lamb) * A[:, j - 1]
                sigma_points[j + M_] = sigma_points[0] - np.sqrt(M_ + lamb) * A[:, j - 1]

                lookahead_egos[j].p.s = ego_prediction.s[i + 1]
                lookahead_egos[j].p.x_tran = ego_prediction.x_tran[i + 1]
                lookahead_egos[j].p.e_psi = ego_prediction.e_psi[i + 1]
                lookahead_egos[j].v.v_long = ego_prediction.v_long[i + 1]

                evolving_states[j].p.s = sigma_points[j][0, 0]
                evolving_states[j].p.x_tran = sigma_points[j][0, 1]
                evolving_states[j].p.e_psi = sigma_points[j][0, 2]
                evolving_states[j].v.v_long = sigma_points[j][0, 3]
                evolving_states[j].w.w_psi = sigma_points[j][0, 4]

                evolving_states[j + M_].p.s = sigma_points[j + M_][0, 0]
                evolving_states[j + M_].p.x_tran = sigma_points[j + M_][0, 1]
                evolving_states[j + M_].p.e_psi = sigma_points[j + M_][0, 2]
                evolving_states[j + M_].v.v_long = sigma_points[j + M_][0, 3]
                evolving_states[j + M_].w.w_psi = sigma_points[j + M_][0, 4]

                track.update_curvature(evolving_states[j])
                track.update_curvature(evolving_states[j + M_])

            evolving_states[0].p.s = sigma_points[0][0, 0]
            evolving_states[0].p.x_tran = sigma_points[0][0, 1]
            evolving_states[0].p.e_psi = sigma_points[0][0, 2]
            evolving_states[0].v.v_long = sigma_points[0][0, 3]
            evolving_states[0].w.w_psi = sigma_points[0][0, 4]

        return target_prediction

    def sample_traj_gp_par(self, ego_state: VehicleState, target_state: VehicleState,
                           ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M):

        target_prediction = VehiclePrediction()
        target_prediction.s = []
        target_prediction.x_tran = []
        target_prediction.e_psi = []
        target_prediction.v_long = []
        tar_preds = []
        horizon = len(ego_prediction.x)
        evolving_states = []
        dummy_state = target_state.copy()
        target_prediction.s.append(target_state.p.s)
        target_prediction.x_tran.append(target_state.p.x_tran)
        target_prediction.e_psi.append(target_state.p.e_psi)
        target_prediction.v_long.append(target_state.v.v_long)
        # TODO instead of feeding in M similar initial states, feed in once and sample M times, for e.g.
        #           M = 50 samples, this will yield 451 instead of 500 gp samples
        for i in range(M):
            evolving_states.append(np.array(
                [target_state.p.s, target_state.p.x_tran, target_state.p.e_psi, target_state.v.v_long,
                 target_state.w.w_psi,
                 target_state.lookahead.curvature[0], target_state.lookahead.curvature[1],
                 target_state.lookahead.curvature[2]]))
            tar_preds.append(target_prediction.copy())

        lookahead_egos = np.array([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.v.v_long])

        for i in range(horizon - 1):
            next_pred, _ = self.sample_gp_par_vec(lookahead_egos, evolving_states)
            # get future ego state
            lookahead_egos[0] = ego_prediction.s[i + 1]
            lookahead_egos[1] = ego_prediction.x_tran[i + 1]
            lookahead_egos[2] = ego_prediction.e_psi[i + 1]
            lookahead_egos[3] = ego_prediction.v_long[i + 1]
            for j in range(M):
                # prepare next inputs
                evolving_states[j][0] += next_pred[j][0].item()
                evolving_states[j][1] += next_pred[j][1].item()
                evolving_states[j][2] += next_pred[j][2].item()
                evolving_states[j][3] += next_pred[j][3].item()
                evolving_states[j][4] += next_pred[j][4].item()

                # need a dummy state to get next curvatures
                dummy_state.p.s = evolving_states[j][0]
                track.update_curvature(dummy_state)

                evolving_states[j][5] = dummy_state.lookahead.curvature[0]
                evolving_states[j][6] = dummy_state.lookahead.curvature[1]
                evolving_states[j][7] = dummy_state.lookahead.curvature[2]

                # get predictions for evaluation
                tar_preds[j].s.append(evolving_states[j][0])
                tar_preds[j].x_tran.append(evolving_states[j][1])
                tar_preds[j].e_psi.append(evolving_states[j][2])
                tar_preds[j].v_long.append(evolving_states[j][3])

        # TODO: had to convert to array format for ROS msg conversions, make more efficient
        for i in range(M):
            tar_preds[i].s = array.array('d', tar_preds[i].s)
            tar_preds[i].x_tran = array.array('d', tar_preds[i].x_tran)
            tar_preds[i].e_psi = array.array('d', tar_preds[i].e_psi)
            tar_preds[i].v_long = array.array('d', tar_preds[i].v_long)

        return tar_preds

    def sample_multi_traj_gp_par(self, ego_state: VehicleState, target_state: VehicleState,
                           ego_prediction: List[VehiclePrediction], track: RadiusArclengthTrack, M):
        n_hyp = len(ego_prediction)
        target_prediction = []
        for i in range(n_hyp):
            target_prediction.append(VehiclePrediction())
            target_prediction[-1].s = []
            target_prediction[-1].x_tran = []
            target_prediction[-1].e_psi = []
            target_prediction[-1].v_long = []
            target_prediction[-1].s.append(target_state.p.s)
            target_prediction[-1].x_tran.append(target_state.p.x_tran)
            target_prediction[-1].e_psi.append(target_state.p.e_psi)
            target_prediction[-1].v_long.append(target_state.v.v_long)
        tar_preds = []
        horizon = len(ego_prediction[0].x)
        evolving_states = []
        dummy_state = target_state.copy()

        # TODO instead of feeding in M similar initial states, feed in once and sample M times, for e.g.
        #           M = 50 samples, this will yield 451 instead of 500 gp samples
        for i in range(M):
            for j in range(n_hyp):
                evolving_states.append(np.array(
                    [target_state.p.s, target_state.p.x_tran, target_state.p.e_psi, target_state.v.v_long,
                     target_state.w.w_psi,
                     target_state.lookahead.curvature[0], target_state.lookahead.curvature[1],
                     target_state.lookahead.curvature[2]]))
                tar_preds.append(target_prediction[j].copy())

        lookahead_egos = [np.array([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.v.v_long]),
                          np.array([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.v.v_long]),
                          np.array([ego_state.p.s, ego_state.p.x_tran, ego_state.p.e_psi, ego_state.v.v_long])]

        for i in range(horizon - 1):
            next_pred, _ = self.sample_gp_par_multi_vec(lookahead_egos, evolving_states, n_hyp)
            # get future ego state
            for j in range(n_hyp):
                lookahead_egos[j][0] = ego_prediction[j].s[i + 1]
                lookahead_egos[j][1] = ego_prediction[j].x_tran[i + 1]
                lookahead_egos[j][2] = ego_prediction[j].e_psi[i + 1]
                lookahead_egos[j][3] = ego_prediction[j].v_long[i + 1]
            for j in range(M*n_hyp):
                # prepare next inputs
                evolving_states[j][0] += next_pred[j][0].item()
                evolving_states[j][1] += next_pred[j][1].item()
                evolving_states[j][2] += next_pred[j][2].item()
                evolving_states[j][3] += next_pred[j][3].item()
                evolving_states[j][4] += next_pred[j][4].item()

                # need a dummy state to get next curvatures
                dummy_state.p.s = evolving_states[j][0]
                track.update_curvature(dummy_state)

                evolving_states[j][5] = dummy_state.lookahead.curvature[0]
                evolving_states[j][6] = dummy_state.lookahead.curvature[1]
                evolving_states[j][7] = dummy_state.lookahead.curvature[2]

                # get predictions for evaluation
                tar_preds[j].s.append(evolving_states[j][0])
                tar_preds[j].x_tran.append(evolving_states[j][1])
                tar_preds[j].e_psi.append(evolving_states[j][2])
                tar_preds[j].v_long.append(evolving_states[j][3])

        # TODO: had to convert to array format for ROS msg conversions, make more efficient
        for i in range(M*n_hyp):
            tar_preds[i].s = array.array('d', tar_preds[i].s)
            tar_preds[i].x_tran = array.array('d', tar_preds[i].x_tran)
            tar_preds[i].e_psi = array.array('d', tar_preds[i].e_psi)
            tar_preds[i].v_long = array.array('d', tar_preds[i].v_long)

        return tar_preds


    def mean_and_cov_from_list(self, l_pred: List[VehiclePrediction], M):
        """
        Extracts sample mean trajectory and covariance from list of VehiclePredictions
        """
        mean = l_pred[0].copy()
        mean.sey_cov = []
        for i in range(len(mean.s)):
            mean.s[i] = np.average([k.s[i] for k in l_pred])
            mean.x_tran[i] = np.average([k.x_tran[i] for k in l_pred])
            cov1 = np.sqrt(np.sum([(mean.s[i] - k.s[i]) ** 2 for k in l_pred]) / (M - 1))
            cov2 = np.sqrt(np.sum([(mean.x_tran[i] - k.x_tran[i]) ** 2 for k in l_pred]) / (M - 1))
            mean.sey_cov.append(np.array([[cov1, 0], [0, cov2]])[:2, :2].flatten())
            mean.e_psi[i] = np.average([k.e_psi[i] for k in l_pred])
            mean.v_long[i] = np.average([k.v_long[i] for k in l_pred])

        mean.s = array.array('d', mean.s)
        mean.x_tran = array.array('d', mean.x_tran)
        mean.sey_cov = array.array('d', np.array(mean.sey_cov).flatten())
        mean.e_psi = array.array('d', mean.e_psi)
        mean.v_long = array.array('d', mean.v_long)
        return mean

    def get_true_prediction_par(self, ego_state: VehicleState, target_state: VehicleState,
                                ego_prediction: VehiclePrediction, track: RadiusArclengthTrack, M=3):
        """
        Rather than evolving the next state from the mean of previous state,
        this function samples multiple trajectories and calculates mean + covariance from that
        input:
            ego_state: VehicleState of ego vehicle
            target_state: VehicleState of tar vehicle (the one that is to be predicted)
            ego_prediction: VehiclePrediction, prediction of ego vehicle, used to get
                                tar_prediction
            track: Simulation track, needed to get track curvature at future predicted states
            M: number of samples to draw and average. Heavily influences computational time
        """
        # Set GP model to eval-mode
        self.set_evaluation_mode()
        # draw M samples
        preds = self.sample_traj_gp_par(ego_state, target_state, ego_prediction, track, M)
        # numeric mean and covariance calculation.
        pred = self.mean_and_cov_from_list(preds, M)
        pred.t = ego_state.t
        return pred

    def get_multi_prediction_par(self, ego_state: VehicleState, target_state: VehicleState,
                                ego_predictions: List[VehiclePrediction], track: RadiusArclengthTrack, M=3):
        """
        Adapt true_prediciton_par to return predictions for multiple hypotheses, passed in as a List of ego_predictions
        Rather than evolving the next state from the mean of previous state,
        this function samples multiple trajectories and calculates mean + covariance from that
        input:
            ego_state: VehicleState of ego vehicle
            target_state: VehicleState of tar vehicle (the one that is to be predicted)
            ego_prediction: VehiclePrediction, prediction of ego vehicle, used to get
                                tar_prediction
            track: Simulation track, needed to get track curvature at future predicted states
            M: number of samples to draw and average. Heavily influences computational time
        """
        # Set GP model to eval-mode
        self.set_evaluation_mode()
        # draw M samples
        preds = self.sample_multi_traj_gp_par(ego_state, target_state, ego_predictions, track, M)
        pred = []
        # numeric mean and covariance calculation.
        for i in range(len(ego_predictions)):
            preds_ = []
            for j in range(M):
                preds_.append(preds[j+i])
            pred.append(self.mean_and_cov_from_list(preds_, M))
            pred[-1].t = ego_state.t
        return pred

    def get_UKF_prediction_par(self, ego_state: VehicleState, target_state: VehicleState,
                               ego_prediction: VehiclePrediction, track: RadiusArclengthTrack):
        """
        Rather than evolving the next state from the mean of previous state,
        this function samples sigma points to efficiently approximate thr true
        trajectory and account for uncertainty propagation
        input:
            ego_state: VehicleState of ego vehicle
            target_state: VehicleState of tar vehicle (the one that is to be predicted)
            ego_prediction: VehiclePrediction, prediction of ego vehicle, used to get
                                tar_prediction
            track: Simulation track, needed to get track curvature at future predicted states
            M: number of samples to draw and average. Heavily influences computational time
        """
        # Set GP model to eval-mode
        self.set_evaluation_mode()
        preds = self.UKF_traj_gp_par(ego_state, target_state, ego_prediction, track)
        return preds

    def convert_to_tensor(self, sample: Sample):
        """
            (For single sample)
            The GP will take as input EV + TV state
            The simplified state (using parametric position and velocity) will be constructed as:
                [s, x_tran, e_psi, v_long]^T for both ego and TV states

            Returns:
                (x_n : torch.tensor, y_n : torch.tensor) input and output for a single sample
        """
        ego_state = sample.input[0]
        tv_state = sample.input[1]
        dtv = sample.output
        dtv_gp = torch.tensor([dtv.p.s, dtv.p.x_tran, dtv.p.e_psi, dtv.v.v_long, dtv.w.w_psi])

        input, output = self.state_to_tensor(ego_state, tv_state), dtv_gp
        if input.shape[0] != self.input_size or output.shape[0] != self.output_size:
            raise RuntimeError(
                f"Input or output size is incorrect: input_size: {self.input_size} given: {input.shape[0]}"
                f"output_size: {self.output_size} given: {output.shape[0]}")
        return input, output

    def state_to_tensor(self, ego_state: VehicleState, tv_state: VehicleState):
        """
        Format states to GP input
        """
        ego_state_gp = torch.tensor([ego_state.p.x_tran-tv_state.p.x_tran, ego_state.p.e_psi, ego_state.v.v_long])
        tv_state_gp = torch.tensor(
            [tv_state.p.s - ego_state.p.s, tv_state.p.x_tran, tv_state.p.e_psi, tv_state.v.v_long, tv_state.w.w_psi,
             tv_state.lookahead.curvature[0], tv_state.lookahead.curvature[1], tv_state.lookahead.curvature[2]])
        return torch.cat((ego_state_gp, tv_state_gp))

    def state_to_tensor_vec(self, ego_state, tv_state):
        """
        Format states (in vector form) to GP input
        """
        ego_state_gp = torch.tensor([ego_state[1] - tv_state[1], ego_state[2], ego_state[3]])
        tv_state_gp = torch.tensor(
            [tv_state[0] - ego_state[0], tv_state[1], tv_state[2], tv_state[3], tv_state[4],
             tv_state[5], tv_state[6], tv_state[7]])
        return torch.cat((ego_state_gp, tv_state_gp))

    def set_evaluation_mode(self):
        self.model.eval()
        self.likelihood.eval()

    def save_model(self, name):
        model_to_save = dict()
        model_to_save['model'] = self.model
        model_to_save['likelihood'] = self.likelihood
        model_to_save['in_size'] = self.input_size
        model_to_save['out_size'] = self.output_size
        model_to_save['model_class'] = self.model_class
        model_to_save['mean_x'] = self.means_x
        model_to_save['std_x'] = self.stds_x
        model_to_save['mean_y'] = self.means_y
        model_to_save['std_y'] = self.stds_y
        model_to_save['independent'] = self.independent

        pickle_write(model_to_save, os.path.join(model_dir, name + '.pkl'))
        print('Successfully saved model', name)

    def load_model(self, name):
        model = pickle_read(os.path.join(model_dir, name + '.pkl'))
        self.model = model['model']
        self.likelihood = model['likelihood']
        self.input_size = model['in_size']
        self.output_size = model['out_size']
        self.model_class = model['model_class']
        self.means_x = model['mean_x']
        self.means_y = model['mean_y']
        self.stds_x = model['std_x']
        self.stds_y = model['std_y']
        # self.independent = model['independent'] TODO uncomment
        self.independent = True
        print('Successfully loaded model', name)

    def load_model_from_object(self, model):
        self.model = model['model']
        self.likelihood = model['likelihood']
        self.input_size = model['in_size']
        self.output_size = model['out_size']
        self.model_class = model['model_class']
        self.means_x = model['mean_x']
        self.means_y = model['mean_y']
        self.stds_x = model['std_x']
        self.stds_y = model['std_y']
        # self.independent = model['independent'] TODO uncomment
        self.independent = True
