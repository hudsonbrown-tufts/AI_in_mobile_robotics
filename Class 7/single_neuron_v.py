"""One neuron + ReLU learning a V-shaped data set, with PyTorch.

Model  = y_hat = ReLU(w * x + b)          (one neuron, one ReLU activation)
Data   = y = |x| + noise, x in [-2, 2]    (the "V")
Loss   = mean squared error, trained with gradient descent (Adam)

A live plot shows the data, the neuron's current output, and the loss
dropping over time.

Note: a single ReLU neuron is a "hinge" (flat on one side, a straight line on
the other), so it can never fit the whole V. Depending on the random start
(--seed) it settles on one arm of the V (seed 0), or a nearly flat line at
y ~= 1, the average height (seeds 1-3), or it can "die" if the ReLU starts out
off for every x so no gradient flows (seed 4). Run with --neurons 2 to give it a hidden layer of 2 ReLU neurons,
which is enough to fit the whole V: |x| = ReLU(x) + ReLU(-x).

Usage:
    python single_neuron_v.py                 # one neuron, live plot
    python single_neuron_v.py --neurons 2     # 2 hidden ReLU neurons -> full V
    python single_neuron_v.py --no-plot       # just print the loss
"""

import argparse

import torch
from torch import nn

NUM_POINTS = 100
NOISE = 0.1
EPOCHS = 2000
LEARNING_RATE = 0.01
PLOT_EVERY = 20  # epochs between plot updates


def make_v_data(num_points=NUM_POINTS, noise=NOISE):
    x = torch.linspace(-2, 2, num_points).unsqueeze(1)  # shape (N, 1)
    y = x.abs() + noise * torch.randn_like(x)
    return x, y


def make_model(neurons):
    if neurons == 1:
        # One neuron, one ReLU: y_hat = ReLU(w*x + b)
        return nn.Sequential(nn.Linear(1, 1), nn.ReLU())
    # Hidden layer of ReLU neurons, then a linear output neuron to add them up.
    return nn.Sequential(nn.Linear(1, neurons), nn.ReLU(), nn.Linear(neurons, 1))


def describe(model):
    if len(model) == 2:
        w = model[0].weight.item()
        b = model[0].bias.item()
        return f"y = ReLU({w:+.3f} x {b:+.3f})"
    return f"{model[0].out_features} hidden ReLU neurons"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neurons", type=int, default=1, help="number of ReLU neurons (default 1)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true", help="don't open the live plot")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x, y = make_v_data()
    model = make_model(args.neurons)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.no_plot:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax_fit.scatter(x, y, s=12, alpha=0.6, label="V data")
        (fit_line,) = ax_fit.plot(x, model(x).detach(), "r-", lw=2, label="model")
        ax_fit.set_xlabel("x")
        ax_fit.set_ylabel("y")
        ax_fit.set_ylim(-0.5, 2.5)
        ax_fit.legend(loc="upper center")
        (loss_line,) = ax_loss.plot([], [])
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("MSE loss")
        ax_loss.set_yscale("log")
        fig.tight_layout()

    losses = []
    for epoch in range(1, args.epochs + 1):
        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % PLOT_EVERY == 0 or epoch == 1:
            if epoch % (PLOT_EVERY * 10) == 0 or epoch == 1:
                print(f"epoch {epoch:5d}  loss {loss.item():.4f}  {describe(model)}")
            if not args.no_plot:
                if not plt.fignum_exists(fig.number):
                    break  # window closed
                fit_line.set_ydata(model(x).detach())
                ax_fit.set_title(f"epoch {epoch}: {describe(model)}")
                loss_line.set_data(range(1, len(losses) + 1), losses)
                ax_loss.relim()
                ax_loss.autoscale_view()
                plt.pause(0.001)

    print(f"final loss {losses[-1]:.4f}  {describe(model)}")
    if not args.no_plot and plt.fignum_exists(fig.number):
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
