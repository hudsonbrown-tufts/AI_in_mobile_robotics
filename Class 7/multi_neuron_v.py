"""More neurons and layers so the network fits the whole V, with PyTorch.

Model  = x -> [Linear -> ReLU] x layers -> Linear -> y_hat
Data   = y = |x| + noise, x in [-2, 2]    (same V as single_neuron_v.py)
Loss   = mean squared error, trained with gradient descent (Adam)

One ReLU neuron is a single hinge and can't make a V (see single_neuron_v.py).
With several hidden ReLU neurons the output neuron adds their hinges together,
and two hinges pointing opposite ways are already a V: |x| = ReLU(x) + ReLU(-x).
Extra neurons make it robust to a bad random start (some can die and the rest
still cover both arms), and extra layers let it build bendier shapes.

Usage:
    python multi_neuron_v.py                  # 2 hidden layers of 8 neurons
    python multi_neuron_v.py --hidden 16      # 1 hidden layer of 16 neurons
    python multi_neuron_v.py --hidden 16 16 16
    python multi_neuron_v.py --no-plot        # just print the loss
"""

import argparse

import torch
from torch import nn

from single_neuron_v import make_v_data

HIDDEN = [8, 8]  # neurons in each hidden layer
EPOCHS = 2000
LEARNING_RATE = 0.01
PLOT_EVERY = 20  # epochs between plot updates


def make_model(hidden):
    layers = []
    width_in = 1
    for width in hidden:
        layers += [nn.Linear(width_in, width), nn.ReLU()]
        width_in = width
    layers.append(nn.Linear(width_in, 1))  # output neuron adds up the hinges
    return nn.Sequential(*layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, nargs="+", default=HIDDEN,
                        help="neurons per hidden layer, e.g. --hidden 8 8 (default)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true", help="don't open the live plot")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x, y = make_v_data()
    model = make_model(args.hidden)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    shape = " x ".join(str(n) for n in args.hidden)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"hidden layers: {shape}  ({num_params} weights + biases)")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax_fit.scatter(x, y, s=12, alpha=0.6, label="V data")
        ax_fit.plot(x, x.abs(), "k--", lw=1, alpha=0.5, label="true |x|")
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
                print(f"epoch {epoch:5d}  loss {loss.item():.4f}")
            if not args.no_plot:
                if not plt.fignum_exists(fig.number):
                    break  # window closed
                fit_line.set_ydata(model(x).detach())
                ax_fit.set_title(f"hidden {shape}, epoch {epoch}, loss {loss.item():.4f}")
                loss_line.set_data(range(1, len(losses) + 1), losses)
                ax_loss.relim()
                ax_loss.autoscale_view()
                plt.pause(0.001)

    print(f"final loss {losses[-1]:.4f}  (noise alone gives ~{0.1 ** 2:.2f})")
    if not args.no_plot and plt.fignum_exists(fig.number):
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
