import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng


class CoinGame:
    def __init__(self,
                 num_steps: int = 100,
                 reward_multiplier: float = 3.0,
                 loss_multiplier: float = 0.0,
                 starting_money: int = 10,
                 reinvestment_pct: float = 0.5,
                 num_trials: int = 15,
                 strategy: str = "reinvest_pct",
                 anti_corr: bool = False
                 ):
        self.num_steps = num_steps
        self.reward_mutliplier = reward_multiplier
        self.loss_multiplier = loss_multiplier
        self.starting_money = starting_money
        self.balance = None
        self.strategy = strategy
        self.reinvestment_pct = reinvestment_pct
        self.num_trials = num_trials
        self.rg = default_rng()
        self.anti_corr = anti_corr

    def coin_toss(self, amount: int, anti_corr: bool = None) -> float:
        if anti_corr is None:
            anti_corr = self.anti_corr
        if anti_corr:
            return amount * (self.reward_mutliplier + self.loss_multiplier) / 2
        coin_toss = self.rg.integers(2)
        if coin_toss == 0:
            return amount * self.loss_multiplier - amount
        else:
            return amount * self.reward_mutliplier - amount

    def generate_a_rollout(self) -> list:
        self.balance = float(self.starting_money)
        rollout = list()
        rollout.append(self.balance)
        for i in range(self.num_steps):
            amount = self.get_amount()
            reward = self.coin_toss(amount)
            self.balance += reward
            rollout.append(self.balance)
            if self.balance <= 1:
                break
            if self.strategy == "reinvest_pct":
                if int(self.balance * self.reinvestment_pct) < 1:
                    break
        return rollout

    def get_amount(self):
        if self.strategy == "reinvest_all":
            return self.reinvest_all()
        elif self.strategy == "reinvest_pct":
            return self.reinvest_pct()

    def reinvest_all(self):
        return self.balance

    def reinvest_pct(self, pct: float = None) -> int:
        if pct is None:
            pct = self.reinvestment_pct
        return int(self.balance * pct)

    def generate_rollouts(self, f: float = None, verbose: int = 1, plot: bool = True):
        if f is not None:
            self.reinvestment_pct = f
        trials = dict()
        for i in range(self.num_trials):
            trial = self.generate_a_rollout()
            trials[f'trial{i + 1}'] = trial
        df = pd.DataFrame.from_dict(trials, orient="index")
        df = df.transpose()
        if verbose > 0:
            print(df)
        if plot:
            for i in range(self.num_trials):
                # df[f"trial{i+1}"].plot(grid=True, label=f"trial{i+1}")
                df[f"trial{i + 1}"].plot(grid=True, label=f"trial{i + 1}", legend=True)
            plt.show()
        if verbose > 0:
            last_row = np.nan_to_num(df.tail(1).to_numpy()[0])
            print(last_row)
            print(f"Max reward on last row, mlns: {max(last_row) / 1000000}")
            rows_alive = sum([True for trial in last_row if trial != 0.])
            print(f"% rows alive: {rows_alive / self.num_trials * 100}")
        if verbose > 1:
            gm_df = df.pct_change() + 1
            gm_df = gm_df.fillna(1)
            print(gm_df)
            gmean = stats.gmean(gm_df, axis=0)
            gmean = np.nan_to_num(gmean)
            print(list(gmean))
            print(f"Max geomean: {np.max(gmean)}")
            print(f"Mean geomean: {np.mean(gmean)}")
        return df

    def looking_for_optimal_f(self):
        x = np.linspace(0.01, 1.0, 199)
        f_geomeans = list()
        for f in x:
            df = self.generate_rollouts(f=f, verbose=0, plot=False)
            gm_df = df.pct_change() + 1
            gm_df = gm_df.fillna(1)
            gmean = stats.gmean(gm_df, axis=0)
            gmean = np.nan_to_num(gmean)
            geomean = np.mean(gmean)
            f_geomeans.append((f, geomean))
        df = pd.DataFrame(f_geomeans)
        df = df.set_index([0])
        print(df)
        # df.plot(x=df[0], y=df[1], kind="scatter", grid=True, legend=True)
        df.plot(grid=True, legend=True)
        opt_f = df.idxmax(axis=0)
        print(f"Optimal f is {float(opt_f)}")
        plt.show()


if __name__ == '__main__':
    c = CoinGame(reinvestment_pct=.5,
                 reward_multiplier=3.,
                 num_trials=15,
                 starting_money=10,
                 # anti_corr=True,
                 num_steps=100
                 )
    c.generate_rollouts(verbose=1)
    # c.looking_for_optimal_f()
