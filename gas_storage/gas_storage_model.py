import datetime
import numpy as np
import itertools
import pandas as pd
import copy

""" Strategy: Which gas contracts should I buy at which periods and in what quantity in order to maximise profit """

unit_commitment = 'https://medium.com/@AlainChabrier/unit-commitment-cd567add409b#:~:text=The%20Unit%20Commitment%20problem%20answers,use%3F%20(yes%20or%20no)'
gas_matlab = 'https://www.youtube.com/watch?v=Uu2JHzgeDUU' very good 
cme_gas_spreads = 'https://www.youtube.com/watch?v=JII3LGj4xmo' good basic intro 
commodity_models_harvard = 'https://www.youtube.com/watch?v=nmehlS-8b3Y'
presentation_on_storage = 'https://www.youtube.com/watch?v=Oh56GBofz8U'
gas_storage_explained = 'https://www.youtube.com/watch?v=qso-h6Ckgtw'
gas_storage_explained_2 = 'https://www.youtube.com/watch?v=rTE9FYiy4-s'
phd_thesis_defense = 'https://www.youtube.com/watch?v=irR-aD7Cvqw'
long_term_energy_markets = 'https://www.youtube.com/watch?v=yTxrLAdapJw&t=1s'

class StoragePortfolio:

    # todo - need to be able to read in forward prices for pricing (useful to see the changing prices over time, e.g. 3d plot)
    # todo - by reading in option prices we can calculate the implied volatility 

    # todo - need to have a model that takes in forward curve and estimates correlations and volatilities
    # todo - show volatility term structure

    def __init__(self):

        self.start_date = "2023-01-01"
        seld.end_date = "2023-12-31"

        self.portfolio = []
        self.available_contracts = set()

        self.discount_rate = None # todo
        self.liquidity_spread = None # todo - maybe already called transaction cost

        self.no_instruments = 0
        self.realised_pnl = 0

        self.start_volume = 0
        self.end_volume = 0

        self.max_capacity = 1000000  # max working gas volume mm.btu
        self.min_capacity = 0  # min working gas volume 

        self.max_injection_flow_rate_day = 5000  # todo - need to make constraints variable (e.g. at 0% full, withdraw 5000, but at 90% we may only withdraw 3000 per day)
        self.max_withdrawal_flow_rate_day = 5000

        self.flow_ratchets = None
        self.variable_injection = None  # injection unit cost (if daily, we need to work out monthly because we are dealing with monthly contracts)
        self.variable_withdrawal = None  # withdrawl unit cost 

        self.storage_df = pd.DataFrame(columns=['dt', 'injection_mmbtu', 'withdrawl_mmbtu', 'level_mmbtu'])

        # todo 
        self.bid_ask_spread = None 
        self.lot_size = None 

    # todo - need to be able to specify lots
    def add_position(self, forward, direction, contracts=1):

        if self.__validate_new_position():

            if not direction in ['long', 'short']:
                raise ValueError("Direction must be 'long' or 'short'.")

            forward = copy.deepcopy(forward)
            forward.direction = direction
            forward.volume = forward.contract_size * contracts

            self.portfolio.append(forward)
            self.no_instruments = len(self.portfolio)

        else:
            raise ValueError('Cannot add position to portfolio due to operational constraints.')

    def __validate_new_position(self):  # todo

        return self

    def update_available_contracts(self, contracts):

        for i in contracts:
            if i not in self.available_contracts:
                self.available_contracts.add(i)

        return self.available_contracts

    # need to be able to specify how many contracts to close
    def close_position(self, *args):

        items_to_remove = []
        for i in args:

            if i in self.portfolio:

                close_pnl = i.calculate_payoff()
                self.realised_pnl += close_pnl

                items_to_remove.append(i)

            else:
                raise ValueError(f"Instrument not in portfolio, thus cannot be removed.")

        self.portfolio = [item for item in self.portfolio if item not in items_to_remove]
        self.no_instruments = len(self.portfolio)

    # placeholder
    def delta_hedge(self):
        pass

    # placeholder
    def get_combinations(self):

        time_spreads = [sorted(list(i), key=lambda x: x.expiration_dt) for i in
                        itertools.combinations(self.available_contracts, 2)]

        final = []
        for i in time_spreads:
            long, short = copy.deepcopy(i[0]), copy.deepcopy(i[1])

            long.direction = 'long'
            short.direction = 'short'

            position = {'price_delta': short.current_price - long.current_price,
                        'time_delta': (short.expiration_dt - long.expiration_dt).days,
                        'contracts': [long, short]}

            final.append(position)

        return sorted(final, key=lambda x: (-x['price_delta'], x['time_delta']))

    def calculate_pnl(self):

        total_pnl = 0

        for forward in self.portfolio:

            if forward.direction == 'long':
                pnl = (forward.current_price - forward.price) * forward.volume

            elif forward.direction == 'short':
                pnl = (forward.price - forward.current_price) * forward.volume

            else:
                raise ValueError(f"Invalid direction '{forward.direction}' for Forward instance")

            total_pnl += pnl

        return total_pnl + self.realised_pnl


class Forward:
    instance_no = 0

    def __init__(self, price, contract_size, expiration_date, sigma=0.3):

        self.direction = None

        self.price = price
        self.current_price = price

        self.contract_size = contract_size
        self.volume = contract_size
        self.initial_cashflow = self.price * self.volume

        self.open_position = True

        self.expiration_dt = datetime.datetime.strptime(expiration_date, "%Y-%m-%d")
        self.contract_name = datetime.datetime.strftime(self.expiration_dt, "%b_%y").lower()

        self.sigma = sigma

        # show number of instances
        self.__class__.__add_instance()
        self.instance_no = self.instance_no

    def __repr__(self):
        return f'(Expiry: {self.expiration_dt.date()}, Price: {self.current_price}, Volume: {self.volume}, Dir: {self.direction})'

    @property
    def sigma_squared(self):
        return self.sigma ** 2

    @sigma_squared.setter
    def sigma_squared(self, value):
        self.sigma = np.sqrt(value)

    def calculate_payoff(self):

        """
        Calculate the payoff of the forward contract based on the current market price.

        Returns:
            float: The payoff of the forward contract.
        """

        if self.direction == 'long':
            return (self.current_price - self.price) * self.volume
        elif self.direction == 'short':
            return (self.price - self.current_price) * self.volume
        else:
            raise ValueError("Direction must be 'long' or 'short'.")

    @classmethod
    def __add_instance(cls):
        cls.instance_no += 1

    def mark_position(self, value):
        self.current_price = value

    def close_position(self):
        self.open_position = False

    @property
    def delivery_vol(self):

        next_month = self.expiration_dt.replace(day=1) + datetime.timedelta(days=32)
        first_day_of_next_month = next_month.replace(day=1)

        # Calculate the last day of the current month
        last_day_of_current_month = first_day_of_next_month - datetime.timedelta(days=1)

        date_range = pd.date_range(self.expiration_dt, last_day_of_current_month)

        direction_mapping = {'long': 1, 'short': -1}
        a = direction_mapping.get(self.direction, 0)

        df = pd.DataFrame({'dt': date_range, 'vol': a * self.volume})

        return df


# Placeholder for storage asset model used for Deep Q learning.
class StorageAssetModel:
    instantiations = 0

    def __init__(self):

        self.start_str = '2023-01-01'
        self.start_dt = datetime.datetime.strptime(self.start_str, "%Y-%m-%d")
        self.day = 0

        # Initializations
        self.withdrawal_fuel_losses = 0
        self.injection_fuel_losses = 0

        self.max_injection_flow_rate_day = 5000
        self.max_withdrawal_flow_rate_day = 5000

        self.flow_ratchets = None
        self.variable_injection = None
        self.variable_withdrawal = None

        self.min_inventory_level = 0
        self.max_inventory_level = None

        # Current state
        self.current_inventory = 0

        # Contracts
        self.contracts = []

        # class information
        self.__class__.__inc_instantiations()

    @classmethod
    def __inc_instantiations(cls):

        cls.instantiations += 1

    @property
    def current_inventory_pct(self):
        """Get the current inventory as a percentage of the maximum inventory level."""
        return self.current_inventory / self.max_inventory_level

    @current_inventory_pct.setter
    def current_inventory_pct(self, value):
        """Set the current inventory as a percentage of the maximum inventory level."""

        self.current_inventory = self.max_inventory_level * value

    def inject_gas(self, volume):
        """
        Inject gas into the storage facility.

        Args:
            volume (float): Volume of gas to inject in MMBtu.
        """
        if self.variable_injection is not None:
            # Implement variable injection logic here if applicable
            pass
        else:
            # Standard injection logic
            new_inventory = self.current_inventory + volume
            if new_inventory > self.max_inventory_level:
                raise ValueError(
                    f"Injection exceeds maximum storage capacity. Current level is {self.current_inventory:,} MMBtu.")
            elif volume > self.max_injection_flow_rate_day:
                raise ValueError(
                    f"Injection rate exceeds daily limit. Daily injection limit is {self.max_injection_flow_rate_day:,} MMBtu")
            else:
                self.current_inventory = new_inventory
                self.day += 1

    def withdraw_gas(self, volume):
        """
        Withdraw gas from the storage facility.

        Args:
            volume (float): Volume of gas to withdraw in MMBtu.
        """
        if self.variable_withdrawal is not None:
            # Implement variable withdrawal logic here if applicable
            pass
        else:
            # Standard withdrawal logic
            new_inventory = self.current_inventory - volume
            if new_inventory < self.min_inventory_level:
                raise ValueError(
                    f"Withdrawal breaches minimum storage level. Current level is {self.current_inventory:,} MMBtu.")
            elif volume > self.max_withdrawal_flow_rate_day:
                raise ValueError(
                    f"Withdrawal rate exceeds daily limit. Daily withdrawl limit is {self.max_withdrawal_flow_rate_day:,} MMBtu.")
            else:
                self.current_inventory = new_inventory
                self.day += 1

    def check_inventory_level(self):
        """
        Check the current inventory level as a percentage of maximum capacity.

        Returns:
            float: Current inventory level as a percentage.
        """

        return self.current_inventory_pct

    def __is_empty(self):
        """
        Check if the storage facility is empty.

        Returns:
            bool: True if empty, False otherwise.
        """

        return self.current_inventory <= 0

    def __is_full(self):
        """
        Check if the storage facility is full.

        Returns:
            bool: True if full, False otherwise.
        """

        return self.current_inventory >= self.max_inventory_level

    def set_ratchets(self, ratchets):
        """
        Set volume-dependent flow rates (ratchets).

        Args:
            ratchets (dict): A dictionary of ratchets, e.g., {'injection': [(50, 8000), (100, 6000)], 'withdrawal': [(60, 5000)]}.
        """
        self.flow_ratchets = ratchets

    def calculate_injection_rate(self):
        """
        Calculate the current injection rate based on inventory level and ratchets.

        Returns:
            float: Current injection rate in MMBtu/day.
        """
        if self.flow_ratchets and 'injection' in self.flow_ratchets:
            # Implement ratchet-based injection rate calculation here
            pass
        else:
            # Default injection rate calculation
            return self.max_injection_flow_rate_day

    def calculate_withdrawal_rate(self):
        """
        Calculate the current withdrawal rate based on inventory level and ratchets.

        Returns:
            float: Current withdrawal rate in MMBtu/day.
        """
        if self.flow_ratchets and 'withdrawal' in self.flow_ratchets:
            # Implement ratchet-based withdrawal rate calculation here
            pass
        else:
            # Default withdrawal rate calculation
            return self.max_withdrawal_flow_rate_day

    def calculate_days_to_fill(self, pct_level=1):
        """
        Calculate the number of days require to fill storage to desired level.
        Args:
            pct_level (float): Float between 0 and 1, representing the desired level of storage.

        Returns:
            int: Representing days.
        """

        if self.current_inventory_pct >= pct_level:
            raise ValueError(f"Inventory is already at the desired level.")

        return (self.max_inventory_level - self.current_inventory) / self.max_injection_flow_rate_day

    def calculate_days_to_empty(self, pct_level=0):
        """
        Calculate the number of days require to empty storage to desired level.
        Args:
            pct_level (float): Float between 0 and 1, representing the desired level of storage.

        Returns:
            int: Representing days.
        """

        if self.current_inventory_pct <= pct_level:
            raise ValueError(f"Inventory is already at the desired level.")

        return self.current_inventory / self.max_withdrawal_flow_rate_day

    def add_contract(self):
        pass

    def sim(self):
        pass

def model_calibration():
    # forward curve statistics (volatility)
    # monthly volatilities for annualised histoiric vol
    # time varying correlations
    # we can use pca to reduce 12x12 sigma (correlation/covariance) matrix to (for example 3x3)
    pass

# todo 
def valuation_intrinsic(forward_curve):
    # use a linear optimiser to solve for storage optimisation against a forward curve
    intrinsic_val = 0
    discount_intrinsic_val = 0
    # todo - operating and trading constraints e.g. injection and trading lot size, discount rate 
    return intrinsic_val, discount_intrinsic_val

def valuation_spread_option_basket():
    pass

def valuation_rolling_intrinsic(sims=1000):
    pass

def valuation_spot_optimisation():
    pass

if __name__ == '__main__':
    pass
