from gas_storage_model import Forward, StoragePortfolio

if __name__ == "__main__":

    jan_24 = Forward(price=100, contract_size=5000, expiration_date="2023-01-01")
    feb_24 = Forward(price=80, contract_size=5000, expiration_date="2023-02-01")
    mar_24 = Forward(price=70, contract_size=5000, expiration_date="2023-03-01")
    apr_24 = Forward(price=60, contract_size=5000, expiration_date="2023-04-01")
    may_24 = Forward(price=50, contract_size=5000, expiration_date="2023-05-01")
    jun_24 = Forward(price=40, contract_size=5000, expiration_date="2023-06-01")
    jul_24 = Forward(price=40, contract_size=5000, expiration_date="2023-07-01")
    aug_24 = Forward(price=50, contract_size=5000, expiration_date="2023-08-01")
    sep_24 = Forward(price=60, contract_size=5000, expiration_date="2023-09-01")
    oct_24 = Forward(price=70, contract_size=5000, expiration_date="2023-10-01")
    nov_24 = Forward(price=80, contract_size=5000, expiration_date="2023-11-01")
    dec_24 = Forward(price=100, contract_size=5000, expiration_date="2023-12-01")

    port = StoragePortfolio()
    port.update_available_contracts(
        [jan_24, feb_24, mar_24, apr_24, may_24, jun_24, jul_24, aug_24, sep_24, oct_24, nov_24, dec_24])

    port.add_position(jan_24, direction='long', contracts=2)
    port.add_position(feb_24, direction='long')

    # port.close_position('feb_24')  # todo - need to update this function

    g = port.get_combinations()
    for i in g:
        print(i)
