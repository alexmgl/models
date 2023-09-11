def carbon_methodology_ldc(fuel_vol):

    """

    Functon converts fuel energy content to carbon emissions.

    Energy content input is MM.BTU or a Million British Thermal Units.

    Function returns CO2 (Million Tonnes).

    """

    input_one = 948170  # BTU to Gigajoule
    input_two = 49.68  # Gigajoule to C02 kg (S&P use 56.1 LHV)
    unit_scalar = 1000  # tonnes of c02

    btu_to_gigajoule = fuel_vol / input_one
    gigajoule_to_c02 = btu_to_gigajoule * input_two
    co2_emissions = gigajoule_to_c02 / unit_scalar

    return co2_emissions  # output is millions of tonnes of CO2


def carbon_methodology_s_and_p():
    pass


if __name__ == '__main__':
    carbon_methodology_ldc(fuel_vol=1)
