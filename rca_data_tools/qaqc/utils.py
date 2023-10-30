
def coerce_qartod_executed_to_int(ds):

    qartod_executed_vars = [var for var in ds.variables if 'qartod_executed' in var]
    for var in qartod_executed_vars:
        executed_tests = ds[var].tests_executed.replace(' ', '').split(',')
    
        for i, test in enumerate(executed_tests):
            test_var_name = f"{var}_{test}"
            ds[test_var_name] = ds[var].str[i].astype(int)
        
        ds = ds.drop(var)

    return ds
        